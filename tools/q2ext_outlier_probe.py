"""Q2-ext probe: gate/up input per-channel amax dump + LLM.int8-style mixed GEMM sim.

Outputs to ./workdir/q2ext/ (override: Q2EXT_OUT).
  q2ext_amax.npz : gu  (18,16,2048)   gate/up-input per-channel amax per frame
                   b1  (18,16,2048)   attn-input per-call amax
                   dn  (18,16,16384)  down-input per-channel amax
  q2ext_sim.npz  : GEMM-level mixed-precision sim metrics
Env: FVK_PI05_RTX_FORCE_BF16=1, FLASHRT_PI05_EMBED_NP=0, num_views=2, seq=522.
"""
import os, sys, json, ctypes, time
os.environ["FVK_PI05_RTX_FORCE_BF16"] = "1"
os.environ["FLASHRT_PI05_EMBED_NP"] = "0"
sys.path.insert(0, os.environ["FLASHRT_ROOT"])
_p = os.environ.get("OPENPI_SRC"); (_p and sys.path.insert(1, _p))
sys.path.insert(1, os.environ.get("OPENPI_SRC", os.getcwd()))

import numpy as np
import torch
import flash_rt.flash_rt_kernels as fvk
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

torch.backends.cuda.matmul.allow_tf32 = False  # fp32 GEMM must stay IEEE

CKPT = os.environ["PI05_CKPT_DIR"]
FRAMES_NPZ = os.environ.get("Q2EXT_FRAMES_NPZ", "./workdir/calib_task0_real.npz")
OUT = os.environ.get("Q2EXT_OUT", "./workdir/q2ext")
ENC_L, ENC_D, ENC_H = 18, 2048, 16384
SEQ = 0   # derived at runtime from observed encoder calls
NF = 16
KS = (8, 16, 32, 64, 128, 256)
CALIB_N = 8            # frames 0..7 select the static outlier sets
DN_FRAMES = (0, 5, 10, 15)


_mcpy = ctypes.CDLL("libcudart.so").cudaMemcpyAsync
_mcpy.restype = ctypes.c_int
_mcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                  ctypes.c_int, ctypes.c_void_p]
D2D = 3

log_f = None
def log(msg):
    print(msg, flush=True)
    if log_f is not None:
        log_f.write(msg + "\n"); log_f.flush()

def main():
    global log_f
    os.makedirs(OUT, exist_ok=True)
    log_f = open(os.path.join(OUT, "q2ext_run.log"), "w")
    fr = np.load(FRAMES_NPZ)
    img, wrist = fr["image"], fr["wrist_image"]
    try:
        prompt = str(fr["prompt"])
    except Exception:
        prompt = "put both moka pots on the stove"
    log(f"prompt={prompt!r} frames={img.shape}")

    pipe = Pi05TorchFrontendRtx(
        CKPT, num_views=2, num_steps=1, vision_pool_factor=1,
        vision_num_layers=27, use_cuda_graph=False)
    pipe.set_prompt(prompt)
    pipe.calibrate_with_real_data([{
        "image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "state": np.random.uniform(-1, 1, (8,)).astype(np.float32)}])
    log("frontend ready")
    pipe.infer({"image": np.ascontiguousarray(img[0]),
                "wrist_image": np.ascontiguousarray(wrist[0])})   # warmup
    p = pipe.pipeline or getattr(pipe, "_fixed_pipeline", None)
    assert p is not None, "pipeline not built after warmup"
    Wp = p.weights
    ptr2id = {}
    for i in range(ENC_L):
        for typ, key in (("B1", "encoder_norm_w_attn"),
                         ("B4", "encoder_norm_w_ffn")):
            t = Wp[key][i]
            key_ptr = t if isinstance(t, int) else int(t.data_ptr())
            ptr2id[key_ptr] = (typ, i)
    log(f"ptr map built: {len(ptr2id)} entries")

    gu = np.zeros((ENC_L, NF, ENC_D), np.float32)
    b1 = np.zeros((ENC_L, NF, ENC_D), np.float32)
    dn = np.zeros((ENC_L, NF, ENC_H), np.float32)
    X_store = [[None] * NF for _ in range(ENC_L)]   # gate/up inputs, bf16 cuda
    cnt = {"fi": -1, "b1": 0, "b4": 0, "gg": 0, "unk": 0}
    gg_i = [0] * NF
    orig_rms, orig_gg = fvk.rms_norm, fvk.gate_geglu


    def wrap_rms(x_ptr, w_ptr, out_ptr, seq, D, eps, stream=0):
        global SEQ
        orig_rms(x_ptr, w_ptr, out_ptr, seq, D, eps, stream=stream)
        if D != ENC_D or seq < 100:
            return
        if SEQ == 0:
            SEQ = seq
            log(f"SEQ observed = {seq}")
        if seq != SEQ:
            return
        fi = cnt["fi"]
        t = torch.empty(seq, D, dtype=torch.bfloat16, device="cuda")
        _mcpy(t.data_ptr(), out_ptr, seq * D * 2, D2D, ctypes.c_void_p(stream))
        per = t.float().abs().amax(0)
        tid = ptr2id.get(w_ptr if isinstance(w_ptr, int) else int(w_ptr.data_ptr()))
        if tid is None:
            cnt["unk"] += 1
            return
        typ, li = tid
        if typ == "B4":
            gu[li, fi] = per.cpu().numpy()
            X_store[li][fi] = t
            cnt["b4"] += 1
        else:
            b1[li, fi] = per.cpu().numpy()
            cnt["b1"] += 1

    def wrap_gg(gm_ptr, hid_ptr, out_ptr, total, stream=0):
        orig_gg(gm_ptr, hid_ptr, out_ptr, total, stream=stream)
        if SEQ == 0 or total != SEQ * ENC_H:
            return
        fi = cnt["fi"]
        if gg_i[fi] >= ENC_L:
            return
        fi = cnt["fi"]
        li = gg_i[fi]
        gg_i[fi] += 1
        t = torch.empty(total, dtype=torch.bfloat16, device="cuda")
        _mcpy(t.data_ptr(), out_ptr, total * 2, D2D, ctypes.c_void_p(stream))
        dn[li, fi] = t.view(SEQ, ENC_H).float().abs().amax(0).cpu().numpy()
        cnt["gg"] = sum(gg_i)


    fvk.rms_norm = wrap_rms
    fvk.gate_geglu = wrap_gg
    t0 = time.time()
    for fi in range(NF):
        cnt["fi"] = fi
        pipe.infer({"image": np.ascontiguousarray(img[fi]),
                    "wrist_image": np.ascontiguousarray(wrist[fi])})
        log(f"frame {fi} ({time.time()-t0:.1f}s b1={cnt['b1']} b4={cnt['b4']} "
            f"gg={cnt['gg']} unk={cnt['unk']})")
    fvk.rms_norm, fvk.gate_geglu = orig_rms, orig_gg
    log(f"capture: b1={cnt['b1']} b4={cnt['b4']} gg={cnt['gg']} unk={cnt['unk']}")
    MLP_L = sum(1 for li in range(ENC_L) if X_store[li][0] is not None)
    log(f"MLP-bearing layers captured: {MLP_L}")
    assert cnt["b4"] == MLP_L * NF, f"b4 count {cnt['b4']} vs MLP_L {MLP_L}"
    assert cnt["gg"] == MLP_L * NF, f"gg count {cnt['gg']} vs MLP_L {MLP_L}"
    assert cnt["b1"] == ENC_L * NF, f"b1 count {cnt['b1']}"

    np.savez(os.path.join(OUT, "q2ext_amax.npz"),
             gu=gu, b1=b1, dn=dn, seq=SEQ, prompt=prompt)
    log("amax npz saved")
    del b1

    # ---------------- concentration / persistence on amax dumps ----------------
    def conc(A, name):
        L, F, C = A.shape
        amax_f = A.max(axis=1)                        # (L,C) cross-frame max
        p50 = np.percentile(amax_f, 50, axis=1)       # (L,)
        m = np.maximum(0.0, amax_f / p50[:, None] - 1.0)
        mtot = m.sum(axis=1)
        order = np.argsort(-m, axis=1)
        cum = np.cumsum(np.take_along_axis(m, order, axis=1), axis=1)
        share = cum / np.maximum(mtot[:, None], 1e-12)
        top = {K: [round(float(v), 4) for v in share[:, K - 1]] for K in KS}
        cnts, pfrac, pmass = {}, {}, {}
        for tau in (3, 5, 10, 20):
            fl = A >= (tau * p50[:, None, None])
            anyf = fl.any(axis=1)
            n_any = anyf.sum(1)
            frac = fl.sum(1) / float(F)
            pers = (frac >= 0.8) & anyf
            cnts[tau] = [int(v) for v in n_any]
            cnts[f"pers_n_{tau}"] = [int(v) for v in pers.sum(1)]
            pfrac[tau] = [round(float(v), 3) for v in
                          pers.sum(1) / np.maximum(n_any, 1)]
            pm = np.where(pers, m, 0.0).sum(1)
            pmass[tau] = [round(float(v), 4) for v in
                          pm / np.maximum(mtot, 1e-12)]
        return dict(p50=[round(float(v), 4) for v in p50],
                    amax=[round(float(v), 3) for v in amax_f.max(1)],
                    ratio=[round(float(amax_f.max(1)[i]) / max(p50[i], 1e-9), 2)
                           for i in range(L)],
                    top=top, cnts=cnts, pfrac=pfrac, pmass=pmass)


    conc_res = {}
    for name, A in (("gu", gu), ("dn", dn)):
        nz = sum(1 for li in range(ENC_L) if A[li].sum() > 0)
        conc_res[name] = conc(A[:nz], name)
        log(f"{name}: layers with data = {nz}")
    agg = {K: round(float(np.mean([np.mean(conc_res[n]["top"][K])
                                   for n in conc_res])), 4) for K in KS}
    log("agg top-K share: " + json.dumps(agg))
    NL = conc_res["gu"]["p50"].__len__()
    log("gu per-layer p50/amax/ratio: " + json.dumps(
        {i: [conc_res["gu"]["p50"][i], conc_res["gu"]["amax"][i],
             conc_res["gu"]["ratio"][i]] for i in range(NL)}))

    # ---------------- GEMM-level mixed-precision sim ----------------
    ck = pipe._ckpt_bf16
    CK_G, CK_U, CK_DN = (ck["encoder_ffn_gate_w"], ck["encoder_ffn_up_w"],
                         ck["encoder_ffn_down_w"])
    log(f"W gate {tuple(CK_G.shape)} {CK_G.dtype} {CK_G.device}, "
        f"down {tuple(CK_DN.shape)} {CK_DN.dtype}")

    def cosrel(a, b):
        num = float((a * b).sum())
        den = float(a.norm() * b.norm()) + 1e-30
        return num / den, float((a - b).norm() / (a.norm() + 1e-30))

    def rowq(Xf, mask=None):
        Xs = Xf if mask is None else Xf[:, mask]
        s = Xs.abs().amax(1) / 127.0
        q = torch.clamp(torch.round(Xf / s[:, None]), -127, 127)
        return q, s


    def silu_mul(a, b):
        return torch.nn.functional.silu(a) * b

    KS_L = list(KS)
    gu_calib = gu[:, :CALIB_N, :].max(axis=1)
    score = gu_calib / np.percentile(gu_calib, 50, axis=1, keepdims=True)
    order = np.argsort(-score, axis=1)
    set64 = order[:, :max(KS_L)]
    log("set64 head: " + json.dumps(
        {int(l): set64[l, :8].astype(int).tolist() for l in range(ENC_L)}))

    sim = {}
    sim["cos_h_i8"] = np.zeros((ENC_L, NF), np.float32)
    sim["rel_h_i8"] = np.zeros((ENC_L, NF), np.float32)
    sim["cos_y_i8"] = np.zeros((ENC_L, NF), np.float32)
    sim["cos_h_mix"] = {}
    sim["rel_h_mix"] = {}
    sim["cos_y_mix"] = {}
    for K in KS_L:
        sim["cos_h_mix"][K] = np.zeros((ENC_L, NF), np.float32)
        sim["rel_h_mix"][K] = np.zeros((ENC_L, NF), np.float32)
        sim["cos_y_mix"][K] = np.zeros((ENC_L, NF), np.float32)

    sim["cos_dn_i8"] = np.zeros((ENC_L, NF), np.float32)
    sim["rel_dn_i8"] = np.zeros((ENC_L, NF), np.float32)
    sim["cos_dn_mix64"] = np.zeros((ENC_L, NF), np.float32)
    dn_calib = dn[:, :CALIB_N, :].max(axis=1)
    dn_score = dn_calib / np.percentile(dn_calib, 50, axis=1, keepdims=True)
    dn_order = np.argsort(-dn_score, axis=1)
    set64_dn = dn_order[:, :64]
    for l in range(MLP_L):
        Wg = CK_G[l].float()
        Wu = CK_U[l].float()
        Wd = CK_DN[l].float()
        Wgqs = Wg.abs().amax(0) / 127.0
        Wuqs = Wu.abs().amax(0) / 127.0
        Wdqs = Wd.abs().amax(0) / 127.0
        Wgq = torch.round(Wg / Wgqs[None, :])
        Wuq = torch.round(Wu / Wuqs[None, :])
        Wdq = torch.round(Wd / Wdqs[None, :])
        torch.cuda.synchronize()

        for f in range(NF):
            X = X_store[l][f].float()
            Yg = X @ Wg
            Yu = X @ Wu
            h_ref = silu_mul(Yg, Yu)
            Xq, s_full = rowq(X)
            yg = (Xq @ Wgq) * s_full[:, None] * Wgqs[None, :]
            yu = (Xq @ Wuq) * s_full[:, None] * Wuqs[None, :]
            h_i8 = silu_mul(yg, yu)
            sim["cos_h_i8"][l, f] = cosrel(h_ref, h_i8)[0]
            sim["rel_h_i8"][l, f] = cosrel(h_ref, h_i8)[1]
            sim["cos_y_i8"][l, f] = cosrel(Yg, yg)[0]

            for K in KS_L:
                mk = torch.from_numpy(set64[l, :K]).cuda()
                mask = torch.ones(ENC_D, dtype=torch.bool, device="cuda")
                mask[mk] = False
                Xq2, s2 = rowq(X, mask)
                rest = Xq2[:, mask]
                wg_m = Wgq[mask, :]
                wu_m = Wuq[mask, :]
                yg2 = (rest @ wg_m) * s2[:, None] * Wgqs[None, :]
                yu2 = (rest @ wu_m) * s2[:, None] * Wuqs[None, :]
                yg2 = yg2 + X[:, mk] @ Wg[mk, :]
                yu2 = yu2 + X[:, mk] @ Wu[mk, :]
                h2 = silu_mul(yg2, yu2)
                sim["cos_h_mix"][K][l, f] = cosrel(h_ref, h2)[0]
                sim["rel_h_mix"][K][l, f] = cosrel(h_ref, h2)[1]
                sim["cos_y_mix"][K][l, f] = cosrel(Yg, yg2)[0]

            if f in DN_FRAMES:
                H = h_ref
                Ydn = H @ Wd
                Hq, hs = rowq(H)
                ydn = (Hq @ Wdq) * hs[:, None] * Wdqs[None, :]
                cc = cosrel(Ydn, ydn)
                sim["cos_dn_i8"][l, f] = cc[0]
                sim["rel_dn_i8"][l, f] = cc[1]
                mkd = torch.from_numpy(set64_dn[l]).cuda()
                maskd = torch.ones(ENC_H, dtype=torch.bool, device="cuda")
                maskd[mkd] = 0
                Hq2, hs2 = rowq(H, maskd)
                ydn2 = (Hq2[:, maskd] @ Wdq[maskd, :]) * hs2[:, None] \
                    * Wdqs[None, :] + H[:, mkd] @ Wd[mkd, :]
                sim["cos_dn_mix64"][l, f] = cosrel(Ydn, ydn2)[0]
        torch.cuda.synchronize()
        log(f"layer {l} sim done")


    # ---------------- save sim + summary ----------------
    sim_npz = {"cos_h_i8": sim["cos_h_i8"], "rel_h_i8": sim["rel_h_i8"],
               "cos_y_i8": sim["cos_y_i8"],
               "cos_dn_i8": sim["cos_dn_i8"], "rel_dn_i8": sim["rel_dn_i8"],
               "cos_dn_mix64": sim["cos_dn_mix64"],
               "set64": set64, "set64_dn": set64_dn,
               "gu_calib_score": score, "seq": SEQ,
               "ks": np.array(KS_L)}
    for K in KS_L:
        sim_npz[f"cos_h_mix{K}"] = sim["cos_h_mix"][K]
        sim_npz[f"rel_h_mix{K}"] = sim["rel_h_mix"][K]
        sim_npz[f"cos_y_mix{K}"] = sim["cos_y_mix"][K]
    np.savez(os.path.join(OUT, "q2ext_sim.npz"), **sim_npz)
    log("sim npz saved")

    def agg_over_frames(key):
        M = sim[key]
        return {int(l): [round(float(M[l].mean()), 6),
                         round(float(M[l].min()), 6)] for l in range(ENC_L)}
    summary = {
        "cos_h_i8_mean_min_per_layer": agg_over_frames("cos_h_i8"),
        "rel_h_i8_mean_min_per_layer": agg_over_frames("rel_h_i8"),
        "cos_dn_i8_mean_min_per_layer": agg_over_frames("cos_dn_i8"),
        "cos_dn_mix64_mean_min_per_layer": agg_over_frames("cos_dn_mix64"),
    }
    for K in KS_L:
        summary[f"cos_h_mix{K}_mean_min"] = agg_over_frames(f"cos_h_mix{K}")
        summary[f"rel_h_mix{K}_mean_min"] = agg_over_frames(f"rel_h_mix{K}")
    with open(os.path.join(OUT, "q2ext_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log("summary saved")
    log("DONE")


if __name__ == "__main__":
    main()
