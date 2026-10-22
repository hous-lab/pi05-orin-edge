#!/usr/bin/env python3
"""Parity gate: openpi-BF16 (reference) vs FlashRT-INT8 (pool1/pool2), fixed obs + fixed noise."""
import os, sys, pathlib
os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
import numpy as np
import torch

sys.path.insert(0, os.environ.get("OPENPI_SRC", ""))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


PROMPT = "pick up the black bowl on the stove"
np.random.seed(123)
OBS_LIST = []
for k in range(3):
    imgs = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
    state = np.random.uniform(-1, 1, (8,)).astype(np.float32)
    OBS_LIST.append({"imgs": imgs, "state": state, "prompt": PROMPT})
NOISES = [torch.randn(10, 32, generator=torch.Generator().manual_seed(1000 + k)) for k in range(3)]

# ---------- openpi BF16 reference ----------
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.shared import image_tools as _it
from transformers import AutoTokenizer

cfg = Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, pytorch_compile_mode=None)
model = PI0Pytorch(cfg)
from safetensors.torch import load_file
sd = load_file(os.path.join(os.environ["PI05_CKPT_DIR"], "model.safetensors"))
sd.pop("paligemma_with_expert.paligemma.lm_head.weight", None)  # official bundle carries explicit tied copy; openpi ties it
miss, unexp = model.load_state_dict(sd, strict=False)
assert not [k for k in miss if "lm_head" not in k] and not unexp
model = model.to("cuda").eval().requires_grad_(False)

import sentencepiece
_sp = sentencepiece.SentencePieceProcessor(
    model_proto=open(os.path.join(os.environ["PI05_CKPT_DIR"], "assets/tokenizer/tokenizer.model"), "rb").read())
ns = __import__("json").load(open(os.path.join(os.environ["PI05_CKPT_DIR"], "norm_stats.json")))["norm_stats"]
q01_s = np.array(ns["state"]["q01"], dtype=np.float32)
q99_s = np.array(ns["state"]["q99"], dtype=np.float32)
q01_a = np.array(ns["actions"]["q01"], dtype=np.float32)
q99_a = np.array(ns["actions"]["q99"], dtype=np.float32)


def build_prompt_ids(prompt, state_norm_np):
    """openpi tokenizer.py:22-48 pi05 格式: state 256-bins 离散化进 prompt, add_bos, pad 到 200。"""
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    disc = np.digitize(state_norm_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    full = f"Task: {cleaned}, State: {' '.join(map(str, disc))};\nAction: "
    toks = _sp.encode(full, add_bos=True)
    ids = np.zeros(200, dtype=np.int64)
    mask = np.zeros(200, dtype=bool)
    n = min(len(toks), 200)
    ids[:n] = toks[:n]
    mask[:n] = True
    return ids, mask

def openpi_denoise_trace(imgs_u8, state8, prompt, noise):
    """复刻 sample_actions 循环, 捕获每步 v_t (归一化空间)。"""
    imgs = []
    for im in imgs_u8:
        tt = torch.from_numpy(im).float().cuda() / 255.0 * 2 - 1
        imgs.append(tt.permute(2, 0, 1)[None])
    st_n_np = ((state8 - q01_s) / (q99_s - q01_s + 1e-6) * 2.0 - 1.0).astype(np.float32)
    st_n = torch.from_numpy(st_n_np).cuda()[None]
    ids_np, mask_np = build_prompt_ids(prompt, st_n_np)
    ids = torch.from_numpy(ids_np).cuda()[None]
    mask = torch.from_numpy(mask_np).cuda()[None]
    class _Obs:
        pass
    o = _Obs()
    o.images = {k: v for k, v in zip(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), imgs)}
    o.image_masks = {k: torch.ones(1, dtype=torch.bool, device="cuda") for k in o.images}
    o.state = st_n
    o.tokenized_prompt = ids
    o.tokenized_prompt_mask = mask
    o.token_ar_mask = torch.zeros_like(ids)
    o.token_loss_mask = torch.zeros_like(ids, dtype=torch.bool)
    with torch.no_grad():
        pe, pm, am = model.embed_prefix(list(o.images.values()), list(o.image_masks.values()), o.tokenized_prompt, o.tokenized_prompt_mask)
        m4 = model._prepare_attention_masks_4d(make_att_2d_masks(pm, am))
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        pos = torch.cumsum(pm, 1) - 1
        (out_e, kv) = model.paligemma_with_expert.forward(attention_mask=m4, position_ids=pos,
            past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
        x = noise[None].cuda().clone()
        st_t = torch.from_numpy(st_n_np).cuda()[None]
        vs = []
        tsc = 1.0
        for _ in range(10):
            v = model.denoise_step(st_t, pm, kv, x, torch.full((1,), tsc, device="cuda"))
            vs.append(v[0, :, :7].float().cpu().numpy())
            x = x - 0.1 * v
            tsc -= 0.1
        raw = x[0].float().cpu().numpy()[:, :7]
    return vs, raw


def openpi_infer(imgs_u8, state8, prompt, noise):
    imgs = []
    for im in imgs_u8:
        t = torch.from_numpy(im).float().cuda() / 255.0 * 2 - 1  # HWC [-1,1]
        imgs.append(t.permute(2, 0, 1)[None])  # BCHW
    st = torch.from_numpy(state8).float().cuda()[None]
    st_n_np = ((state8 - q01_s) / (q99_s - q01_s + 1e-6) * 2.0 - 1.0).astype(np.float32)  # transforms.py:145
    st_n = torch.from_numpy(st_n_np).cuda()[None]
    ids_np, mask_np = build_prompt_ids(prompt, st_n_np)
    ids = torch.from_numpy(ids_np).cuda()[None]
    mask = torch.from_numpy(mask_np).cuda()[None]
    class _Obs:
        pass
    o = _Obs()
    o.images = {k: v for k, v in zip(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), imgs)}
    o.image_masks = {k: torch.ones(1, dtype=torch.bool, device="cuda") for k in o.images}
    o.state = st_n
    o.tokenized_prompt = ids
    o.tokenized_prompt_mask = mask
    o.token_ar_mask = torch.zeros_like(ids)
    o.token_loss_mask = torch.zeros_like(ids, dtype=torch.bool)
    with torch.no_grad():
        act = model.sample_actions("cuda", o, noise=noise[None].cuda())
    a = act[0].cpu().numpy()[:, :7]  # (10,7)
    # openpi transforms.py unnormalize: (a+1)/2*(q99-q01)+q01
    return (a + 1.0) / 2.0 * (q99_a - q01_a) + q01_a

# ---------- FlashRT INT8 ----------
os.environ["FLASH_RT_PALIGEMMA_TOKENIZER"] = os.path.join(os.environ["PI05_CKPT_DIR"], "assets/tokenizer/tokenizer.model")
sys.path.insert(0, os.environ["FLASHRT_ROOT"])
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

def flashrt_infer(pipe, obs_raw, noise):
    st_n = ((obs_raw["state"] - q01_s) / (q99_s - q01_s + 1e-6) * 2.0 - 1.0).astype(np.float32)
    pipe.set_prompt(obs_raw["prompt"], state=st_n)  # pi05: state 必须进 prompt
    r = pipe.infer({"image": obs_raw["imgs"][0], "wrist_image": obs_raw["imgs"][1],
                    "wrist_image_right": obs_raw["imgs"][2],
                    "state": obs_raw["state"]},
                   noise=noise.numpy())
    return np.asarray(r["actions"])

def cos_mse(a, b):
    cs = float(np.sum(a * b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    return cs, float(np.mean((a - b) ** 2)), float(np.abs(a - b).max())

BF16_MODE = os.environ.get("FRT_BF16") == "1"
print("loading FlashRT pool1 ...", flush=True)
pipe1 = Pi05TorchFrontendRtx(os.environ["PI05_CKPT_DIR"],
                             num_views=3, num_steps=10, vision_pool_factor=1,
                             vision_num_layers=27, cache_frames=1, use_cuda_graph=True, denoise_trace=True)
pipe1.set_prompt(PROMPT)
pipe1.calibrate_with_real_data([{"image": OBS_LIST[0]["imgs"][0], "wrist_image": OBS_LIST[0]["imgs"][1],
                                 "wrist_image_right": OBS_LIST[0]["imgs"][2], "state": OBS_LIST[0]["state"]}])
print("loading FlashRT pool2 ...", flush=True)
pipe2 = Pi05TorchFrontendRtx(os.environ["PI05_CKPT_DIR"],
                             num_views=3, num_steps=10, vision_pool_factor=2,
                             vision_num_layers=27, cache_frames=1, use_cuda_graph=True)
pipe2.set_prompt(PROMPT)
pipe2.calibrate_with_real_data([{"image": OBS_LIST[0]["imgs"][0], "wrist_image": OBS_LIST[0]["imgs"][1],
                                 "wrist_image_right": OBS_LIST[0]["imgs"][2], "state": OBS_LIST[0]["state"]}])

if BF16_MODE:
    print("\n=== STEP-TRACE: openpi vs FlashRT per-step v_t ===", flush=True)
    i = 0  # obs0
    obs = OBS_LIST[0]
    nz = NOISES[0]
    vs_ref, raw_ref = openpi_denoise_trace(obs["imgs"], obs["state"], obs["prompt"], nz)
    st_n = ((obs["state"] - q01_s) / (q99_s - q01_s + 1e-6) * 2.0 - 1.0).astype(np.float32)
    pipe1.set_prompt(obs["prompt"], state=st_n)
    r = pipe1.infer({"image": obs["imgs"][0], "wrist_image": obs["imgs"][1],
                     "wrist_image_right": obs["imgs"][2], "state": obs["state"]},
                    noise=nz.numpy())
    tr = r.get("denoise_trace")
    print("trace type:", type(tr), flush=True)
    if tr is None:
        print("no trace (denoise_trace init off?)", flush=True)
    else:
        arr = np.asarray(tr)
        print("trace shape:", arr.shape, flush=True)
        if arr.ndim == 3 and arr.shape[-2] == 10 or (arr.ndim == 3 and arr.shape[0] == 10):
            pass
        for s in range(min(10, arr.shape[0] if arr.ndim >= 1 else 0)):
            pass
    print("raw_ref[:3,:3]:", np.round(raw_ref[:3, :3], 4).tolist(), flush=True)
    print("raw_frt[:3,:3]:", np.round(r.get("raw_actions", np.zeros((1, 1)))[:, :3], 4).tolist() if "raw_actions" in r else "NA", flush=True)
    if "raw_actions" in r:
        c1, m1, x1 = cos_mse(raw_ref.flatten(), r["raw_actions"][:, :7].flatten())
        print(f"RAW normalized-space parity: cos={c1:.6f} mse={m1:.6f} max={x1:.4f}", flush=True)
    print("PARITY_DONE")
    sys.exit(0)

print("\n=== PARITY: openpi-BF16 vs FlashRT-INT8 (fixed obs+noise) ===", flush=True)
for i, obs in enumerate(OBS_LIST):
    ref = openpi_infer(obs["imgs"], obs["state"], obs["prompt"], NOISES[i])
    a1 = flashrt_infer(pipe1, obs, NOISES[i])
    a2 = flashrt_infer(pipe2, obs, NOISES[i])
    c1, m1, x1 = cos_mse(ref, a1)
    c2, m2, x2 = cos_mse(ref, a2)
    cp1, mp1, _ = cos_mse(a1, a2)
    print(f"obs{i}: BF16 vs INT8-pool1: cos={c1:.5f} mse={m1:.5f} max={x1:.4f} | "
          f"BF16 vs pool2: cos={c2:.5f} mse={m2:.5f} max={x2:.4f} | pool1-vs-pool2: cos={cp1:.5f}", flush=True)
print("PARITY_DONE")
