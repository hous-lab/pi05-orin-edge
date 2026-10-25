#!/usr/bin/env python3
"""π0.5 FlashRT RTX FP8 latency bench on the 4090 Laptop.

Caliber: Orin production config - 2 views, pool=1, 27 layers, 1 step,
FP8, CUDA graph, exact-state mode. Model caliber = wall p50 of
pipe.infer() incl. host pre/post (no transport), synced per call.
"""
import sys, time, json, argparse, os
import numpy as np

FLASHRT = os.environ.get("FLASHRT_ROOT", ".")

CKPT = os.environ.get("PI05_CKPT_DIR", "ckpts/pi05_libero_official")
sys.path.insert(0, FLASHRT)

import torch
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

ap = argparse.ArgumentParser()
ap.add_argument("--warmup", type=int, default=30)
ap.add_argument("--iters", type=int, default=100)
ap.add_argument("--fp8", type=int, default=1)
ap.add_argument("--graph", type=int, default=1)
ap.add_argument("--out", default="results_pi05_rtx_fp8.json")
args = ap.parse_args()

z = np.load(os.environ.get("PI05_CALIB_NPZ", "assets/calib_task0_real.npz"))
obs = {
    "image": z["image"][0].copy(),
    "wrist_image": z["wrist_image"][0].copy(),
    "state": z["state"][0].copy(),
}
prompt = str(z["prompt"])

pipe = Pi05TorchFrontendRtx(
    checkpoint_dir=CKPT,
    num_views=2,
    chunk_size=10,
    max_prompt_len=200,
    num_steps=1,
    vision_pool_factor=1,
    vision_num_layers=27,
    cache_frames=1,
    use_fp8=bool(args.fp8),
    use_cuda_graph=bool(args.graph),
)

pipe.set_prompt(prompt, state=obs["state"])
pipe.calibrate_with_real_data([obs])

def one_infer():
    return pipe.infer(dict(obs), noise=np.zeros((10, 32), dtype=np.float32))

r0 = one_infer()["actions"]
r1 = one_infer()["actions"]
a0n = r0.detach().float().cpu().numpy() if hasattr(r0, "detach") else np.asarray(r0)
a1n = r1.detach().float().cpu().numpy() if hasattr(r1, "detach") else np.asarray(r1)
assert np.isfinite(a0n).all(), "non-finite actions"
det = bool(np.array_equal(a0n, a1n))
print("determinism(bitwise):", det, "action[0,:3]:", np.round(a0n[0, :3], 4))

for _ in range(args.warmup):
    one_infer()
torch.cuda.synchronize()

lat = []
for _ in range(args.iters):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    one_infer()
    torch.cuda.synchronize()
    lat.append((time.perf_counter() - t0) * 1e3)

lat.sort()
p50 = lat[len(lat) // 2]
res = {
    "gpu": torch.cuda.get_device_name(0),
    "views": 2, "pool": 1, "layers": 27, "steps": 1,
    "fp8": bool(args.fp8), "graph": bool(args.graph),
    "p50_ms": round(p50, 2),
    "p10_ms": round(lat[len(lat)//10], 2),
    "p90_ms": round(lat[9*len(lat)//10], 2),
    "mean_ms": round(sum(lat)/len(lat), 2),
    "iters": args.iters,
    "stats": {k: v for k, v in pipe.get_latency_stats().items()
              if isinstance(v, (int, float))},
}
out = args.out
json.dump(res, open(out, "w"), indent=1)
print("P50 %.2f ms | p10 %.2f | p90 %.2f | mean %.2f | det=%s" %
      (p50, res["p10_ms"], res["p90_ms"], res["mean_ms"], det))
print("saved:", out)
