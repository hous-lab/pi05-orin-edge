# pi05-orin-edge: π0.5 VLA on Jetson AGX Orin at 92 ms

A complete, reproducible deployment of the π0.5 VLA (pi05_libero, ~3.5B params) on Jetson AGX Orin 64GB:
INT8 weight-only quantized GEMMs with a channel-split mix that keeps gate/up outliers safe in BF16,
a fused gate/up GEGLU epilogue kernel (GEG8), CUDA Graph single-step denoise, and a websocket server
speaking the openpi protocol. Same 92%-class LIBERO-10 quality as the best published results,
at roughly one third lower latency than the best public Orin number.

## Results

All numbers: Jetson AGX Orin 64GB (SM87, MAXN, jetson_clocks), LIBERO-10, 500 episodes
(10 tasks x 50), seed 7, replan 5, H=10 chunk; latency is model-caliber p50 (server-side
`pipe_infer` phase median). Full per-episode records: [benchmark/results/](benchmark/results/README.md).

| Config | Model p50 | Client p50 | LIBERO-10 500ep |
|---|---|---|---|
| shipped: INT8 mix (GU_MIX K16) + GEG8 | **92.4 ms** | 113-150 ms | **461/500 = 92.2%** |
| GU_MIX without GEG8 | 95.4 ms | 115.3-116.4 ms | 462/500 = 92.4% |
| BF16 arm (no INT8 mix) | - | 218.8-222.4 ms | 459/500 = 91.8% |
| APXInf (published, same class) | 147.15 ms | - | 92.0% |
| openpi reference | - | - | 92.4% |

The shipped recipe and the no-GEG8 variant sit within binomial noise of each other
(+-2.3 pp at N=500); the fused kernel buys -3.2 ms at a noise-level quality delta.

## What we changed on top of FlashRT

FlashRT (Apache-2.0, arXiv 2606.20537) is the upstream engine this work builds on; see
[NOTICE](NOTICE) and Acknowledgements. On top of FlashRT @ f2f04db8 the overlay is 18 files
(4 new + 14 modified, listed in flashrt_overlay/README.md):

- **GU_MIX K16 channel-split INT8**: per-layer 16-channel INT8 gate/up side weights with the
  BF16 side kept for outlier channels (Q2-ext mix), replacing the naive full-INT8 gate/up that
  collapsed long-horizon tasks (t8 -48pp / t9 -73pp).
- **GEG8 fused gate/up + GEGLU epilogue** (new CUDA kernel): bit-exact vs the unfused path at
  three gate levels; -3.2 ms.
- **FAST_QUANT** quantize plumbing (bit-exact, -5.6 ms) + vectorized elementwise + masked-norm
  + vectorized QKV split.
- **st8 fix**: pass the raw 8-dim normalized state conditioning instead of the 32-pad convention
  (a v044-era artifact that silently cost accuracy with official weights).
- Production server: warm-start noise shifter, memoized set_prompt, and removal of a duplicated
  prep block (byte-identical fix, removes a double prompt-embeds upload per request).

## Get started

```bash
# 0. one-time setup (docs/environment.md, docs/pipeline_guide.md)
#    x86 host: LIBERO sim + eval venv; Orin: venv + FlashRT + overlay + build

# 1. download + convert weights (x86)
bash environment/download_lerobot_base.sh
bash environment/download_norm_stats.sh
bash environment/download_paligemma_tokenizer.sh
python3 export/tools/convert_official_weights.py \
  --src ckpts/pi05_libero_base/model.safetensors \
  --out ckpts/pi05_libero_official/model.safetensors

# 2. apply the FlashRT overlay + build on Orin (clones FlashRT @ f2f04db8 + CUTLASS v4.4.2)
bash flashrt_overlay/apply_overlay.sh
#    build commands: flashrt_overlay/README.md

# 3. serve on Orin
source configs/gumix_geg8.env
python runtime/server_flashrt_official_ws.py --views 2 --pool 1 --layers 27 --steps 1 --port 8020

# 4. closed-loop eval from x86 (LIBERO-10, multitask 10x50 episodes)
PYTHONPATH=<openpi>/third_party/libero python benchmark/libero_eval_multitask.py \
  --host <orin-ip> --port 8020
```

run_pipeline.sh chains steps 1-3 with fail-fast guards (downloads, conversion, sync).

## Repo map

| Path | What it is |
|---|---|
| flashrt_overlay/ | our 18-file overlay over FlashRT @ f2f04db8 + apply script |
| export/tools/convert_official_weights.py | fp32->bf16 bundle converter (byte-exact verified) |
| runtime/server_flashrt_official_ws.py | production server (ws / uds / shm transports) |
| benchmark/ | closed-loop eval (task0 + multitask), parity gate, transport bench |
| benchmark/results/ | full 500-episode records for the three arms |
| configs/gumix_geg8.env | the 13-env production recipe |
| environment/ | version locks + offline wheelhouse + download scripts |
| tools/ | K16 mix-set generator + outlier probe |
| ckpts/ data/ | upstream artifacts: link-only policy + download table |
| docs/ | environment, pipeline, recipe semantics, quantization journey, lessons |

## Acknowledgements

- FlashRT (Apache-2.0) - the engine this overlay targets: github.com/flashrt-project/FlashRT (arXiv 2606.20537)
- openpi / pi0.5 (Apache-2.0) - model, weights, protocol, LIBERO example: github.com/Physical-Intelligence/openpi
- lerobot/pi05_libero_base - the official fp32 weights: huggingface.co/lerobot/pi05_libero_base
- LIBERO benchmark - github.com/Lifelong-Robot-Learning/LIBERO
- CUTLASS (BSD-3) - github.com/NVIDIA/cutlass
- PaliGemma tokenizer - Google big_vision public bucket (Gemma terms of use apply)
