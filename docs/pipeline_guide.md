# Pipeline guide

End-to-end: download → convert → overlay+build → serve → eval. Two machines: steps in the
x86 column run on the eval host; serve runs on Orin. Every stage has a verifiable checkpoint.

| Stage | Where | Command | Checkpoint |
|---|---|---|---|
| 1. weights | x86 | `bash environment/download_lerobot_base.sh` (14.5 GB) | sha256 21b87117... printed OK |
| 2. assets | x86 | `bash environment/download_norm_stats.sh` + `download_paligemma_tokenizer.sh` | hashes print OK |
| 3. convert | x86 | `python3 export/tools/convert_official_weights.py --src ... --out ...` | 813 tensors BF16, 8,286,945,016 B, sha256 41cb8647... |
| 4. overlay+build | Orin | `bash flashrt_overlay/apply_overlay.sh` then cmake/make (below) | fingerprint lines at server start |
| 5. sync | x86 | `bash scripts/sync_to_orin.sh <user@host>` | rsync output |
| 6. serve | Orin | `source configs/gumix_geg8.env && python runtime/server_flashrt_official_ws.py --views 2 --pool 1 --layers 27 --steps 1 --port 8020` | 6 fingerprint lines + serving |
| 7. eval | x86 | `PYTHONPATH=<openpi>/third_party/libero python benchmark/libero_eval_multitask.py --host <orin> --port 8020` | 10 task JSONs, 500 eps |

## Convert (stage 3)

The converter streams the 14.5 GB fp32 source in 64 MiB chunks, casts fp32→bf16 RNE
(bit-exact with torch), materializes embed_tokens from the tied lm_head, and emits a compact
JSON safetensors header (insertion order, 8-byte-aligned padding). Verified byte-identical to
the production bundle. `--verify <ref>` compares per-tensor bytes against an existing bundle.

## Overlay + build (stage 4)

```bash
bash flashrt_overlay/apply_overlay.sh   # clones FlashRT @ f2f04db8 + CUTLASS v4.4.2, copies overlay files
cd ../FlashRT                           # nvcc must be on PATH (/usr/local/cuda/bin)
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DFA2_ARCH_NATIVE_ONLY=ON -DFA2_DTYPES=bf16 -DFA2_HDIMS="96;128;256"
make -j8                                # kernels .so lands inside the FlashRT tree
```

The FA2 flags keep the vendored FlashAttention matrix to the detected arch (the default
full matrix targets compute_120, which CUDA 12.6 on L4T cannot build). SM80-family
CUTLASS INT8 auto-enables on sm_87; `-DENABLE_SM80_INT8_CUTLASS=ON` is the explicit
spelling.

The overlay build produces the kernels .so that flash_rt loads; the startup fingerprint lines
(next section) are the checkpoint that the quantization mix loaded as designed.

## Serve (stage 6) - startup fingerprint

A correct launch prints exactly this shape (counts matter):

```
INT8 quantized 90 decoder GEMM weights
Q2-ext mix: zeroed INT8 + BF16 side weights for 18 layers K=16
INT8 quantized 36 encoder ATTN GEMM weights (qkv+o)
INT8 quantized 36 encoder GATEUP GEMM weights (gate/up)
INT8 quantized 18 encoder DOWN GEMM weights
CUDA Graph captured for Pi05Pipeline (+ decoder-only variant)
serving on 0.0.0.0:8020
```

If GATEUP shows 36/36 full-INT8 or the Q2-ext mix line is missing, the recipe file was not
sourced - the server silently runs a different configuration.

## Eval (stage 7)

`libero_eval_multitask.py` wraps `libero_eval_task0.py` per task (0..9), 50 episodes each,
writes one JSON per task, and prints a SUMMARY line per task. A 500-episode run takes hours;
for a quick gate use `libero_eval_task0.py --num_episodes 20`. The task0 production anchor is
45/50 (90%) over the 50-episode protocol, so a quick gate of 18/20 is consistent with the
anchor; 19/20+ is a strong pass.

## Parity harness (optional, openpi reference)

`benchmark/parity_gate_official.py` three-way compares (openpi reference fp32 / our bundle
bf16 / overlay stack) on synthetic frames. Needs the openpi repo: `OPENPI_SRC=<openpi>/src`.
