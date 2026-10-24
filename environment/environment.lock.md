# Orin Runtime Environment Lock

> Verified on Jetson AGX Orin 64GB, L4T 36.5.0 (JetPack 6.x), python 3.10.12.
> The runtime is FlashRT + this repo's overlay; the openpi repo is NOT needed on Orin.

## Locked versions
| Component | Version | Source |
|---|---|---|
| torch | 2.8.0+cu126 | NVIDIA Jetson system stack (do not touch) |
| CUDA / cuDNN | 12.6 / 9.3.0 | system |
| transformers | 4.53.2 | wheelhouse |
| sentencepiece / tokenizers | 0.2.2 / 0.21.4 | wheelhouse |
| numpy | 1.26.4 | wheelhouse |
| openpi-client / websockets / msgpack / tyro | current | wheelhouse |
| FlashRT | upstream @ f2f04db8 + overlay | git clone + [flashrt_overlay/](../flashrt_overlay/) |
| CUTLASS | v4.4.2 | git clone ([flashrt_overlay/apply_overlay.sh](../flashrt_overlay/apply_overlay.sh) does it) |

## Conventions
- venv with `--system-site-packages` so the NVIDIA torch stack stays primary; pip installs only python-only wheels from the wheelhouse built by [environment/make_orin_wheelhouse.sh](make_orin_wheelhouse.sh).
- FlashRT builds against the system torch; the kernels .so lands inside the FlashRT tree (see [flashrt_overlay/README.md](../flashrt_overlay/README.md) for the build commands).
- Everything installs offline on Orin: build the wheelhouse on x86 and copy it over (Orin has no external network in our setup).
