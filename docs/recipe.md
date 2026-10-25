# Recipe: the 13-env production configuration

[configs/gumix_geg8.env](../configs/gumix_geg8.env) is the configuration behind every number in this repo. Source it
before launching the server; the startup fingerprint is the proof it took effect.

## The 13 variables

| Env | Value | Effect |
|---|---|---|
| FVK_PI05_RTX_INT8_DECODER_ONLY | 1 | INT8 GEMMs for the 90 decoder GEMM weights |
| FVK_PI05_RTX_ENC8_ATTN | 1 | encoder ATTN (qkv+o) INT8, 36 GEMMs |
| FVK_PI05_RTX_ENC8_MLP | 1 | enables the encoder MLP path (required for the mix below) |
| FVK_PI05_RTX_ENC8_MIX | 1 | K16 channel-split mix on gate/up: keep the K=16 largest-outlier channels in BF16 |
| FVK_PI05_RTX_ENC8_MIX_FILE | ${PI05_MIX_SETS} | the K16 channel sets ([tools/mix_sets_k16.npz](../tools/mix_sets_k16.npz)) |
| FVK_PI05_RTX_ENC8_MIX_GEG8 | 1 | fused gate/up+GEGLU epilogue kernel on the mix path |
| FVK_PI05_RTX_ENC8_DOWN | 1 | encoder DOWN GEMMs INT8 (18 GEMMs) |
| FLASHRT_PI05_INT8_FAST_QUANT | 1 | bit-exact fast quantize plumbing (-5.6 ms) |
| FVK_PI05_RTX_VEC_ELEMENTWISE | 1 | vectorized elementwise kernels |
| FVK_PI05_RTX_FLOW_START | 0.5 | partial flow start |
| FLASHRT_WS_ALPHA | 0.5 | warm-start noise shifter blend (see below) |
| FLASHRT_PI05_EMBED_NP | 0 | embed path toggle |
| FVK_PI05_RTX_FORCE_INT8 | 1 | force the INT8 path on (see the setdefault trap below) |

## Two traps documented loudly

1. **FORCE_INT8 setdefault**: the server does `os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")`.
   That means an A/B "off" arm that just unsets the variable still runs INT8. A/B arms must set
   it explicitly to 0, not just unset it.
2. **WS_ALPHA default 0.8 vs recipe 0.5**: the script default (0.8) is NOT the production value.
   The recipe file sets 0.5 - the value behind the 500-episode anchors. The server prints the
   effective value at startup in a loud comment (see [runtime/server_flashrt_official_ws.py](../runtime/server_flashrt_official_ws.py)).

## Latency calibers (do not mix them)

| Caliber | What it includes | Where you see it |
|---|---|---|
| model p50 | server-side `pipe_infer` phase median | server PHASE log lines |
| server infer | model + server glue (set_prompt, msgpack, handler) | server wall clock |
| client p50 | server infer + transport round-trip | eval JSON `infer_ms_p50` |

On the 2-machine USB topology, client p50 runs ~20-35 ms above model caliber. Every number in
this repo states its caliber. The README table uses model p50 (92.4 ms shipped).

## GPU health windows

The GPU occasionally enters degraded windows (long sessions, thermals): latencies inflate 2x+
and (rarely) numerics corrupt. Before trusting any latency or parity measurement:
1. clock lock: MAXN + jetson_clocks on,
2. latency anchor: run the standard bench and compare against the healthy-window band,
3. numeric anchor: a known checksum from a healthy window (a corrupted-window run reproduces a
   deterministic bad checksum - cos-metrics can be blind to it).
