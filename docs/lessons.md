# Lessons and failure ledger

Methodology and the failures that shaped it. Written from the build log; each entry is
phenomenon -> root cause -> what we changed. These applied to an AGX Orin + x86 two-machine
setup running FlashRT + custom kernels.

## Methodology

- **Closed-loop-only acceptance**: kernel micro-benchmarks and cos-similarity gates are necessary,
  never sufficient. A bit-exact kernel that moves the closed-loop number is rejected.
  (Exception on purpose: the GEG8 kernel was bit-exact AND closed-loop-neutral - both required.)
- **Evidence grading**: an observation ranks as L0 measurement / L1 derived / L2 hypothesis /
  L3 conjecture; only L0/L1 justify config changes.
- **Both-direction ablations**: removing a component can look free when its work hides in slack
  (dependency-chain floor). Add AND remove to attribute; a single direction can invert the
  conclusion (we reclassified a carry mechanism after a subtraction ablation).
- **Window judgment**: degraded GPU windows forge latency AND numeric evidence. A deterministic
  bad checksum that cos-metrics rate 0.8814 is a window artifact fingerprint, not a bug - run
  the probe pair (small/large sizes) plus latency and numeric anchors before verdicts.
- **A/B noise floor**: measured at 1.2 pp on the 500-episode protocol (binomial +-2.3 pp at
  N=500); deltas inside the floor are recorded as noise, not wins.
- **Stream discipline**: side-stream pipelines must pass the producing stream into every capture;
  default-stream captures forge "output garbage" verdicts (hit twice: kernel dump + tap dump).
- **Caliber discipline**: GPU-busy / server-infer / client-observed are different numbers; every
  claim states its caliber. Mixing calibers in a headline ratio hides ~20-35 ms.

## Failure ledger (abridged)

| # | Failure | Root cause | What changed |
|---|---|---|---|
| 1 | "Wrong-weights" era: 60.0% +-6.8pp ceiling | community checkpoint from the wrong era treated as official | sha256-verified official weights; bundle hash in [ckpts/README.md](../ckpts/README.md) |
| 2 | task0 gate blind to long-horizon collapse | 20-episode task0 gate at 92% policy = coin flip | 500-episode multitask record is the only acceptance record |
| 3 | INT8 t8/t9 collapse discovered late | quick gates only | 500ep sweep added to every config change |
| 4 | Fake latency wins from unlocked clocks | DVFS states mixed inside A/B | clock lock + health-window check before every A/B |
| 5 | nsys adds overhead to the measured process | profiler is an intervener | latency claims come from non-profiled runs; nsys only for attribution |
| 6 | Same-grid kernels binned together in nsys | bucket by (kernel, grid) mixes shapes | waterfall replay as cheap strict verification |
| 7 | Parity harness passed a corrupted kernel | cos gate blind to pure-gain corruption | every cos gate paired with an absmax check |
| 8 | Server ran a different config than measured | env setdefault traps | explicit env in recipes; fingerprint lines as launch proof |
| 9 | CUDA graph replay garbage with >=3 engines co-resident | multi-engine graph capture state | single-engine chains; hard reject at chain level |
| 10 | fp16 buffer downloaded as float16 = misread bits | BF16 buffer read via np.float16 | uint16 shift conversion in every comparison tool |
| 11 | "Missing" embed_tokens on load | tied lm_head alias in safetensors metadata | bundle materializes embed_tokens (see [scripts/export/](../scripts/export/)) |
| 12 | make silently skipped recompiles after reboot | clock skew: future mtimes on .o | delete stale .o / touch before rebuild; check for skew warnings |
