# benchmark/results — anchor eval records

Self-produced closed-loop records (LIBERO-10, x86 client + Orin server). Each arm is the
complete 10-task record: task0..task9.json × 50 episodes = 500 episodes, protocol identical
across arms (seed 7, replan every 5 actions, H=10 action chunk, 180° image flip, quat→axis-angle).

Caliber note: `client_infer_ms_mean` and per-task `infer_ms_p50` are **x86-client-observed**
latency (transport + round-trip included). Model-caliber p50 (server-side, `pipe_infer` phase
median) for the shipped recipe is reported in the [README](../../README.md) table; client-observed p50 runs
~20–35 ms higher than model caliber on the 2-machine USB topology. The BF16 baseline arm predates the
server-side per-call stats wrapper, so only client caliber exists for it (the dash in the README table).

Measurement metadata (applies to every arm below):

| Field | Value |
|---|---|
| Devices | two-machine topology: x86 eval host runs the LIBERO-10 simulator + openpi-protocol client; Jetson AGX Orin 64GB runs the model server over the stock Jetson USB link (192.168.55.1) |
| Power and clocks | Orin locked to MAXN + jetson_clocks before every measured run; the x86 host was not clock-locked |
| Stack versions | Orin: L4T 36.5.0, CUDA 12.6, torch 2.8.0+cu126; x86: CUDA 12.8, torch 2.8.0+cu128, python 3.10.12 ([environment/environment.lock.md](../../environment/environment.lock.md)) |
| Protocol | LIBERO-10, 10 tasks × 50 episodes per arm, seed 7, replan every 5 actions, H=10 action chunk, 180° image flip, quat→axis-angle |
| Sample counts | accuracy: 500 episodes per arm; latency: the two 4090L arms and the Orin rerun record one server-side sample per policy call (~27-29k samples per arm) over the full closed loop; the Orin anchor arms report model p50 from server-side phase timing across the run |
| Latency calibers | model p50 = server-side `pipe_infer` phase median; client p50 = model + server glue + transport round-trip; defined in [docs/recipe.md](../../docs/recipe.md), never mixed inside one number |
| Quality metric | per-episode task success within the LIBERO step limit, aggregated per arm over 500 episodes; per-task JSONs keep the raw per-task split |

| Directory | Config | Successes/500 | Rate | Client p50 per task |
|---|---|---|---|---|
| [gumix_geg8_500ep/](gumix_geg8_500ep/) | shipped recipe (13-env GU_MIX + GEG8) | 461 | 92.2% | 113–150 ms |
| [gumix_500ep/](gumix_500ep/) | GU_MIX without GEG8 | 462 | 92.4% | 115.3–116.4 ms |
| [bf16_500ep/](bf16_500ep/) | BF16 arm (no INT8 mix) | 459 | 91.8% | 218.8–222.4 ms |
| [x86_4090l_b_500ep/](x86_4090l_b_500ep/) | Orin INT8 recipe on sm_89 (control arm) | 450 | 90.0% | 26.7–26.9 ms |
| [x86_4090l_a_500ep/](x86_4090l_a_500ep/) | FP8 W8A16-fp16 (RTX-native) | 445 | 89.0% | 27.4–27.6 ms |
| [orin_gumix_geg8_stats_500ep/](orin_gumix_geg8_stats_500ep/) | shipped recipe, stats rerun (second 500-episode anchor) | 462 | 92.4% | 112.1–112.6 ms |

The shipped recipe and the no-GEG8 variant are within binomial noise of each other
(±2.3 pp at N=500); GEG8 bought −3.2 ms model-caliber latency at a noise-level quality
delta. All three Orin arms stay at parity with the best published pi05_libero results
(openpi 92.4%, APXInf 92.0%), so the INT8 mix is closed-loop-neutral here.

The two 4090L arms (see [platforms/x86](../../platforms/x86/README.md)) rerun the same
protocol with the device as the only variable: both stay within binomial noise of Orin
(b vs Orin p=0.23, a vs b p=0.60, a vs Orin p=0.082) - losses concentrate on cliff tasks
that shift per config (b loses 8 eps on task2, a loses 15 on task8 but gains 8 on task9),
not a systematic accuracy-floor drop. Full-percentile latency records (model caliber,
server-side, one sample per policy call over the full closed loop) sit next to each
4090L arm's task JSONs; the Orin shipped recipe got the same full-percentile treatment in
a dedicated rerun archived as [orin_gumix_geg8_stats_500ep/](orin_gumix_geg8_stats_500ep/).

Model-caliber closed-loop latency, all arms with per-call server-side records (jitter = p95 − p50):

| Arm | calls | p50 | p95 | p99 | max | jitter |
|---|---|---|---|---|---|---|
| Orin, shipped recipe (stats rerun) | 26,650 | 93.30 | 93.80 | 93.94 | 125.1 | 0.50 ms |
| 4090L, config b (same recipe, sm_89) | 26,924 | 24.78 | 25.18 | 28.96 | 71.9 | 0.40 ms |
| 4090L, config a (FP8 W8A16-fp16) | 28,644 | 25.56 | 28.46 | 43.38 | 87.7 | 2.90 ms |

The Orin tail is razor-thin (p99 − p50 = 0.64 ms): the closed-loop model latency is essentially
a constant 93.3 ms on the locked-clock device, and its records double as a second 500-episode
accuracy anchor (462/500 = 92.4%, same recipe as the 461/500 primary anchor).
