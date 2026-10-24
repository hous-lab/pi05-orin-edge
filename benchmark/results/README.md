# benchmark/results — anchor eval records

Self-produced closed-loop records (LIBERO-10, x86 client + Orin server). Each arm is the
complete 10-task record: task0..task9.json × 50 episodes = 500 episodes, protocol identical
across arms (seed 7, replan every 5 actions, H=10 action chunk, 180° image flip, quat→axis-angle).

Caliber note: `client_infer_ms_mean` and per-task `infer_ms_p50` are **x86-client-observed**
latency (transport + round-trip included). Model-caliber p50 (server-side, `pipe_infer` phase
median) for the shipped recipe is reported in the [README](../../README.md) table; client-observed p50 runs
~20–35 ms higher than model caliber on the 2-machine USB topology.

| Directory | Config | Successes/500 | Rate | Client p50 per task |
|---|---|---|---|---|
| [gumix_geg8_ep/](gumix_geg8_500ep/) | shipped recipe (13-env GU_MIX + GEG8) | 461 | 92.2% | 113–150 ms |
| [gumix_500ep/](gumix_500ep/) | GU_MIX without GEG8 | 462 | 92.4% | 115.3–116.4 ms |
| [bf16_500ep/](bf16_500ep/) | BF16 arm (no INT8 mix) | 459 | 91.8% | 218.8–222.4 ms |

The shipped recipe and the no-GEG8 variant are within binomial noise of each other
(±2.3 pp at N=500); GEG8 bought −3.2 ms model-caliber latency at a noise-level quality
delta. All three arms stay at parity with the best published pi05_libero results
(openpi 92.4%, APXInf 92.0%), so the INT8 mix is closed-loop-neutral here.
