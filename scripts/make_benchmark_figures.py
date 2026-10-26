#!/usr/bin/env python3
"""Benchmark figures for the README, from the tagged per-call latency records.

C1: per-call latency timeline + distribution over the 26,650 server-side records
    archived in benchmark/results/orin_gumix_geg8_stats_500ep/ (the v1.0-tagged
    500-episode stats rerun).
C2: per-task success heatmap, six arms x ten tasks, from the committed task
    JSONs. Cliff tasks (task2 here) and per-config drift become visible.

Path-parameterized like everything else in this repo; defaults point at the
committed records. Output: assets/benchmarks/*.png.
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="benchmark/results",
                help="repo results dir (task JSONs + stats rerun records)")
ap.add_argument("--outdir", default="assets/benchmarks")
a = ap.parse_args()
OUT = pathlib.Path(a.outdir); OUT.mkdir(parents=True, exist_ok=True)
RES = pathlib.Path(a.results)

# ---------------- C1: latency timeline + distribution ----------------
z = np.load(RES / "orin_gumix_geg8_stats_500ep/stats_500ep_orin.records.npz")
lat = z["latency_records"]
p50, p95, p99 = np.percentile(lat, [50, 95, 99])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), dpi=160,
                               gridspec_kw={"height_ratios": [3, 2]})
ax1.plot(lat, lw=0.3, color="#1f77b4")
for y, c, lbl in [(p50, "#2ca02c", f"p50 {p50:.2f}"),
                  (p95, "#ff7f0e", f"p95 {p95:.2f}"),
                  (p99, "#d62728", f"p99 {p99:.2f}")]:
    ax1.axhline(y, ls="--", lw=0.9, color=c)
ax1.legend([f"p50 {p50:.2f} ms", f"p95 {p95:.2f} ms", f"p99 {p99:.2f} ms"],
           loc="lower right", fontsize=8, framealpha=0.9)
imax = int(np.argmax(lat))
ax1.annotate(f"max {lat[imax]:.1f} ms", (imax, lat[imax]),
             xytext=(imax - 4200, lat[imax] + 2), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.7))
ax1.set_ylabel("model-caliber latency (ms)")
ax1.set_xlabel("policy call # (500 episodes, in order)")
ax1.set_ylim(90, 130)
ax1.set_title("Orin model-caliber latency, 26,650 closed-loop policy calls "
              "(GU_MIX + GEG8, MAXN + jetson_clocks)")
ax1.margins(x=0)

ax2.hist(lat, bins=220, color="#1f77b4")
for x, c in [(p50, "#2ca02c"), (p95, "#ff7f0e"), (p99, "#d62728")]:
    ax2.axvline(x, ls="--", lw=0.9, color=c)
ax2.set_xlabel("latency (ms)")
ax2.set_ylabel("calls")
ax2.set_xlim(90, 128)
fig.suptitle(f"jitter (p95-p50) = {p95 - p50:.2f} ms", fontsize=9, y=0.995)
fig.tight_layout()
fig.savefig(OUT / "latency_timeline.png", bbox_inches="tight")
plt.close(fig)

# ---------------- C2: per-task success heatmap, six arms ----------------
ARMS = [
    ("Orin GU_MIX+GEG8 (primary)", "gumix_geg8_500ep"),
    ("Orin GU_MIX (no GEG8)", "gumix_500ep"),
    ("Orin BF16 (no INT8)", "bf16_500ep",
     ),
    ("Orin GU_MIX+GEG8 (stats rerun)", "orin_gumix_geg8_stats_500ep"),
    ("4090L INT8 recipe (b)", "x86_4090l_b_500ep"),
    ("4090L FP8 (a)", "x86_4090l_a_500ep"),
]
M = np.zeros((len(ARMS), 10))
for r, (_, d) in enumerate(ARMS):
    for t in range(10):
        M[r, t] = json.load(open(RES / d / f"task{t}.json"))["summary"]["successes"]

fig, ax = plt.subplots(figsize=(11, 3.6), dpi=160)
im = ax.imshow(M, cmap="RdYlGn", vmin=35, vmax=50, aspect="auto")
for r in range(M.shape[0]):
    for t in range(M.shape[1]):
        ax.text(t, r, f"{M[r, t]:.0f}", ha="center", va="center", fontsize=9)
ax.set_xticks(range(10)); ax.set_xticklabels([f"task{t}" for t in range(10)])
ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels([n for n, *_ in ARMS], fontsize=9)
ax.set_title("LIBERO-10 successes per task (out of 50 episodes), six arms - "
             "losses concentrate on cliff tasks that shift per config")
fig.colorbar(im, ax=ax, label="successes / 50")
fig.tight_layout()
fig.savefig(OUT / "task_success.png", bbox_inches="tight")
plt.close(fig)

print("wrote", OUT / "latency_timeline.png", OUT / "task_success.png")