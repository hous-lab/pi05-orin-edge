#!/usr/bin/env python3
"""Benchmark figures for the README, from the tagged per-call latency records.

C0: latency calibers - model p50 vs client p50 as one stacked bar, so the
    "is the 92.4 ms inside the 113-150 ms?" question answers itself.
C1: per-call latency timeline + distribution over the 26,650 server-side
    records archived in benchmark/results/orin_gumix_geg8_stats_500ep/
    (the v1.0-tagged 500-episode stats rerun).
C2: per-task success heatmap, six arms x ten tasks, from the committed task
    JSONs. Cliff tasks and per-config drift become visible.

Path-parameterized like everything else in this repo; defaults point at the
committed records. Output: assets/benchmarks/*.png.
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import Normalize

import figstyle as S

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="benchmark/results",
                help="repo results dir (task JSONs + stats rerun records)")
ap.add_argument("--outdir", default="assets/benchmarks")
a = ap.parse_args()
OUT = pathlib.Path(a.outdir); OUT.mkdir(parents=True, exist_ok=True)
RES = pathlib.Path(a.results)
S.apply()

# ---------------- C0: the two latency calibers ----------------
MODEL_P50 = 92.4        # shipped recipe, model caliber (README headline)
CLIENT = [json.load(open(RES / "gumix_geg8_500ep" / f"task{t}.json"))
          ["summary"]["infer_ms_p50"] for t in range(10)]
CLIENT_LO, CLIENT_HI = min(CLIENT), max(CLIENT)

fig, ax = plt.subplots(figsize=(9.6, 2.35))
ax.barh(1, CLIENT_HI, height=0.46, color=S.ORANGE, alpha=0.18, zorder=2)
ax.barh(1, MODEL_P50, height=0.46, color=S.ORANGE, zorder=3)
ax.barh(0, MODEL_P50, height=0.46, color=S.ORANGE, zorder=3)
ax.axvline(MODEL_P50, color=S.MUTED, ls=(0, (3, 2)), lw=1.0, zorder=4)

ax.text(MODEL_P50 / 2, 0, f"{MODEL_P50:.1f} ms", color="white", fontsize=10,
        weight="bold", ha="center", va="center", zorder=5)
ax.text(MODEL_P50 / 2, 1, f"{MODEL_P50:.1f} ms", color="white", fontsize=10,
        weight="bold", ha="center", va="center", zorder=5)
ax.text((MODEL_P50 + CLIENT_HI) / 2, 1, "transport + client glue",
        fontsize=8.6, color=S.ORANGE, ha="center", va="center", zorder=5)
ax.text(CLIENT_HI + 2.5, 1, f"{CLIENT_LO:.0f}-{CLIENT_HI:.0f} ms", fontsize=10,
        weight="bold", color=S.ORANGE, va="center")

ax.set_yticks([0, 1])
ax.set_yticklabels(["model caliber\n(server pipe_infer)",
                    "client caliber\n(eval infer_ms_p50)"], fontsize=9)
ax.set_xlim(0, 176); ax.set_ylim(-0.45, 1.45)
ax.set_xlabel("latency (ms) - same 500-episode run, two calipers")
S.tidy(ax, grid_axis="x")
ax.set_title("Same run, two calipers: the model p50 sits inside the client p50",
             loc="left")
fig.tight_layout()
fig.savefig(OUT / "latency_calibers.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- C1: latency timeline + distribution ----------------
z = np.load(RES / "orin_gumix_geg8_stats_500ep/stats_500ep_orin.records.npz")
lat = z["latency_records"]
n = lat.size
p50, p95, p99 = np.percentile(lat, [50, 95, 99])
lo, hi = 91.0, 127.0

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.9), dpi=S.DPI,
                               gridspec_kw={"height_ratios": [3, 2], "hspace": 0.42})
ax1.plot(lat, lw=0.35, color=S.BLUE, zorder=3)
ax1.axhline(p50, ls=(0, (5, 3)), lw=0.9, color=S.GREEN, zorder=2)
ax1.text(0.985, 0.95,
         "\n".join([f"p50  {p50:.2f} ms", f"p95  {p95:.2f} ms",
                    f"p99  {p99:.2f} ms", f"max  {lat.max():.1f} ms"]),
         transform=ax1.transAxes, ha="right", va="top", fontsize=8.6,
         color=S.MUTED, linespacing=1.55, zorder=6,
         bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.5))
ax1.set_ylabel("latency (ms)")
ax1.set_xlabel("policy call index (chronological, all 500 episodes)")
ax1.set_ylim(lo, hi); ax1.set_xlim(0, n - 1); ax1.margins(x=0)
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
S.tidy(ax1, grid_axis="y")
ax1.set_title("Engine inference latency on Orin", loc="left")

ax2.hist(lat, bins=np.arange(lo, hi + 0.25, 0.25), color=S.BLUE, zorder=3)
ax2.set_yscale("log")
ax2.set_xlim(lo, hi); ax2.set_ylim(0.8, 4e4)
ax2.set_xlabel("latency (ms)")
ax2.set_ylabel("call count (log)")
ax2.text(0.985, 0.95,
         f"99% of calls land within {p99 - p50:.2f} ms of p50\n"
         f"the other 1% are isolated spikes, 1-5 calls each",
         transform=ax2.transAxes, ha="right", va="top", fontsize=8.8,
         color=S.MUTED, linespacing=1.5, zorder=6)
S.tidy(ax2, grid_axis="y")
for a_, tag in ((ax1, "(a)"), (ax2, "(b)")):
    a_.text(-0.075, 1.04, tag, transform=a_.transAxes, fontsize=9,
            color=S.MUTED, weight="bold")

fig.savefig(OUT / "latency_timeline.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------- C2: per-task success heatmap, six arms ----------------
ARMS = [
    ("Orin GU-Mix + GEG8 (primary)", "gumix_geg8_500ep"),
    ("Orin GU-Mix (no GEG8)", "gumix_500ep"),
    ("Orin BF16 (no INT8)", "bf16_500ep"),
    ("Orin GU-Mix + GEG8 (stats rerun)", "orin_gumix_geg8_stats_500ep"),
    ("4090L INT8 recipe (b)", "x86_4090l_b_500ep"),
    ("4090L FP8 (a)", "x86_4090l_a_500ep"),
]
M = np.zeros((len(ARMS), 10))
for r, (_, d) in enumerate(ARMS):
    for t in range(10):
        M[r, t] = json.load(open(RES / d / f"task{t}.json"))["summary"]["successes"]

norm = Normalize(vmin=30, vmax=50)
cmap = plt.get_cmap("RdYlGn")

fig = plt.figure(figsize=(10.6, 3.6))
ax = fig.add_axes([0.205, 0.145, 0.665, 0.70])
ax_tot = fig.add_axes([0.878, 0.145, 0.055, 0.70])

ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
for r in range(M.shape[0]):
    for t in range(M.shape[1]):
        r_, g_, b_, _ = cmap(norm(M[r, t]))
        lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
        ax.text(t, r, f"{M[r, t]:.0f}", ha="center", va="center", fontsize=9.5,
                color="white" if lum < 0.55 else S.INK,
                weight="bold" if M[r, t] <= 40 else "normal")
ax.set_xticks(range(10))
ax.set_xticklabels([f"task{t}" for t in range(10)], fontsize=9)
ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels([n_ for n_, *_ in ARMS], fontsize=9)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(ARMS), 1), minor=True)
ax.grid(which="minor", color="white", lw=1.2)
ax.tick_params(which="both", length=0)

ax_tot.set_xlim(0, 1); ax_tot.set_ylim(len(ARMS) - 0.5, -0.5); ax_tot.axis("off")
ax_tot.text(0.5, -0.72, "total", fontsize=8.5, color=S.MUTED, ha="center",
            weight="bold")
for r in range(len(ARMS)):
    tot = int(M[r].sum())
    ax_tot.text(0.5, r, f"{tot}", fontsize=9.5, ha="center", va="center",
                weight="bold" if r == 0 else "normal",
                color=S.INK if r == 0 else S.MUTED)

# no colorbar: every cell is labelled, colour only carries the gestalt
fig.text(0.02, 0.955, "LIBERO-10 successes per task, out of 50 episodes",
         fontsize=12.5, weight="bold", color=S.INK, ha="left", va="top")
fig.savefig(OUT / "task_success.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

print("wrote", *(OUT / f for f in
                 ("latency_calibers.png", "latency_timeline.png", "task_success.png")))
