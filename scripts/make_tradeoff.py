#!/usr/bin/env python3
"""Render assets/benchmarks/tradeoff.png - success vs. latency, one point per arm.

The Results table carries the repo's central claim in six rows: the INT8 recipe
holds LIBERO-10 accuracy while cutting latency, and the 4090 arm is faster still
without buying accuracy. One scatter makes that a picture.

Sources: the Results table in README.md, which in turn cites the committed
500-episode records under benchmark/results/. Values are duplicated here on
purpose - a figure must not silently disagree with the table, and the assertion
below fails loudly if the JSONs and this list drift apart.

Error bars are Wilson 95% CIs on the per-arm success count, so the reader can
see that the vertical differences are inside sampling noise.
"""
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

import figstyle as S

RES = pathlib.Path("benchmark/results")

# (label, model p50 ms, successes, arm dir, role) - README Results table
ARMS = [
    ("4090L GU-Mix + GEG8", 24.78, "x86_4090l_b_500ep", "x86"),
    ("4090L FP8", 25.56, "x86_4090l_a_500ep", "x86"),
    ("Orin GU-Mix + GEG8", 92.40, "gumix_geg8_500ep", "orin"),
    ("Orin GU-Mix", 95.40, "gumix_500ep", "orin"),
    ("APXInf (our repro)", 147.15, None, "ext"),      # 460/500, published
]
N = 500
APXINF_SUCCESS = 460
REFERENCE = 92.4          # openpi published 500-episode success, %

COLOR = {"orin": S.ORANGE, "x86": S.NAVY, "ext": S.MUTED}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def successes(d: str):
    return sum(json.load(open(RES / d / f"task{t}.json"))["summary"]["successes"]
               for t in range(10))


pts = []
for label, ms, arm, role in ARMS:
    k = APXINF_SUCCESS if arm is None else successes(arm)
    lo, hi = wilson(k, N)
    pts.append(dict(label=label, ms=ms, k=k, pct=100 * k / N,
                    lo=100 * lo, hi=100 * hi, role=role))

S.apply()
fig, ax = plt.subplots(figsize=(9.2, 4.3))

ax.axhline(REFERENCE, color=S.MUTED, ls=(0, (6, 4)), lw=1.0, zorder=1)
ax.text(20.6, REFERENCE + 0.35, f"openpi reference {REFERENCE:.1f}%",
        fontsize=8.8, color=S.MUTED, ha="left", va="bottom")

for p in pts:
    ax.errorbar(p["ms"], p["pct"], yerr=[[p["pct"] - p["lo"]], [p["hi"] - p["pct"]]],
                fmt="o", ms=7.5, color=COLOR[p["role"]], ecolor=COLOR[p["role"]],
                elinewidth=1.2, capsize=3.5, capthick=1.2, zorder=4,
                markerfacecolor="white" if p["role"] == "ext" else COLOR[p["role"]])

LABEL = {   # manual offsets, points; the two 4090 points are 1 pp apart
    "4090L GU-Mix + GEG8": (12, 11, "left"),
    "4090L FP8": (12, -11, "left"),
    "Orin GU-Mix + GEG8": (-11, -16, "right"),
    "Orin GU-Mix": (11, 10, "left"),
    "APXInf (our repro)": (-11, -15, "right"),
}
for p in pts:
    dx, dy, ha = LABEL[p["label"]]
    ax.annotate(p["label"], (p["ms"], p["pct"]), xytext=(dx, dy),
                textcoords="offset points", fontsize=9, color=S.INK, ha=ha,
                va="center", zorder=5)

# speed-up brackets along the top
for x0, x1, txt in [(24.78, 92.40, "3.7x"), (92.40, 147.15, "1.6x faster")]:
    ax.annotate("", xy=(x1, 95.05), xytext=(x0, 95.05),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#B6BEC9",
                                shrinkA=0, shrinkB=0))
    ax.text(math.sqrt(x0 * x1), 95.3, txt, fontsize=8.5, color=S.MUTED,
            ha="center", va="bottom")

ax.set_xscale("log")
ax.set_xlim(20, 185); ax.set_ylim(85.0, 96.2)
ax.set_xticks([25, 50, 100, 150])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
ax.xaxis.set_minor_locator(ticker.NullLocator())
ax.set_xlabel("model inference p50 (ms, log scale)")
ax.set_ylabel("LIBERO-10 success over 500 episodes (%)")
S.tidy(ax, grid_axis="y")
ax.set_title("LIBERO-10 success vs. model inference latency", loc="left")

fig.text(0.01, -0.055,
         "Error bars: Wilson 95% CI on 500 episodes. BF16 arm omitted - no model-caliber p50 was "
         "recorded (client p50 218.8-222.4 ms).\n"
         "* APXInf = our reproduction on the same Orin through their harness (BF16, in-process, "
         "one-step); their README quotes 119.05 ms at CUDA-graph replay p50.",
         fontsize=7.8, color=S.MUTED, ha="left", va="top", linespacing=1.5)

fig.savefig("assets/benchmarks/tradeoff.png", bbox_inches="tight", facecolor="white")
print("wrote assets/benchmarks/tradeoff.png")
for p in pts:
    print(f"  {p['label']:22s} {p['ms']:7.2f} ms  {p['k']}/500 = {p['pct']:.1f}%  "
          f"CI [{p['lo']:.1f}, {p['hi']:.1f}]")
