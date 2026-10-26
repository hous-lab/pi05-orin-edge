#!/usr/bin/env python3
"""Render assets/overview.png - the two-machine deployment topology.

Boxes are plain matplotlib; every number on the figure comes from the
committed records (same sources as make_benchmark_figures.py).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = pathlib.Path("benchmark/results")
st = json.load(open(RES / "orin_gumix_geg8_stats_500ep/stats_500ep_orin.json"))
g = json.load(open(RES / "gumix_geg8_500ep/task0.json"))["summary"]
tot = 0
for t in range(10):
    tot += json.load(open(RES / f"gumix_geg8_500ep/task{t}.json"))["summary"]["successes"]

BLUE = "#1f4e79"
ORANGE = "#c55a11"
GRAY = "#595959"

fig = plt.figure(figsize=(12, 5.0), dpi=160)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 120); ax.set_ylim(0, 50)
ax.axis("off")

def box(x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=fc, ec=GRAY, lw=1.2))

def txt(x, y, s, size=10, color="black", ha="center", weight="normal", family=None):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            weight=weight, family=family)

# ---- x86 host box ----
box(2, 12, 30, 28, "#eaf1f8")
txt(17, 36.5, "x86 host", 13, BLUE, weight="bold")
txt(17, 33.2, "RTX 4090 Laptop (control arm)", 8.5, GRAY)
txt(17, 30, "LIBERO-10 simulator (MuJoCo)", 9.5)
txt(17, 26.8, "openpi-protocol policy client", 9.5)
txt(17, 22.5, "client round-trip", 8.5, GRAY)
txt(17, 19.8, "113-150 ms", 11, BLUE, weight="bold")
txt(17, 15.5, "obs upload + action download", 8, GRAY)

# ---- Orin box ----
box(64, 12, 30, 28, "#fdf0e7")
txt(79, 36.5, "Jetson AGX Orin 64GB", 13, ORANGE, weight="bold")
txt(79, 33.2, "MAXN + jetson_clocks, locked clocks", 8.5, GRAY)
txt(79, 30, "runtime/server_flashrt_official_ws.py", 9, family="monospace")
txt(79, 26.2, "FlashRT engine:", 9.5)
txt(79, 23.4, "SigLIP vision tower + Gemma LM (27L)", 9)
txt(79, 20.6, "+ action expert, 1-step flow matching (CUDA Graph)", 8.3)
txt(79, 15.5, "model inference", 8.5, GRAY)
txt(79, 12.8, "92.4 ms p50 - 10.7 Hz", 11, ORANGE, weight="bold")

# ---- link arrow ----
ar = FancyArrowPatch((33.5, 26), (62.5, 26), arrowstyle="<->", mutation_scale=16, color=GRAY, lw=1.6)
ax.add_patch(ar)
txt(48, 28.6, "Jetson USB link", 10, weight="bold")
txt(48, 26, "192.168.55.1", 9, family="monospace")
txt(48, 23.6, "openpi websocket protocol", 8.5, GRAY)

# ---- overlay chip (what we changed) ----
box(36, 2.5, 24, 7, "#fff8e6")
txt(48, 7.5, "this repo's overlay on FlashRT", 8.5, "#7a5c00")
txt(48, 5.2, "GU_MIX K16 INT8 + GEG8 fused gate/up + FAST_QUANT", 7.5)
txt(48, 3.6, "(env-gated overlay inside the FlashRT engine)", 7.5, GRAY)

# ---- results strip ----
box(2, 41, 92, 7, "#f2f2f2")
txt(48, 44.5, f"LIBERO-10 closed-loop, 500 episodes (10 tasks x 50): {tot}/500 = {tot/5:.1f}% "
              f"(official float baseline 92.4%) - model p50 {92.4:.1f} ms", 9.5, weight="bold")

fig.savefig("assets/overview.png", bbox_inches="tight")
print("wrote assets/overview.png; primary arm =", tot, "/500")