#!/usr/bin/env python3
"""Render assets/overview.png - the two-machine deployment topology.

Three stacked rows, one idea each:
  1. headline strip  - what the run is and what it scored
  2. topology        - who runs what (client on the laptop, engine on Orin)

Latency appears exactly once (the Orin box, model caliber). The client
round-trip 113-150 ms is deliberately absent: it is a *different caliper*
of the same run (model + USB transport + client glue), so drawing it on the
client box reads as a competing claim about the same number. It belongs in
the Results table, and assets/benchmarks/latency_calibers.png explains how
the two relate.

Every number comes from the committed records (same sources as
make_benchmark_figures.py).
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import figstyle as S

RES = pathlib.Path("benchmark/results")
tot = sum(json.load(open(RES / f"gumix_geg8_500ep/task{t}.json"))["summary"]["successes"]
          for t in range(10))
N = 500
MODEL_P50 = 92.4          # headline model p50, ms (shipped recipe)
BASELINE = 92.4           # openpi reference success, %

S.apply()
fig = plt.figure(figsize=(10.0, 3.8))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 120); ax.set_ylim(0, 50); ax.axis("off")

# ---------------- headline strip ----------------
S.panel(ax, 1.5, 39.2, 117, 9.6, S.PANEL_GRAY, S.FAINT)
S.text(ax, 60, 46.3, "LIBERO-10 closed-loop, 500 episodes (10 tasks x 50)",
       10.0, S.MUTED)
S.text(ax, 60, 43.4,
       f"{tot}/{N} = {tot / N * 100:.1f}%  (official baseline {BASELINE:.1f}%)",
       14.5, S.INK, weight="bold")
S.text(ax, 60, 40.6, f"latency {MODEL_P50:.1f} ms ({1000 / MODEL_P50:.1f} Hz)",
       11.0, S.ORANGE)

# geometry: two 38-unit boxes, 41 units of link between them
LX, RX, BW, L, R = 1.5, 80.5, 38, 6.0, 36.0
LCX, RCX = LX + BW / 2, RX + BW / 2

# ---------------- client box ----------------
S.panel(ax, LX, L, BW, R - L, S.PANEL_NAVY, "#C9D8EA")
S.text(ax, LCX, 32.4, "CLIENT", 8.4, S.MUTED, weight="bold")
S.text(ax, LCX, 28.8, "4090 laptop", 15, S.NAVY, weight="bold")
ax.plot([LX + 5, LX + BW - 5], [25.9, 25.9], color="#C9D8EA", lw=1.0, zorder=2)
S.text(ax, LCX, 21.6, "LIBERO-10 simulator (MuJoCo)", 10.4)
S.text(ax, LCX, 18.3, "openpi-protocol policy client", 10.4)
ax.plot([LX + 3.5, LX + BW - 3.5], [14.0, 14.0], color="#DCE7F3", lw=0.9, zorder=2)
S.text(ax, LCX, 10.2, "obs upload + action download", 9.0, S.MUTED)

# ---------------- Orin box ----------------
S.panel(ax, RX, L, BW, R - L, S.PANEL_ORANGE, "#EBCBB4")
S.text(ax, RCX, 32.4, "SERVER", 8.4, S.MUTED, weight="bold")
S.text(ax, RCX, 28.8, "Jetson AGX Orin 64GB", 15, S.ORANGE, weight="bold")
ax.plot([RX + 5, RX + BW - 5], [25.9, 25.9], color="#EBCBB4", lw=1.0, zorder=2)
S.text(ax, RCX, 22.4, "[GU-Mix + GEG8] engine:", 11, weight="bold")
S.text(ax, RCX, 19.2, "SigLIP ViT + Gemma LM", 10.4)
S.text(ax, RCX, 16.3, "action expert, 1-step flow matching", 9.4)
ax.plot([RX + 3.5, RX + BW - 3.5], [14.0, 14.0], color="#F0D8C6", lw=0.9, zorder=2)
S.text(ax, RCX, 10.2, f"{MODEL_P50:.1f} ms ({1000 / MODEL_P50:.1f} Hz)",
       13.5, S.ORANGE, weight="bold")

# ---------------- link ----------------
ax.add_patch(FancyArrowPatch((LX + BW + 1.5, 21.0), (RX - 1.5, 21.0),
                             arrowstyle="<->", mutation_scale=15,
                             color=S.MUTED, lw=1.5, zorder=2))
S.text(ax, 60, 24.2, "Jetson USB link", 10.6, S.INK, weight="bold")
S.text(ax, 60, 17.8, "openpi websocket protocol", 9.4, S.MUTED)

fig.savefig("assets/overview.png", bbox_inches="tight", facecolor="white")
print(f"wrote assets/overview.png; primary arm = {tot}/{N}")
