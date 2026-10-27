#!/usr/bin/env python3
"""Render assets/pipeline.png - the Get started flow as a two-lane diagram.

The four-step bash block in the README is precise but flat: the reader has to
notice from the comments that steps 2-3 run on the Orin and steps 1 and 4 run
on the x86 host, and that two artifacts cross the boundary. This diagram shows
the same four steps as a snake across two lanes, with the artifact each step
leaves behind.

No numbers on this figure - it is orientation only. Every command stays in the
README so it remains copy-pasteable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import figstyle as S

S.apply()
fig = plt.figure(figsize=(10.0, 3.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 124); ax.set_ylim(0, 44); ax.axis("off")

# ---------------- lanes ----------------
S.panel(ax, 1.0, 24.0, 122, 19.0, S.PANEL_NAVY, "#D8E3F0")
S.panel(ax, 1.0, 1.0, 122, 19.0, S.PANEL_ORANGE, "#EBCBB4")
ax.text(3.4, 33.5, "x86 host  \u00b7  4090 laptop", rotation=90, fontsize=8.6,
        color=S.NAVY, ha="center", va="center", weight="bold")
ax.text(3.4, 10.5, "Jetson AGX Orin", rotation=90, fontsize=8.6,
        color=S.ORANGE, ha="center", va="center", weight="bold")

# ---------------- nodes ----------------
NODES = [
    (9.0, 27.0, "1 \u00b7 download + convert weights", S.NAVY,
     "environment/*.sh + export/tools/", "\u2192 ckpts/pi05_libero_official (bf16)"),
    (9.0, 5.0, "2 \u00b7 apply overlay + build", S.ORANGE,
     "flashrt_overlay/apply_overlay.sh", "\u2192 FlashRT f2f04db + CUTLASS 4.4.2"),
    (68.0, 5.0, "3 \u00b7 serve the shipped recipe", S.ORANGE,
     "configs/gumix_geg8.env", "\u2192 :8020  (ws / uds / shm)"),
    (68.0, 27.0, "4 \u00b7 closed-loop eval", S.NAVY,
     "benchmark/libero_eval_multitask.py", "\u2192 500 episodes \u2192 benchmark/results/"),
]
W, H = 52.0, 12.0
for x, y, title, c, cmd, out in NODES:
    ax.add_patch(plt.Rectangle((x, y), W, H, fc="white", ec=c, lw=1.2, zorder=3))
    ax.text(x + 1.8, y + H - 3.0, title, fontsize=10.2, weight="bold", color=c,
            ha="left", va="center", zorder=4)
    ax.text(x + 1.8, y + 5.6, cmd, fontsize=8.0, color=S.INK, ha="left",
            va="center", zorder=4, family="monospace")
    ax.text(x + 1.8, y + 2.4, out, fontsize=8.0, color=S.MUTED, ha="left",
            va="center", zorder=4)

# ---------------- arrows ----------------
ARROWS = [
    ((35.0, 27.0), (35.0, 17.0), "rsync", S.MUTED, 0),
    ((61.0, 11.0), (68.0, 11.0), "", S.MUTED, 0),
    ((94.0, 17.0), (94.0, 27.0), "ws over the Jetson USB link", S.MUTED, 0),
]
for (x0, y0), (x1, y1), lbl, c, rad in ARROWS:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=c, lw=1.3, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}"))
    if lbl:   # horizontal, in the gap between the two lanes
        ax.text(x0 + 1.6, 22.0, lbl, fontsize=8.2, color=c, ha="left",
                va="center", zorder=5)

fig.savefig("assets/pipeline.png", bbox_inches="tight", facecolor="white")
print("wrote assets/pipeline.png")
