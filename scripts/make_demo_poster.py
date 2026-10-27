#!/usr/bin/env python3
"""Render assets/demos/poster_grid.png - a contact sheet of the six demo clips.

The Demo section embeds GitHub user-attachment URLs, which do not exist inside
the repo: a clone, a fork or a mirror shows nothing until the reader presses
play, and nothing at all if the attachments go away. This sheet is the
in-repo anchor and the fallback image.

Frames: one frame from the last third of each clip (extracted with ffmpeg from
the committed mp4 masters), cropped to the 512x512 agentview the policy
actually consumes. A late-but-not-final frame shows the manipulation instead of
a parked arm, and it does not ask the reader to judge success from a still.
Captions: steps and policy calls come from the per-episode records in
benchmark/results/demo_512_rerecord/, not from the burnt-in video panel.
"""
import json
import pathlib
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import figstyle as S

DEMOS = pathlib.Path("assets/demos")
RES = pathlib.Path("benchmark/results/demo_512_rerecord")
OUT = DEMOS / "poster_grid.png"

# (task, episode, order in the sheet) - the failure goes last, as in the README
CLIPS = [(0, 1), (3, 0), (4, 0), (5, 0), (9, 0), (2, 1)]
CROP = (0, 0, 512, 512)          # agentview occupies the top-left 512x512
CHECK, CROSS = "\u2713", "\u2715"


def _duration(mp4: pathlib.Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0", str(mp4)],
                         check=True, capture_output=True, text=True)
    return float(out.stdout.strip())


def grab_frame(mp4: pathlib.Path, dst: pathlib.Path, at=0.68):
    """One frame from the last third: the arm is still manipulating."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{_duration(mp4) * at:.2f}",
                    "-i", str(mp4), "-frames:v", "1", "-update", "1", str(dst)],
                   check=True)
    return Image.open(dst).convert("RGB").crop(CROP)


def episode_record(task: int, ep: int):
    d = json.load(open(RES / f"task{task}.json"))
    return d["episodes"][ep]


S.apply()
tmp = pathlib.Path(tempfile.mkdtemp())

IMG = 1.72            # inches per cell image
GAP = 0.22
MARGIN = 0.10
CAP = 0.42            # caption band height
COLS, ROWS = 3, 2
W = COLS * IMG + (COLS - 1) * GAP + 2 * MARGIN
H = ROWS * (IMG + CAP) + (ROWS - 1) * 0.14 + 2 * MARGIN + 0.34

fig = plt.figure(figsize=(W, H))
for i, (task, ep) in enumerate(CLIPS):
    r, c = divmod(i, COLS)
    x = MARGIN + c * (IMG + GAP) / W
    y = 1 - (MARGIN + 0.34 + r * (IMG + CAP + 0.14) + IMG) / H
    ax = fig.add_axes([x, y, IMG / W, IMG / H])
    ax.imshow(grab_frame(DEMOS / f"demo_task{task}_ep{ep}_"
                         f"{'success' if episode_record(task, ep)['success'] else 'failure'}.mp4",
                         tmp / f"t{task}_e{ep}.png"))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(S.FAINT); s.set_linewidth(1.0)

    rec = episode_record(task, ep)
    ok = rec["success"]
    fig.text(x, y - 0.055 / H, f"task {task}", fontsize=10.5, weight="bold",
             color=S.INK, ha="left", va="top")
    fig.text(x + IMG / W, y - 0.052 / H, CHECK if ok else CROSS, fontsize=12,
             weight="bold", color=S.GREEN if ok else S.RED, ha="right", va="top")
    fig.text(x, y - 0.20 / H,
             f"{rec['steps']} steps \u00b7 {rec['policy_calls']} policy calls",
             fontsize=8.2, color=S.MUTED, ha="left", va="top")

fig.text(0.5, 1 - 0.06 / H, "one frame from the last third of each clip",
         fontsize=8.6, color=S.MUTED, ha="center", va="top")

fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
