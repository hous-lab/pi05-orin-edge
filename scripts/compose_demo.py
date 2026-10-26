#!/usr/bin/env python3
"""Compose the featured demo clips from raw eval recordings.

Input: the per-episode MP4s the eval harness already writes of the exact
224x224 agentview frames the policy consumed (benchmark/libero_eval_task0.py),
plus the committed task JSON. Output: a dual-view clip - task view on the
left, a live engineering panel on the right, an authenticity banner along
the bottom. Every number on the panel is read from the committed records at
compose time; nothing is hand-typed.

The raw per-episode MP4s stay out of git (large, regenerable); this script is
the regeneration path.

Example:
  python3 scripts/compose_demo.py \
    --video <eval-out>/videos_task0/ep00_success.mp4 \
    --task-json benchmark/results/orin_gumix_geg8_stats_500ep/task0.json \
    --episode 0 --out assets/demos/demo_task0_ep0_success.mp4
"""
import argparse, json, pathlib

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont

W, H = 768, 480            # both multiples of 16 (yuv420p-safe)
VID = 448                  # left pane: 224x224 upscaled 2x, nearest
PANEL_X = 448
BANNER_Y = 448

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="raw per-episode mp4 (224x224 policy frames)")
ap.add_argument("--task-json", required=True)
ap.add_argument("--stats-json",
                default="benchmark/results/orin_gumix_geg8_stats_500ep/stats_500ep_orin.json")
ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--trim-frames", type=int, default=0,
                help="keep only the first N frames (0 = full episode)")
ap.add_argument("--note", default="", help="extra honesty note shown on the panel")
a = ap.parse_args()

tj = json.load(open(a.task_json))
ep = next(e for e in tj["episodes"] if e["episode"] == a.episode)
st = json.load(open(a.stats_json))

lang = tj["summary"]["task"]
outcome = ep["success"]
steps, calls = ep["steps"], ep["policy_calls"]
cmean = ep["client_infer_ms_mean"]

font_r = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 12)
font_m = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 13)
font_b = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", weight="bold")), 14)
font_s = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 11)

WHITE = (235, 235, 235)
DIM = (150, 150, 160)
GREEN = (90, 200, 120)
RED = (235, 95, 95)
ACCENT = (120, 190, 255)

def wrap(s, width, f):
    lines, cur = [], ""
    for wd in s.split():
        trial = (cur + " " + wd).strip()
        if f.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur); cur = wd
    if cur:
        lines.append(cur)
    return lines

def panel(draw, i):
    x = PANEL_X + 14
    y = 20
    draw.text((x, y), "LIBERO-10 - closed-loop simulation", font=font_b, fill=ACCENT)
    y += 28
    draw.text((x, y), "instruction:", font=font_s, fill=DIM)
    y += 17
    for ln in wrap(lang, 292, font_m):
        draw.text((x, y), ln, font=font_m, fill=WHITE)
        y += 17
    y += 8
    oc = GREEN if outcome else RED
    draw.text((x, y), f"episode {a.episode} - " + ("SUCCESS" if outcome else "FAILURE (step limit)"),
              font=font_b, fill=oc)
    y += 25
    draw.text((x, y), f"steps {steps}  -  policy calls {calls}", font=font_r, fill=DIM)
    y += 19
    draw.text((x, y), f"t = {(i + 1) / 10:.1f} s", font=font_r, fill=DIM)
    y += 21
    draw.line([(x, y), (x + 292, y)], fill=(60, 60, 70), width=1)
    y += 11
    draw.text((x, y), f"client round-trip mean  {cmean:.1f} ms", font=font_r, fill=WHITE)
    y += 18
    draw.text((x, y), "x86 -> Orin over the Jetson USB link", font=font_s, fill=DIM)
    y += 17
    draw.text((x, y), f"model p50 {st['p50_ms']:.2f} ms  -  {st['hz']:.1f} Hz", font=font_r, fill=WHITE)
    y += 18
    draw.text((x, y), f"{st['count']:,} calls - v1.0 stats rerun", font=font_s, fill=DIM)
    y += 17
    draw.line([(x, y), (x + 292, y)], fill=(60, 60, 70), width=1)
    y += 11
    if a.note:
        for ln in wrap(a.note, 292, font_s):
            draw.text((x, y), ln, font=font_s, fill=(220, 170, 90))
            y += 15
    draw.text((x, 420), "Jetson AGX Orin 64GB - MAXN + jetson_clocks", font=font_s, fill=DIM)
    draw.text((x, 436), f"recorded {ep['steps']} sim steps - eval harness v1.0", font=font_s, fill=DIM)

def banner(draw):
    draw.rectangle([0, BANNER_Y, W, H], fill=(18, 18, 22))
    draw.text((14, BANNER_Y + 9),
              "pi0.5 (~3.6B) on Jetson AGX Orin - simulation, not real hardware",
              font=font_s, fill=(200, 200, 210))
    right = "frames = the 224x224 agentview policy input"
    draw.text((W - 14 - int(font_s.getlength(right)), BANNER_Y + 9), right,
              font=font_s, fill=(140, 140, 150))

c_in = av.open(a.video)
vstream = next(s for s in c_in.streams if s.type == "video")
nframes = vstream.frames or sum(1 for _ in c_in.decode(video=0))
c_in.close()
c_in = av.open(a.video)

n = min(nframes, a.trim_frames) if a.trim_frames else nframes
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
c_out = av.open(a.out, mode="w", format="mp4")
vout = c_out.add_stream("libx264", rate=10)
vout.width, vout.height = W, H
vout.pix_fmt = "yuv420p"
vout.options = {"crf": "18"}

def render(i, frame):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[0:VID, 0:VID] = np.asarray(Image.fromarray(frame).resize((VID, VID), Image.NEAREST))
    im = Image.fromarray(canvas)
    d = ImageDraw.Draw(im)
    d.rectangle([PANEL_X, 0, W, BANNER_Y], fill=(24, 24, 28))
    d.rectangle([PANEL_X, 0, PANEL_X + 2, BANNER_Y], fill=(70, 70, 85))
    panel(d, i)
    banner(d)
    return np.asarray(im)

n_written = 0
for i, frame in enumerate(c_in.decode(video=0)):
    if a.trim_frames and i >= a.trim_frames:
        break
    vframe = av.VideoFrame.from_ndarray(render(i, frame.to_ndarray(format="rgb24")), format="rgb24")
    for pkt in vout.encode(vframe):
        c_out.mux(pkt)
    n_written += 1
    last_raw = frame.to_ndarray(format="rgb24")

hold = render(n - 1, last_raw)
for pkt in vout.encode(av.VideoFrame.from_ndarray(hold, format="rgb24")):
    c_out.mux(pkt)
for _ in range(10):
    for pkt in vout.encode(av.VideoFrame.from_ndarray(hold, format="rgb24")):
        c_out.mux(pkt)
for pkt in vout.encode(None):
    c_out.mux(pkt)
c_out.close()
print("wrote", a.out, f"({n_written} frames + 11 hold, from {nframes}-frame source)")
