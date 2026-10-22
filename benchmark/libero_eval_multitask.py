"""Multi-task LIBERO runner: 10 tasks x N episodes, aggregated (persistent copy)."""
import argparse, json, pathlib, subprocess, sys, time, math
ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--host", default="192.168.55.1")
ap.add_argument("--port", type=int, default=8000)
a = ap.parse_args()
OUT = pathlib.Path(a.outdir); OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "run.log"
REPO = pathlib.Path(__file__).resolve().parents[1]
for task_idx in range(10):
    out_json = OUT / f"task{task_idx}.json"
    if out_json.exists():
        print(f"task{task_idx} done, skip", flush=True)
        continue
    print(f"=== TASK {task_idx} START {time.strftime('%H:%M:%S')} ===", flush=True)
    with open(LOG, "a") as log:
        subprocess.run([sys.executable, str(REPO / "benchmark/libero_eval_task0.py"),
            "--args.host", a.host, "--args.port", str(a.port),
            "--args.task-idx", str(task_idx),
            "--args.num-episodes", str(a.episodes),
            "--args.video-dir", str(OUT / f"videos_task{task_idx}"),
            "--args.out-json", str(out_json)],
            stdout=log, stderr=log, check=False)
    if out_json.exists():
        d = json.loads(out_json.read_text())["summary"]
        print(f"=== TASK {task_idx} RESULT: {d['successes']} / {d['episodes']} sr={d['success_rate']} p50={d['infer_ms_p50']} ===", flush=True)
    else:
        print(f"=== TASK {task_idx} FAILED (no json) ===", flush=True)
total_s = total_n = 0
print("=== AGGREGATE ===", flush=True)
for t in range(10):
    p = OUT / f"task{t}.json"
    if not p.exists():
        print(f"task{t}: MISSING", flush=True); continue
    d = json.loads(p.read_text())["summary"]
    total_s += d["successes"]; total_n += d["episodes"]
    print(f"task{t}: {d['successes']}/{d['episodes']} sr={d['success_rate']}", flush=True)
ci = 1.96 * math.sqrt((total_s/total_n) * (1 - total_s/total_n) / total_n) if total_n else 0
print(f"TOTAL: {total_s}/{total_n} = {total_s/total_n:.1%} +-{ci*100:.1f}pp (95% CI)", flush=True)
