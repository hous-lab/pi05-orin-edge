"""Closed-loop eval: LIBERO-10 task0 x N episodes against an openpi-protocol policy server.

Mirrors examples/libero/main.py protocol exactly (wait steps, 180-deg image flip,
quat->axisangle, replan every 5, init_states[episode_idx]). Single task only.
Persistent copy (was /tmp/libero_eval_task0.py, lost to /tmp cleanup).
"""
import collections, dataclasses, json, math, pathlib, time

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _wcp
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


@dataclasses.dataclass
class Args:
    host: str = "192.168.55.1"
    port: int = 8007
    task_suite: str = "libero_10"
    task_idx: int = 0
    num_episodes: int = 20
    seed: int = 7
    replan_steps: int = 5
    num_steps_wait: int = 10
    video_dir: str = "./workdir/eval_task0/videos_task0"
    out_json: str = "./workdir/eval_task0/eval_task0_result.json"
    ws_alpha: float = -1.0  # client-side STEP warm-start; -1 = server-side (server FLASHRT_WS_ALPHA)


def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main(args: Args):
    np.random.seed(args.seed)
    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    task = suite.get_task(args.task_idx)
    init_states = suite.get_task_init_states(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    max_steps = MAX_STEPS[args.task_suite]
    print(f"TASK: {task.language}", flush=True)

    client = _wcp.WebsocketClientPolicy(args.host, args.port)
    print(f"METADATA: {client.get_server_metadata()}", flush=True)

    vd = pathlib.Path(args.video_dir); vd.mkdir(parents=True, exist_ok=True)
    results, all_infer_ms = [], []
    for ep in range(args.num_episodes):
        env.reset()
        obs = env.set_init_state(init_states[ep])
        action_plan = collections.deque()
        frames, infer_ms_ep, policy_calls = [], [], 0
        t, done = 0, False
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
            frames.append(img)
            if not action_plan:
                el = {
                    "reset": bool(not action_plan and t == args.num_steps_wait),
                    "observation/image": img,
                    "observation/wrist_image": wrist,
                    "observation/state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                    "prompt": str(task.language),
                }
                t0 = time.monotonic()
                try:
                    ret = client.infer(el)
                except Exception as e:
                    print(f"RECONNECT after {type(e).__name__}", flush=True)
                    client = _wcp.WebsocketClientPolicy(args.host, args.port)
                    ret = client.infer(el)
                infer_ms_ep.append((time.monotonic() - t0) * 1000)
                act = ret["actions"]
                assert len(act) >= args.replan_steps, f"policy predicted {len(act)} < {args.replan_steps}"
                action_plan.extend(act[: args.replan_steps])
                policy_calls += 1
            action = action_plan.popleft()
            obs, reward, done, info = env.step(action.tolist())
            t += 1
            if done:
                break
        all_infer_ms.extend(infer_ms_ep)
        rec = {
            "episode": ep, "success": bool(done), "steps": t, "policy_calls": policy_calls,
            "client_infer_ms_mean": round(float(np.mean(infer_ms_ep)), 1) if infer_ms_ep else None,
        }
        results.append(rec)
        print("EP " + json.dumps(rec), flush=True)
        imageio.mimwrite(vd / f"ep{ep:02d}_{'success' if done else 'failure'}.mp4", [np.asarray(f) for f in frames], fps=10)

    n = len(results); s = sum(r["success"] for r in results)
    summary = {
        "task": task.language, "task_idx": args.task_idx, "episodes": n, "successes": s,
        "success_rate": round(s / n, 4) if n else None,
        "steps_total": sum(r["steps"] for r in results),
        "policy_calls_total": sum(r["policy_calls"] for r in results),
        "infer_ms_p50": round(float(np.percentile(all_infer_ms, 50)), 1),
        "infer_ms_p95": round(float(np.percentile(all_infer_ms, 95)), 1),
        "infer_ms_mean": round(float(np.mean(all_infer_ms)), 1),
        "wall_s": None,
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    pathlib.Path(args.out_json).write_text(json.dumps({"episodes": results, "summary": summary}, indent=2))


if __name__ == "__main__":
    import sys
    tyro.cli(main)
