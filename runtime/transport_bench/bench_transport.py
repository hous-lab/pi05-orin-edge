"""Transport RTT benchmark client (pairs with echo_server.py).

Modes:
  --profile-msgpack      : pack/unpack/sizes for a typical obs + response (no server needed)
  --transport ws|tcp|uds|shm : N round-trips against echo server, p50/p95/mean
Typical obs mirrors libero_eval_task0.py exactly:
  2 x (224,224,3) uint8 + state 8xf32 + prompt str + reset bool.
"""
import argparse
import json
import socket
import statistics
import time

import numpy as np

import transport_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--profile-msgpack", action="store_true")
ap.add_argument("--transport", choices=["ws", "tcp", "uds", "shm"], default="ws")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8301)
ap.add_argument("--uds-path", default="/tmp/pi05_bench_uds.sock")
ap.add_argument("--shm-name", default="pi05_bench_shm")
ap.add_argument("--views", type=int, default=2, choices=[2, 3])
ap.add_argument("--rounds", type=int, default=300)
ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--json-out", default=None)
args = ap.parse_args()


def make_obs(views: int) -> dict:
    el = {
        "reset": False,
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.uniform(-1, 1, (8,)).astype(np.float32),
        "prompt": "pick up the black bowl on the stove",
    }
    if views >= 3:
        el["observation/wrist_image_right"] = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return el


def pstats(xs):
    xs = sorted(xs)
    n = len(xs)
    return {
        "p50": xs[n // 2],
        "p95": xs[int(n * 0.95)],
        "mean": sum(xs) / n,
        "min": xs[0],
        "max": xs[-1],
    }


# ---------------------------------------------------------------- profile-msgpack
if args.profile_msgpack:
    for views in (2, 3):
        obs = make_obs(views)
        packer = tc.Packer()
        # warm
        for _ in range(50):
            blob = packer.pack(obs)
            tc.unpackb(blob)
        pack_ms, unpack_ms = [], []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            blob = packer.pack(obs)
            t1 = time.perf_counter_ns()
            back = tc.unpackb(blob)
            t2 = time.perf_counter_ns()
            pack_ms.append((t1 - t0) / 1e6)
            unpack_ms.append((t2 - t1) / 1e6)
        resp = {"actions": np.zeros((10, 7), dtype=np.float32), "server_timing": {"infer_ms": 92.4}}
        rpack_ms = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            rb = packer.pack(resp)
            rpack_ms.append((time.perf_counter_ns() - t0) / 1e6)

        # component: np .tobytes() copies inside the pack hook
        img = obs["observation/image"]
        tb_ms = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            img.tobytes()
            tb_ms.append((time.perf_counter_ns() - t0) / 1e6)

        s, rs = pstats(pack_ms), pstats(rpack_ms)
        print(f"== msgpack profile (views={views}) ==")
        print(f"  payload size        : {len(blob)} bytes ({len(blob)/1024:.1f} KiB)")
        print(f"  obs pack     p50/p95: {s['p50']:.3f} / {s['p95']:.3f} ms")
        print(f"  obs unpack   p50/p95: {pstats(unpack_ms)['p50']:.3f} / {pstats(unpack_ms)['p95']:.3f} ms")
        print(f"  resp pack    p50/p95: {rs['p50']:.4f} / {rs['p95']:.4f} ms (size {len(rb)} B)")
        print(f"  single 150KiB .tobytes(): {pstats(tb_ms)['p50']:.3f} ms x2 per pack")
        if args.json_out:
            json.dump({"views": views, "size": len(blob), "pack": s, "unpack": pstats(unpack_ms)},
                      open(args.json_out, "w"))
    raise SystemExit(0)

# ---------------------------------------------------------------- rtt bench
obs = make_obs(args.views)
packer = tc.Packer()
blob = packer.pack(obs)
print(f"payload: {len(blob)} bytes, transport={args.transport}, rounds={args.rounds}")


def one_round_ws():
    global _ws
    t0 = time.perf_counter_ns()
    _ws.send(blob)
    back = _ws.recv()
    dt = (time.perf_counter_ns() - t0) / 1e6
    tc.unpackb(back)  # time unpack separately
    return dt


def one_round_stream(sock):
    t0 = time.perf_counter_ns()
    tc.send_framed(sock, blob)
    back = tc.recv_framed(sock)
    dt = (time.perf_counter_ns() - t0) / 1e6
    tc.unpackb(back)
    return dt


def one_round_shm():
    t0 = time.perf_counter_ns()
    ch.client_send_req(blob)
    ch.client_wait_resp()
    dt = (time.perf_counter_ns() - t0) / 1e6
    tc.unpackb(ch.read_resp())
    return dt


_ws = None
ch = None
if args.transport == "ws":
    import websockets.sync.client as _wsc

    for attempt in range(60):
        try:
            _ws = _wsc.connect(f"ws://{args.host}:{args.port}", compression=None, max_size=None)
            break
        except OSError:
            time.sleep(0.5)
    else:
        raise SystemExit("ws connect failed")
    tc.unpackb(_ws.recv())  # metadata on connect (production protocol)
elif args.transport in ("tcp", "uds"):
    if args.transport == "tcp":
        s = socket.create_connection((args.host, args.port), timeout=10)
    else:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(args.uds_path)
    if args.transport == "tcp":
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    tc.unpackb(tc.recv_framed(s))  # metadata on connect
elif args.transport == "shm":
    ch = tc.ShmChannel(args.shm_name, create=False)
    ch.client_open()
    tc.unpackb(ch.client_read_metadata())  # metadata handshake

# warmup + measure
lat = []
srv_ms = []  # server_timing.infer_ms reported by the server (model-side latency)
_out = {}
for i in range(args.warmup + args.rounds):
    if args.transport == "ws":
        _ws.send(blob)
        t0 = time.perf_counter_ns()
        back = _ws.recv()
        dt = (time.perf_counter_ns() - t0) / 1e6
        _out.update(tc.unpackb(back))
    elif args.transport in ("tcp", "uds"):
        t0 = time.perf_counter_ns()
        tc.send_framed(s, blob)
        back = tc.recv_framed(s)
        dt = (time.perf_counter_ns() - t0) / 1e6
        _out.update(tc.unpackb(back))
    else:
        t0 = time.perf_counter_ns()
        ch.client_send_req(blob)
        ch.client_wait_resp()
        dt = (time.perf_counter_ns() - t0) / 1e6
        _out.update(tc.unpackb(ch.read_resp()))
    if isinstance(_out.get("server_timing"), dict) and "infer_ms" in _out["server_timing"]:
        srv_ms.append(_out["server_timing"]["infer_ms"])
    if i >= args.warmup:
        lat.append(dt)

st = pstats(lat)
res = {"transport": args.transport, "host": args.host, "views": args.views,
       "payload_KiB": round(len(blob) / 1024, 1), **{k: round(v, 4) for k, v in st.items()}}
if srv_ms:
    res["server_infer_p50"] = round(pstats(srv_ms)["p50"], 2)
    res["server_infer_mean"] = round(pstats(srv_ms)["mean"], 2)
print(json.dumps(res))
if args.json_out:
    json.dump({"transport": args.transport, **st}, open(args.json_out, "w"))
