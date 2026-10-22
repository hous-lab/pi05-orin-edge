"""Echo server for transport microbenchmarks (NO model, NO GPU).

Mirrors production server_flashrt_official_ws.py protocol:
  recv msgpack obs -> (stub infer) -> send {"actions": (10,7) f32, "server_timing": {...}}
Supports: ws (websocket), tcp, uds, shm transports.
"""
import argparse
import asyncio
import threading
import time

import numpy as np

import transport_common as tc

ap = argparse.ArgumentParser()
ap.add_argument("--transport", choices=["ws", "tcp", "uds", "shm"], default="ws")
ap.add_argument("--port", type=int, default=8301)
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--uds-path", default="/tmp/pi05_bench_uds.sock")
ap.add_argument("--shm-name", default="pi05_bench_shm")
ap.add_argument("--fake-infer-ms", type=float, default=0.0,
                help="simulate model time (sleep); 0 = pure echo")
args = ap.parse_args()

packer = tc.Packer()
ACTIONS = np.zeros((10, 7), dtype=np.float32)
METADATA = {"views": 2, "pool": 1, "steps": 1, "echo_stub": True}


def log(msg):
    print(f"[echo-{args.transport}] {msg}", flush=True)


def stub_infer(obs: dict) -> dict:
    if args.fake_infer_ms > 0:
        time.sleep(args.fake_infer_ms / 1000.0)
    return {
        "actions": ACTIONS,
        "server_timing": {"infer_ms": args.fake_infer_ms},
    }


# ------------------------------------------------------------------- ws
async def _ws_handler(ws):
    log(f"client connected: {ws.remote_address}")
    await ws.send(packer.pack(METADATA))  # production protocol: metadata on connect
    while True:
        obs = tc.unpackb(await ws.recv())
        t0 = time.monotonic()
        out = stub_infer(obs)
        out["server_timing"]["echo_unpack_ms"] = (time.monotonic() - t0) * 1000
        await ws.send(packer.pack(out))


async def _ws_main():
    import websockets.asyncio.server as _server

    async with _server.serve(_ws_handler, args.host, args.port, compression=None, max_size=None) as srv:
        log(f"serving ws on {args.host}:{args.port}")
        await srv.serve_forever()


# ------------------------------------------------------- tcp / uds (threaded)
def _stream_handler(conn):
    try:
        tc.send_framed(conn, packer.pack(METADATA))  # production protocol
        while True:
            obs = tc.unpackb(tc.recv_framed(conn))
            t0 = time.monotonic()
            out = stub_infer(obs)
            out["server_timing"]["echo_unpack_ms"] = (time.monotonic() - t0) * 1000
            tc.send_framed(conn, packer.pack(out))
    except (ConnectionError, OSError):
        pass
    finally:
        conn.close()


def _stream_main(sock):
    sock.listen(1)
    where = args.uds_path if args.transport == "uds" else f"{args.host}:{args.port}"
    log(f"serving {args.transport} on {where}")
    while True:
        conn, _ = sock.accept()
        threading.Thread(target=_stream_handler, args=(conn,), daemon=True).start()


# ------------------------------------------------------------------- shm
def _shm_main():
    ch = tc.ShmChannel(args.shm_name, create=True)
    ch.server_open()
    ch.server_put_metadata(packer.pack(METADATA))
    log(f"serving shm name={args.shm_name} path={ch.path}")
    last_seq = 0
    while True:
        seq, _ = ch.server_wait_req()
        if seq <= last_seq:
            continue
        last_seq = seq
        obs = tc.unpackb(ch.read_req())
        t0 = time.monotonic()
        out = stub_infer(obs)
        out["server_timing"]["echo_unpack_ms"] = (time.monotonic() - t0) * 1000
        ch.server_send_resp(packer.pack(out))


if __name__ == "__main__":
    if args.transport == "ws":
        asyncio.run(_ws_main())
    elif args.transport == "tcp":
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((args.host, args.port))
        _stream_main(s)
    elif args.transport == "uds":
        import os
        import socket

        if os.path.exists(args.uds_path):
            os.unlink(args.uds_path)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(args.uds_path)
        _stream_main(s)
    elif args.transport == "shm":
        _shm_main()
