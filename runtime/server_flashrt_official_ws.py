#!/usr/bin/env python3
"""pi0.5 policy server speaking the openpi websocket protocol, on the FlashRT overlay stack.

Usage (Orin):
  python server_flashrt_official_ws.py --ckpt ckpts/pi05_libero_official \
      --views 2 --pool 1 --steps 1 --port 8020
Protocol: on connect the server sends metadata; infer accepts {"observation/image",
"observation/wrist_image", "observation/wrist_image_right", "observation/state", "prompt"}
and returns {"actions": (10, 7) fp32}.

Transport (--transport, default ws):
  ws   : websocket (default; the protocol the openpi client expects)
  uds  : Unix domain socket, 4-byte LE length prefix + msgpack (same-machine recommended)
  shm  : /dev/shm mmap slot + FIFO signaling (uses runtime/transport_bench/transport_common.py)
Stub mode (--echo): serves zero actions without loading the model - transport benchmarks only.
"""
import argparse
import asyncio
import http
import json
import logging
import os
import struct
import time
import traceback

import numpy as np

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("flashrt_server")

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.environ.get("PI05_CKPT_DIR", "ckpts/pi05_libero_official"))
ap.add_argument("--views", type=int, default=3)
ap.add_argument("--pool", type=int, default=2)
ap.add_argument("--layers", type=int, default=27)
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--cache-frames", type=int, default=1)
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--transport", choices=["ws", "uds", "shm"], default="ws",
                help="ws=现状websocket(默认,向后兼容); uds=Unix socket同机通道; shm=共享内存通道")
ap.add_argument("--uds-path", default="/tmp/pi05_policy.sock")
ap.add_argument("--shm-name", default="pi05_policy_shm")
ap.add_argument("--echo", action="store_true",
                help="echo桩: 不加载模型/GPU, 回零动作 - 仅用于传输层基准")
a = ap.parse_args()

import os
os.environ.setdefault("FVK_PI05_RTX_FORCE_INT8", "1")
os.environ["FLASH_RT_PALIGEMMA_TOKENIZER"] = os.environ.get(
    "FLASH_RT_PALIGEMMA_TOKENIZER",
    os.path.join(a.ckpt, "assets/tokenizer/tokenizer.model"))
import sys
# FLASHRT_ROOT must point at the overlay-applied FlashRT tree (see flashrt_overlay/apply_overlay.sh)
sys.path.insert(0, os.environ["FLASHRT_ROOT"])

import websockets.asyncio.server as _server
import websockets.frames
from openpi_client import msgpack_numpy

import math
if not a.echo:  # --echo stub mode skips checkpoint-side file deps (no GPU needed)
    NS = __import__("json").load(open(
        os.path.join(a.ckpt, "norm_stats.json")))["norm_stats"]
    Q01_S = np.array(NS["state"]["q01"], dtype=np.float32)
    Q01_A = np.array(NS["actions"]["q01"], dtype=np.float32)
    Q99_A = np.array(NS["actions"]["q99"], dtype=np.float32)
    # NOTE: script default is 0.8, but the published recipe (configs/gumix_geg8.env) uses 0.5 -
    # the value behind the 92.4% @ 500-episode result. Source the recipe file to get 0.5.
    _WS_ALPHA = float(__import__("os").environ.get("FLASHRT_WS_ALPHA", "0.8"))
    Q99_S = np.array(NS["state"]["q99"], dtype=np.float32)


class EchoPolicy:
    """--echo 桩: 传输层基准专用, 语义镜像 FlashRTPolicy.infer 的输出形状。"""

    def infer(self, obs: dict) -> dict:
        return {"actions": np.zeros((10, 7), dtype=np.float32),
                "server_timing": {"infer_ms": 0.0}}


class FlashRTPolicy:
    def __init__(self, views, pool, layers, steps, cache_frames):
        from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx
        self.pipe = Pi05TorchFrontendRtx(
            a.ckpt, num_views=views, num_steps=steps,
            vision_pool_factor=pool, vision_num_layers=layers,
            cache_frames=cache_frames, use_cuda_graph=True)
        self.views = views
        self._prev_norm = None
        self._ws_rng = np.random.default_rng(7)
        self.pipe.set_prompt("pick up the black bowl on the stove")
        # calibration: 构建 graph/stream (合成帧即可, INT8 激活范围随帧更新无害)
        calib = {"image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                 "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                 "wrist_image_right": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
                 "state": np.random.uniform(-1, 1, (8,)).astype(np.float32)}
        self.pipe.calibrate_with_real_data([calib])
        log.info("calibration done")

    def infer(self, obs: dict) -> dict:
        if obs.get("reset"):
            self._prev_norm = None
        import time as _t
        _t0 = _t.perf_counter()
        prompt = obs.get("prompt", "")
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        st_n = ((state - Q01_S) / (Q99_S - Q01_S + 1e-6) * 2.0 - 1.0).astype(np.float32)
        # OFFICIAL-WEIGHTS FIX: pass RAW 8-dim normalized state; openpi official
        # conditioning digitizes the raw dims (32-pad = v044-era convention, 24 fake
        # digits for pi05_libero_base). Keep 32-pad only for v044-era checkpoints.
        _t1 = _t.perf_counter()
        # Memoize set_prompt: with an unchanged (prompt, state) the call is
        # pure overhead (tokenizer-cache lookup + embeds H2D + backend sync).
        # Skip it, but replicate its _frame_count reset so full/decode-only
        # cadence stays identical for any cache_frames.
        _sp_key = (prompt, st_n.tobytes())
        if obs.get("reset") or _sp_key != getattr(self, "_sp_key", None):
            self.pipe.set_prompt(prompt, state=st_n)
            self._sp_key = _sp_key
        else:
            self.pipe._frame_count = 0
        _t2 = _t.perf_counter()
        req = {"image": obs["observation/image"],
               "wrist_image": obs["observation/wrist_image"]}
        if self.views >= 3 and "observation/wrist_image_right" in obs:
            req["wrist_image_right"] = obs["observation/wrist_image_right"]
        noise = None
        if _WS_ALPHA > 0 and self._prev_norm is not None:
            sh = np.empty_like(self._prev_norm)
            rp = min(5, sh.shape[0])
            sh[: sh.shape[0] - rp] = self._prev_norm[rp:]
            sh[sh.shape[0] - rp:] = self._prev_norm[-1]
            sh32 = np.concatenate([sh, np.zeros((sh.shape[0], 32 - sh.shape[1]), dtype=np.float32)], axis=1)
            eps = self._ws_rng.standard_normal((sh.shape[0], 32)).astype(np.float32)
            noise = np.ascontiguousarray(_WS_ALPHA * sh32 + (1.0 - _WS_ALPHA) * eps, dtype=np.float32)
        # QF 2026-10-06: the block below was an exact duplicate of the
        # memoized prep above it (state norm + set_prompt + req build) and ran
        # UNCONDITIONALLY, defeating the memoization and re-uploading the
        # prompt embeds twice per request (the second staging round in the
        # nsys waterfall). Removing it is byte-identical: the second
        # set_prompt re-uploaded the same bytes.
        _t3 = _t.perf_counter()
        r = self.pipe.infer(req, noise=noise)
        _t4 = _t.perf_counter()
        self._ph = getattr(self, "_ph", [])
        self._ph.append((_t1 - _t0, _t2 - _t1, _t4 - _t3))
        if len(self._ph) % 30 == 0:
            import numpy as _np
            arr = _np.array(self._ph[-30:])
            print("PHASE prep=%.1fms set_prompt=%.1fms pipe_infer=%.1fms" % tuple(_np.median(arr, 0) * 1000), flush=True)
        a = np.asarray(r["actions"], dtype=np.float32)
        self._prev_norm = ((a - Q01_A) / (Q99_A - Q01_A + 1e-6) * 2.0 - 1.0).astype(np.float32)
        return {"actions": a}


async def _handler(ws, policy, packer, metadata):
    await ws.send(packer.pack(metadata))
    log.info("client connected: %s", ws.remote_address)
    while True:
        obs = msgpack_numpy.unpackb(await ws.recv())
        t0 = time.monotonic()
        try:
            out = policy.infer(obs)
        except Exception:
            await ws.send(traceback.format_exc())
            await ws.close(code=1011, reason="infer error")
            raise
        out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        await ws.send(packer.pack(out))


# ---------------------------------------------------------- uds 通道 (同机推荐)
# 帧格式: 4字节LE长度前缀 + msgpack 载荷; 连接后先发 metadata (与 ws 语义一致)。

async def _uds_handler(reader, writer, policy, packer, metadata):
    log.info("uds client connected: %s", writer.get_extra_info("peername"))
    try:
        _meta = packer.pack(metadata)
        writer.write(struct.pack("<I", len(_meta)) + _meta)
        await writer.drain()
        while True:
            (ln,) = struct.unpack("<I", await reader.readexactly(4))
            obs = msgpack_numpy.unpackb(await reader.readexactly(ln))
            t0 = time.monotonic()
            try:
                out = policy.infer(obs)
                out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
            except Exception:
                out = traceback.format_exc()  # 字符串镜像 ws 错误语义
            _b = packer.pack(out)
            writer.write(struct.pack("<I", len(_b)) + _b)
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


# ---------------------------------------------------------- shm 通道 (可选)
# /dev/shm mmap slot + FIFO 信号; 布局与信号协议单一来源: transport_bench/transport_common.py

def _shm_loop(policy, packer, metadata):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "transport_bench"))
    import transport_common as _tc

    ch = _tc.ShmChannel(a.shm_name, create=True)
    ch.server_open()
    ch.server_put_metadata(packer.pack(metadata))
    log.info("serving shm name=%s path=%s", a.shm_name, ch.path)
    last_seq = 0
    while True:
        seq, _n = ch.server_wait_req()
        if seq <= last_seq:
            continue
        last_seq = seq
        obs = msgpack_numpy.unpackb(ch.read_req())
        t0 = time.monotonic()
        try:
            out = policy.infer(obs)
            out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        except Exception:
            out = traceback.format_exc()
        ch.server_send_resp(packer.pack(out))


async def _main():
    if a.echo:
        policy = EchoPolicy()
        log.info("ECHO STUB mode: no model, no GPU (transport benchmarking only)")
    else:
        log.info("loading FlashRT (views=%d pool=%d layers=%d steps=%d)...",
                 a.views, a.pool, a.layers, a.steps)
        policy = FlashRTPolicy(a.views, a.pool, a.layers, a.steps, a.cache_frames)
    packer = msgpack_numpy.Packer()
    metadata = {"views": a.views, "pool": a.pool, "steps": a.steps}
    if a.echo:
        metadata["echo"] = True

    if a.transport == "shm":
        import threading
        threading.Thread(target=_shm_loop, args=(policy, packer, metadata), daemon=True).start()
        await asyncio.Event().wait()  # shm loop runs in its own thread
        return

    if a.transport == "uds":
        if os.path.exists(a.uds_path):
            os.unlink(a.uds_path)
        uds_srv = await asyncio.start_unix_server(
            lambda r, w: _uds_handler(r, w, policy, packer, metadata), path=a.uds_path)
        log.info("serving uds on %s", a.uds_path)
        async with uds_srv:
            await uds_srv.serve_forever()
        return

    async with _server.serve(lambda ws: _handler(ws, policy, packer, metadata),
                             "0.0.0.0", a.port, compression=None, max_size=None) as srv:
        log.info("serving on 0.0.0.0:%d", a.port)
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(_main())
