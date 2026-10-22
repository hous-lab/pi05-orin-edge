"""Shared transport primitives for pi05 client<->server same-machine channels.

Framings:
  tcp/uds: 4-byte LE uint32 length prefix + msgpack payload.
  ws:      websockets library framing (msgpack payload), unchanged from production.
  shm:     /dev/shm file (mmap) + 2 FIFOs for signaling. Payload is msgpack bytes
           written into a fixed slot (keeps protocol semantics identical to ws).
"""
import mmap
import os
import struct

import numpy as np

# ---------------------------------------------------------------- msgpack numpy
# Identical semantics to openpi_client.msgpack_numpy (vendored copy, inlined so
# bench tools have no openpi dependency).


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


try:
    import msgpack

    Packer = None  # set below; functools.partial style
    import functools

    Packer = functools.partial(msgpack.Packer, default=pack_array)
    packb = functools.partial(msgpack.packb, default=pack_array)
    Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
    unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
except ImportError:  # pragma: no cover
    msgpack = None

# ---------------------------------------------------------------- length-prefix framing


def send_framed(sock, payload: bytes):
    sock.sendall(struct.pack("<I", len(payload)) + payload)


def recv_exact(sock, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        chunk = sock.recv_into(view[got:])
        if chunk == 0:
            raise ConnectionError("peer closed")
        got += chunk
    return bytes(buf)


def recv_framed(sock) -> bytes:
    (n,) = struct.unpack("<I", recv_exact(sock, 4))
    return recv_exact(sock, n)


# ---------------------------------------------------------------- shm transport
# /dev/shm/<name>  layout:
#   ctrl (64 bytes, uint64 LE):
#     0:  magic 0x70693035 ("pi05")
#     8:  req_seq    (written by client)
#     16: req_len
#     24: resp_seq   (written by server)
#     32: resp_len
#     40: shutdown
#   4096:                       request  slot (REQ_CAP bytes)
#   4096 + REQ_CAP:             response slot (RESP_CAP bytes)
# FIFOs: <fifo_dir>/<name>_req.fifo (client->server), <name>_resp.fifo (server->client).
# One byte written per signal; blocking os.read = syscall barrier, no busy polling.

SHM_MAGIC = 0x70693035
SHM_CTRL = 4096  # ctrl fields live in first 64B of the mapping anyway; keep payloads page-aligned
REQ_CAP = 4 * 1024 * 1024
RESP_CAP = 1 * 1024 * 1024
SHM_SIZE = SHM_CTRL + REQ_CAP + RESP_CAP
FIFO_DIR = "/tmp"


class ShmChannel:
    """One shm slot pair + 2 FIFOs. Server creates; client attaches. Single client."""

    def __init__(self, name: str, create: bool):
        self.name = name
        self.path = f"/dev/shm/{name}"
        self.req_fifo = f"{FIFO_DIR}/{name}_req.fifo"
        self.resp_fifo = f"{FIFO_DIR}/{name}_resp.fifo"
        if create:
            if os.path.exists(self.path):
                os.unlink(self.path)
            for p in (self.req_fifo, self.resp_fifo):
                if os.path.exists(p):
                    os.unlink(p)
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, SHM_SIZE)
            except Exception:
                os.close(fd)
                raise
            os.close(fd)
            os.mkfifo(self.req_fifo, 0o600)
            os.mkfifo(self.resp_fifo, 0o600)
        fd = os.open(self.path, os.O_RDWR)
        self._mm = mmap.mmap(fd, SHM_SIZE)
        os.close(fd)
        (magic,) = struct.unpack_from("<Q", self._mm, 0)
        if magic != SHM_MAGIC:
            struct.pack_into("<Q", self._mm, 0, SHM_MAGIC)
            struct.pack_into("<QQQQQ", self._mm, 8, 0, 0, 0, 0, 0)
        self._ctrl = memoryview(self._mm)[:64]
        self._req_slot = memoryview(self._mm)[SHM_CTRL : SHM_CTRL + REQ_CAP]
        self._resp_slot = memoryview(self._mm)[SHM_CTRL + REQ_CAP :]
        self._req_fd = self._resp_fd = None
        self._req_hold = None

    # -- raw ctrl access
    def _u64(self, off):
        return struct.unpack_from("<Q", self._mm, off)[0]

    def _put(self, off, val):
        struct.pack_into("<Q", self._mm, off, val)

    @staticmethod
    def _set_blocking(fd, blocking: bool):
        import fcntl

        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        if blocking:
            flags &= ~os.O_NONBLOCK
        else:
            flags |= os.O_NONBLOCK
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    # -- server side
    def server_open(self):
        # open nonblocking (avoid open() deadlock), then flip both to blocking:
        # server blocks on req FIFO read, client blocks on resp FIFO read.
        self._req_fd = os.open(self.req_fifo, os.O_RDONLY | os.O_NONBLOCK)
        # hold-open writer on req FIFO: without it, read() returns EOF (b"")
        # forever once the client disconnects -> 100% CPU spin.
        self._req_hold = os.open(self.req_fifo, os.O_WRONLY | os.O_NONBLOCK)
        self._resp_fd = os.open(self.resp_fifo, os.O_RDWR | os.O_NONBLOCK)
        self._set_blocking(self._req_fd, True)
        self._set_blocking(self._resp_fd, True)

    def server_put_metadata(self, payload: bytes):
        """Publish connect-time metadata (production protocol sends it on connect)."""
        n = len(payload)
        self._resp_slot[:n] = payload
        self._put(32, n)
        self._put(24, self._u64(24) + 1)  # resp_seq -> 1; client's first wait gets metadata
        # no FIFO signal: client polls seq at open (client_open handles it)

    def server_wait_req(self):
        os.read(self._req_fd, 1)  # blocks until client signals
        return self._u64(8), self._u64(16)  # req_seq, req_len

    def server_send_resp(self, payload: bytes):
        n = len(payload)
        self._resp_slot[:n] = payload
        self._put(32, n)  # resp_len
        self._put(24, self._u64(24) + 1)  # resp_seq bump
        os.write(self._resp_fd, b"x")

    # -- client side
    def client_open(self, timeout_s: float = 30.0):
        # retry until server has created+opened its ends (ENXIO = no reader yet)
        import time

        t0 = time.monotonic()
        while True:
            try:
                self._req_fd = os.open(self.req_fifo, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as e:
                if e.errno not in (2, 6):  # ENOENT, ENXIO
                    raise
                if time.monotonic() - t0 > timeout_s:
                    raise TimeoutError(f"server FIFO not ready: {self.req_fifo}")
                time.sleep(0.05)
        self._resp_fd = os.open(self.resp_fifo, os.O_RDONLY | os.O_NONBLOCK)
        # drain stale bytes from a previous session, then flip to blocking
        try:
            while True:
                os.read(self._resp_fd, 4096)
        except BlockingIOError:
            pass
        self._set_blocking(self._resp_fd, True)

    def client_read_metadata(self) -> bytes:
        """Consume the connect-time metadata written by server_put_metadata."""
        import time

        while self._u64(24) == 0:  # server hasn't published yet
            time.sleep(0.002)
        n = self._u64(32)
        return bytes(self._resp_slot[:n])

    def client_send_req(self, payload: bytes):
        n = len(payload)
        if n > REQ_CAP:
            raise ValueError(f"payload {n} > REQ_CAP {REQ_CAP}")
        self._req_slot[:n] = payload
        self._put(16, n)  # req_len
        self._put(8, self._u64(8) + 1)  # req_seq bump (FIFO write below is the barrier)
        os.write(self._req_fd, b"x")

    def client_wait_resp(self):
        os.read(self._resp_fd, 1)  # blocks (blocking fd) until server signals
        return self._u64(24), self._u64(32)  # resp_seq, resp_len

    def read_req(self) -> bytes:
        n = self._u64(16)
        return bytes(self._req_slot[:n])

    def read_resp(self) -> bytes:
        n = self._u64(32)
        return bytes(self._resp_slot[:n])

    def close(self):
        for fd in (self._req_fd, self._resp_fd, self._req_hold):
            if fd:
                os.close(fd)
        self._req_fd = self._resp_fd = self._req_hold = None
        self._mm.close()

    def destroy(self):
        for p in (self.path, self.req_fifo, self.resp_fifo):
            try:
                os.unlink(p)
            except FileNotFoundError:
                pass
