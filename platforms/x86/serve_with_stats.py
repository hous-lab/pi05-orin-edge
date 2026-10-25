#!/usr/bin/env python3
"""Run the production server UNMODIFIED, dump model-side latency stats at exit.

Passive instrumentation: captures the Pi05TorchFrontendRtx instance via an
__init__ patch and dumps get_latency_stats() + raw per-inference records on
SIGINT/exit. The server byte path is untouched - measurement only.

Env:
  FLASHRT_ROOT  engine tree to import flash_rt from (required)
  SERVER_PATH   server script to execute (required)
  STATS_OUT     path of the json dump; records land next to it as *.records.npz
"""
import os, sys, runpy, atexit, json, signal
import numpy as np

FLASHRT = os.environ["FLASHRT_ROOT"]
SERVER = os.environ["SERVER_PATH"]
STATS_OUT = os.environ["STATS_OUT"]

sys.path.insert(0, FLASHRT)
_instances = []

import flash_rt.frontends.torch.pi05_rtx as _m

_orig_init = _m.Pi05TorchFrontendRtx.__init__


def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    _instances.append(self)


_m.Pi05TorchFrontendRtx.__init__ = _patched_init


def _dump(*_):
    if not _instances:
        return
    p = _instances[-1]
    rec = np.asarray(p.latency_records, dtype=np.float64)
    stats = p.get_latency_stats()
    if rec.size:
        stats["p99_ms"] = float(np.percentile(rec, 99))
        stats["jitter_p95_minus_p50_ms"] = float(
            np.percentile(rec, 95) - np.percentile(rec, 50))
    json.dump(stats, open(STATS_OUT, "w"), indent=1)
    np.savez(STATS_OUT.replace(".json", "") + ".records.npz", latency_records=rec)
    print("STATS_DUMP count=%d p50=%.2f p95=%.2f p99=%.2f" % (
        rec.size,
        np.percentile(rec, 50) if rec.size else -1,
        np.percentile(rec, 95) if rec.size else -1,
        np.percentile(rec, 99) if rec.size else -1), flush=True)


atexit.register(_dump)
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

if __name__ == "__main__":
    sys.argv = [SERVER] + sys.argv[1:]
    runpy.run_path(SERVER, run_name="__main__")
