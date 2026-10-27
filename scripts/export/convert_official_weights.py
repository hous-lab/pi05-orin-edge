#!/usr/bin/env python3
"""Convert the official fp32 pi05_libero_base checkpoint into the bf16 bundle this repo deploys.

Source: lerobot/pi05_libero_base model.safetensors (812 tensors, F32). The tied
lm_head is declared in the safetensors __metadata__ as an alias entry:
  __metadata__["<embed_tokens key>"] = "<lm_head key>"
so the embedding table is not stored in the source file.

Output bundle contract (what the FlashRT overlay stack loads):
  - 813 tensors, all BF16 (fp32 -> bf16, round-to-nearest-even)
  - embed_tokens materialized from lm_head, inserted right after it (index 206)
  - lm_head kept, __metadata__ dropped
  - compact JSON header, insertion order, header padded with spaces so the data
    section starts 8-byte aligned
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

EMBED_KEY = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"
DTYPE_F32 = "F32"
DTYPE_BF16 = "BF16"
CHUNK_BYTES = 64 << 20  # stream big tensors in 64 MiB fp32 chunks


def f32_to_bf16(u32: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 round-to-nearest-even, bit-exact with torch .to(torch.bfloat16)."""
    u32 = u32.astype(np.uint32, copy=False)
    lsb = (u32 >> np.uint32(16)) & np.uint32(1)
    u16 = (u32 + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    return u16.astype(np.uint16)


def header_of(path: str) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def serialize_header(entries: list[tuple]) -> bytes:
    """Compact-JSON safetensors header, insertion order, padded to 8-byte align."""
    obj = {
        k: {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": v["data_offsets"]}
        for k, v in entries
    }
    js = json.dumps(obj, separators=(",", ":"))
    n = len(js)
    pad = (8 - ((8 + n) % 8)) % 8
    return js.encode() + b" " * pad


def build_plan(src_header: dict) -> list[tuple]:
    """Output plan: source order with embed_tokens inserted right after lm_head."""
    meta = src_header.get("__metadata__", {})
    alias = meta.get(EMBED_KEY)
    if alias != LM_HEAD_KEY:
        raise SystemExit(
            f"unexpected tied-weight alias in source: {alias!r} (expected {LM_HEAD_KEY!r}); "
            "if upstream changes the serialization, update EMBED_KEY/LM_HEAD_KEY"
        )
    keys = [k for k in src_header if k != "__metadata__"]
    plan = []
    for k in keys:
        plan.append((k, src_header[k]))
        if k == LM_HEAD_KEY:
            plan.append((EMBED_KEY, src_header[LM_HEAD_KEY]))
    if EMBED_KEY not in [p[0] for p in plan]:
        raise SystemExit("lm_head not found in source; cannot materialize embed_tokens")
    return plan


def convert(src_path: str, dst_path: str) -> dict:
    src_header = header_of(src_path)
    plan = build_plan(src_header)
    n_out = len(plan)
    print(f"source: {len(src_header) - 1} tensors (F32) -> output: {n_out} tensors (BF16)")

    # pass 1: build the output header (data_offsets accumulate over bf16 sizes)
    out_entries = []
    off = 0
    for key, info in plan:
        nbytes = 2
        for d in info["shape"]:
            nbytes *= d
        out_entries.append((key, {"dtype": "BF16", "shape": info["shape"],
                                  "data_offsets": [off, off + nbytes]}))
        off += nbytes
    hdr = serialize_header(out_entries)

    # pass 2: stream each tensor, cast, write
    with open(src_path, "rb") as src, open(dst_path, "wb") as dst:
        dst.write(struct.pack("<Q", len(hdr)))
        dst.write(hdr)
        base = 8 + struct.unpack("<Q", open(src_path, "rb").read(8))[0]
        for key, info in plan:
            s_off, s_end = info["data_offsets"]
            # seek per tensor: the plan visits lm_head twice (once as lm_head,
            # once as the materialized embed_tokens), so sequential reads would
            # run off the end of the file.
            src.seek(base + s_off)
            remaining = s_end - s_off
            while remaining > 0:
                chunk = min(CHUNK_BYTES, remaining)
                buf = src.read(chunk)
                if len(buf) < chunk:
                    raise SystemExit(f"short read on {key}")
                u32 = np.frombuffer(buf, dtype="<u4")
                if len(u32) % 2:
                    raise SystemExit(f"{key}: fp32 byte count not a multiple of 4")
                u16 = f32_to_bf16(u32)
                dst.write(u16.astype("<u2").tobytes())
                remaining -= chunk
    return {"tensors": n_out, "bytes": 8 + len(hdr) + off}


def verify(src_path: str, ref_path: str) -> int:
    """Per-tensor byte comparison of a converted file against a reference bundle."""
    if os.path.getsize(src_path) != os.path.getsize(ref_path):
        print("FAIL: file size mismatch")
        return 1
    h_a = header_of(src_path)
    h_b = header_of(ref_path)
    if h_a != h_b:
        keys_a = set(h_a)
        keys_b = set(h_b)
        only_a = keys_a - keys_b
        only_b = keys_b - keys_a
        print(f"FAIL: header mismatch; only-new={list(only_a)[:3]} only-ref={list(only_b)[:3]}")
        return 1
    base_a = 8 + struct.unpack("<Q", open(src_path, "rb").read(8))[0]
    base_b = 8 + struct.unpack("<Q", open(ref_path, "rb").read(8))[0]
    mism = 0
    for key, info in h_a.items():
        s_a, e_a = info["data_offsets"]
        s_b, e_b = h_b[key]["data_offsets"]
        n = e_a - s_a
        with open(src_path, "rb") as fa, open(ref_path, "rb") as fb:
            fa.seek(base_a + s_a)
            fb.seek(base_b + s_b)
            a = fa.read(n)
            b = fb.read(n)
            if a != b:
                mism += 1
                print(f"  tensor mismatch: {key}")
    if mism:
        print(f"FAIL: {mism} tensors differ")
        return 1
    print(f"PASS: {len(h_a)} tensors byte-identical to reference bundle")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="path to fp32 model.safetensors (lerobot/pi05_libero_base)")
    ap.add_argument("--out", required=True, help="output bundle path (model.safetensors, BF16)")
    ap.add_argument("--verify", default=None, metavar="REF",
                    help="after converting, compare per-tensor bytes against a reference bundle")
    args = ap.parse_args()
    info = convert(args.src, args.out)
    print(f"wrote {args.out}: {info['tensors']} tensors, {info['bytes']} bytes")
    if args.verify:
        with open(args.out, "rb") as f:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        print(f"sha256: {h.hexdigest()}")
        return verify(args.out, args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
