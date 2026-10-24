# data/ — LIBERO setup (upstream link-only)

The closed-loop eval runs LIBERO-10 in simulation on the x86 host; no demo data is needed —
the benchmark suite definition (BDDL files + init states) ships inside the LIBERO package.

Upstream sources (link-only, nothing committed here):

- LIBERO benchmark: [github.com/Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- The eval scripts follow the openpi LIBERO example protocol
  ([github.com/Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi),
  `examples/libero/`), which vendors LIBERO at `third_party/libero`.
- The openpi-client wheel (server protocol client) comes from PyPI.

Practical setup on x86 (what we use): clone openpi, install the LIBERO deps into the eval
venv, then run the eval with `PYTHONPATH=<openpi>/third_party/libero`. See [docs/pipeline_guide.md](../docs/pipeline_guide.md) for the full command lines.
