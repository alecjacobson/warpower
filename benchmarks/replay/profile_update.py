"""Nsight range for a warmed, single geometry transition (not benchmark timing)."""

import argparse
import ctypes
import json
import sys
from pathlib import Path

from benchmarks.replay.backends import CONTRACTS, create_backend
from benchmarks.replay.fixture import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--transition", type=int, required=True)
    parser.add_argument("--backend", choices=list(CONTRACTS), default="vtm-fragment")
    parser.add_argument("--native-module-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = Fixture(args.fixture)
    backend = create_backend(args.backend, native_module_dir=args.native_module_dir)
    previous, current = [
        backend.prepare(fixture.frame(i)) for i in [args.transition - 1, args.transition]
    ]
    for _ in range(2):
        backend.update(previous)
        backend.update(current)
        backend.synchronize()
    backend.update(previous)
    backend.synchronize()
    runtime = ctypes.CDLL(
        str(
            next(
                Path(sys.prefix).glob(
                    "lib/python*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
                )
            )
        )
    )
    for name in ["cudaProfilerStart", "cudaProfilerStop"]:
        getattr(runtime, name).argtypes = []
        getattr(runtime, name).restype = ctypes.c_int
    assert runtime.cudaProfilerStart() == 0
    backend.update(current)
    backend.synchronize()
    assert runtime.cudaProfilerStop() == 0
    result = backend.readback()
    report = dict(
        fixture_sha256=fixture.sha256,
        transition=args.transition,
        backend=args.backend,
        error=result.error,
        rows=len(result.rows),
        diagnostics=result.diagnostics,
        scope="Instrumented single warmed geometry update, including synchronization; use replay runner for clean timing",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
