"""Measure benchmark-local sharing of transient FP64 clipping storage (#122).

Every constructor keeps its own hierarchy, connectivity, retained fragments,
strategy hints and certificate. The saved clipping cells are only kernel-local
work; the existing builders already overwrite this storage between chunks.
"""

import argparse
import contextlib
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

import warp as wp

import warpower
from warpower import _graph, _mesher


@contextlib.contextmanager
def share():
    """Setup-only experiment for one serial complete-mesher constructor family."""
    original = _graph.allocate_scratch
    owner = dict(scratch=None, requests=[], shared_bytes=0, separate_bytes=0)

    def allocate(n, capacity=64, device="cuda:0", robust=False):
        if not robust or capacity != 1024:
            return original(n, capacity, device, robust)
        size = ((n + 31) // 32) * 32 * capacity
        owner["requests"].append(dict(cells=n, slots=size, capacity=capacity))
        owner["separate_bytes"] += size * 100
        if owner["scratch"] is None:
            owner["scratch"] = original(n, capacity, device, robust)
            owner["shared_bytes"] = size * 100
        scratch = owner["scratch"]
        if len(scratch.planes) < size or scratch.planes.device != wp.get_device(device):
            raise ValueError("First scratch allocation does not cover later compatible requests")
        return scratch

    _graph.allocate_scratch = allocate
    try:
        yield owner
    finally:
        _graph.allocate_scratch = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--kind", choices=["benchmark", "gallery"], default="benchmark")
    parser.add_argument("--scratch-report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.scratch_report.exists():
        parser.error("Report already exists")
    base = _mesher.VariationalMeshingWorkspace
    observations = []

    class Measured(base):
        def __init__(self, domain, *a, **kw):
            before = wp.get_mempool_used_mem_current(domain.device)
            start = time.perf_counter()
            sharing = None
            if args.shared:
                with share() as sharing:
                    super().__init__(domain, *a, **kw)
            else:
                super().__init__(domain, *a, **kw)
            observations.append(
                dict(
                    constructor_seconds=time.perf_counter() - start,
                    persistent_pool_bytes=wp.get_mempool_used_mem_current(domain.device) - before,
                    sharing=None
                    if sharing is None
                    else {k: v for k, v in sharing.items() if k != "scratch"},
                )
            )
            # Constructors and graphs retain the arrays themselves. Also retain
            # this explicit owner so the experimental ownership is reviewable.
            self._experimental_clipping_owner = sharing

    report = dict(
        scope=__doc__,
        shared=args.shared,
        completed=False,
        source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), *sorted(Path("warpower").glob("*.py"))]
        },
        observations=observations,
    )
    try:
        warpower.VariationalMeshingWorkspace = Measured
        _mesher.VariationalMeshingWorkspace = Measured
        sys.argv = [sys.argv[0], *remaining]
        runpy.run_module(
            "benchmarks.mesher" if args.kind == "benchmark" else "examples.vtm_gallery",
            run_name="__main__",
        )
        report["completed"] = True
    finally:
        warpower.VariationalMeshingWorkspace = base
        _mesher.VariationalMeshingWorkspace = base
        args.scratch_report.parent.mkdir(parents=True, exist_ok=True)
        args.scratch_report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
