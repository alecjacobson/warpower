"""Trace a changing-input producer/geometry/consumer loop; summarize only safe counters."""

import argparse
import collections
import hashlib
import json
import re
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dot", type=Path, required=True)
    parser.add_argument("--analyze", type=Path, help="Nsight exported SQLite database")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--optimizer", action="store_true", help="Audit three ODT iterations per replay"
    )
    args = parser.parse_args()
    if args.analyze:
        database = sqlite3.connect(args.analyze)
        tables = {
            row[0] for row in database.execute("select name from sqlite_master where type='table'")
        }

        def count(table):
            return (
                database.execute(f"select count(*) from {table}").fetchone()[0]
                if table in tables
                else 0
            )

        calls = dict(
            database.execute(
                "select value,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r "
                "join StringIds s on r.nameId=s.id group by value"
            )
        )
        dot = args.dot.read_text()
        node_types = dict(collections.Counter(re.findall(r'label="\{\s*([A-Z0-9_]+)\s*\n', dot)))
        copy_kinds = dict(collections.Counter(re.findall(r"\{kind \| ([^}]+)\}", dot)))
        report = {
            "scope": (
                "Three warmed replays of a changing 1000-site producer/ODT/consumer graph, "
                "three optimizer iterations each; one final synchronization outside construction"
                if args.optimizer
                else "Three warmed replays of a changing 1000-site weighted producer/diagram/tetrahedra/consumer graph; one final synchronization outside construction"
            ),
            "gpu_kernel_activities": count("CUPTI_ACTIVITY_KIND_KERNEL"),
            "gpu_memset_activities": count("CUPTI_ACTIVITY_KIND_MEMSET"),
            "gpu_memcpy_activities": count("CUPTI_ACTIVITY_KIND_MEMCPY"),
            "cuda_api_calls": calls,
            "dot_node_types": node_types,
            "dot_copy_kinds": copy_kinds,
            "copy_trace_limit": "Nsight omitted activity records for graph DtoD copies; DOT node directions establish that all captured copies stay on device.",
            "dot_sha256": hashlib.sha256(args.dot.read_bytes()).hexdigest(),
            "sqlite_sha256": hashlib.sha256(args.analyze.read_bytes()).hexdigest(),
        }
        assert report["gpu_memcpy_activities"] == 0
        assert not any(re.search(r"(alloc|free|memcpy)", name, re.I) for name in calls)
        assert set(node_types) <= {"KERNEL", "MEMSET", "MEMCPY", "CONDITIONAL"}
        assert sum(copy_kinds.values()) == node_types.get("MEMCPY", 0)
        assert all(kind == "DtoD (DEVICE to DEVICE)" for kind in copy_kinds)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return

    import numpy as np
    import torch
    import warp as wp

    import warpower
    from tests.test_graph import advance, consume, produce

    p = np.random.default_rng(2026).uniform(-1, 1, (1000, 3)).astype(np.float32)
    base = wp.array(p, dtype=wp.vec3, device="cuda:0")
    points = wp.clone(base)
    weights = wp.zeros(len(p), dtype=float, device="cuda:0")
    phase = wp.zeros(1, dtype=int, device="cuda:0")
    volume = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    workspace = (
        warpower.ODTWorkspace(points, max_iterations=3)
        if args.optimizer
        else warpower.DiagramWorkspace(points, weights, return_tetrahedra=True)
    )
    with wp.ScopedCapture() as capture:
        wp.launch(produce, len(p), inputs=[base, points, weights, phase])
        workspace.run()
        volume.zero_()
        wp.launch(
            consume,
            len(workspace.tetrahedra),
            inputs=[
                points,
                workspace.tetrahedra,
                workspace.tetrahedra_count,
                workspace.error,
                volume,
            ],
        )
        wp.launch(advance, 1, inputs=[phase])
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    assert workspace.error.numpy()[0] == 0
    if args.optimizer:
        assert workspace.iterations.numpy()[0] == 3
    args.dot.parent.mkdir(parents=True, exist_ok=True)
    wp.capture_debug_dot_print(capture.graph, str(args.dot), verbose=True)
    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    assert workspace.error.numpy()[0] == 0
    if args.optimizer:
        assert workspace.iterations.numpy()[0] == 3
    print("Valid final tetrahedra:", int(workspace.tetrahedra_count.numpy()[0]))


if __name__ == "__main__":
    main()
