"""Replay recorded fitting updates through fresh and incremental geometry backends."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from integration import initialize_runtime, native

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical


def timed(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - started, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    initialize_runtime()
    metadata = json.loads(args.trajectory.with_suffix(".json").read_text())
    with np.load(args.trajectory) as source:
        frames = [source[f"points_{i}"] for i in range(len(metadata))]
    candidates = [
        i
        for i in range(1, len(frames))
        if frames[i].shape == frames[i - 1].shape and metadata[i]["incremental"]
    ]
    selected = []
    for size in [9216, 85000, 170000, 254000]:
        selected.append(
            min(
                candidates,
                key=lambda i: (
                    abs(len(frames[i]) - size),
                    metadata[i]["step"] - metadata[i - 1]["step"],
                ),
            )
        )
    records = []
    for index in sorted(set(selected)):
        a, b = [torch.tensor(frames[i], device="cuda") for i in [index - 1, index]]
        previous = warpower.voronoi_diagram(a)
        assert torch.all(previous.status == 0)
        names = ["native-full", "native-incremental", "warpower-full", "warpower-hints"]
        samples = {name: [] for name in names}
        outputs, failures = {}, {}
        quota_before = cpu_quota_snapshot()
        for repeat in range(args.repeats + 1):
            order = names if repeat % 2 == 0 else names[::-1]
            for name in order:
                if name in failures:
                    continue
                try:
                    if name == "native-full":
                        seconds, tri = timed(lambda: native.Triangulation(b))
                        permutation = tri.permutation().long()
                    elif name == "native-incremental":
                        tri = native.Triangulation(a)
                        permutation = tri.permutation().long().clone()
                        current = b[permutation].contiguous()
                        seconds, permuted = timed(lambda: tri.rebuild(current, incremental=True))
                        assert not permuted
                    else:
                        kwargs = (
                            dict(
                                initial_guesses=previous.adjacency,
                                initial_guesses_offsets=previous.offsets,
                            )
                            if name == "warpower-hints"
                            else {}
                        )
                        seconds, diagram = timed(lambda: warpower.voronoi_diagram(b, **kwargs))
                        assert torch.all(diagram.status == 0)
                    if repeat:
                        samples[name].append(seconds)
                    if repeat == args.repeats:
                        if name.startswith("native"):
                            outputs[name] = dict(
                                tets=canonical(permutation[tri.tets().long()].cpu().numpy()),
                                edges=tri.point_adjacency().numel(),
                            )
                        else:
                            outputs[name] = dict(edges=diagram.adjacency.numel(), status=0)
                except (RuntimeError, AssertionError) as error:
                    failures[name] = repr(error)
        quota = cpu_quota_delta(quota_before, cpu_quota_snapshot())
        row = dict(
            frame=index,
            previous_step=metadata[index - 1]["step"],
            step=metadata[index]["step"],
            sites=len(b),
            input_sha256=hashlib.sha256(frames[index].tobytes()).hexdigest(),
            maximum_displacement=float(
                np.linalg.norm(frames[index] - frames[index - 1], axis=1).max()
            ),
            cpu_quota=quota,
            cpu_throttled=any(x["throttled_periods"] for x in quota.values()),
            variants={
                name: dict(
                    samples_seconds=values,
                    median_seconds=float(np.median(values)) if values else None,
                    directed_edges=outputs.get(name, {}).get("edges"),
                    failure=failures.get(name),
                )
                for name, values in samples.items()
            },
            native_incremental_matches_full_tetrahedra=(
                bool(
                    np.array_equal(
                        outputs["native-full"]["tets"], outputs["native-incremental"]["tets"]
                    )
                )
                if "native-full" in outputs and "native-incremental" in outputs
                else None
            ),
        )
        records.append(row)
        report = dict(
            scope="Identical original-order recorded Warpower fitting sites. One warm-up and seven alternating-order samples; previous-state setup and readbacks excluded. Recorded frames may skip intermediate geometry refreshes; step gaps are explicit. Native computes a full tetrahedralization plus adjacency; Warpower returns bounded Voronoi adjacency. Zero status is not a full graph certificate; graph equivalence between those contracts is not asserted.",
            records=records,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
