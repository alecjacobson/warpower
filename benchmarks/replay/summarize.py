"""Compact, reproducible summaries of complete representative replay reports."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.replay.fixture import digest


def geometrically_verified(sample):
    return sample.get("error") is None and (
        sample.get("matches_reference") is True
        or sample.get("exact_geometric_check", {}).get("valid") is True
    )


def summarize(path):
    report = json.loads(path.read_text())
    if not report.get("complete"):
        raise ValueError(f"Incomplete benchmark: {path}")
    full = report["contract"] == "full-hull-tetrahedra"
    groups = {"all": list(report["records"].values())}
    if "radfoam" in report["fixture_name"] and full:
        groups.update(
            optimizer=[
                r for r in groups["all"] if r["metadata"]["stage"] == "optimizer-geometry-refresh"
            ],
            growth_or_reset=[
                r for r in groups["all"] if r["metadata"]["stage"] != "optimizer-geometry-refresh"
            ],
        )
    output = dict(
        report=path.name,
        report_sha256=digest(path),
        fixture_manifest_sha256=report["fixture_manifest_sha256"],
        contract=report["contract"],
        repeats=report["repeats"],
        groups={},
        timing_scope=report["timing_scope"],
    )
    for group, rows in groups.items():
        if not rows:
            continue
        variants = {}
        for name in rows[0]["variants"]:
            samples = [s for row in rows for s in row["variants"][name] if s["repeat"] >= 0]
            successful = [s for s in samples if s["error"] is None]
            medians = [row["summary"][name]["median_seconds"] for row in rows]
            complete = all(value is not None for value in medians)
            variants[name] = dict(
                samples=len(samples),
                successful_samples=len(successful),
                reference_matches=sum(s.get("matches_reference") is True for s in samples)
                if full
                else None,
                previous_inputs_verified=sum(
                    s.get("previous_input_matches_reference") is True for s in samples
                )
                if full
                else None,
                geometrically_verified_samples=sum(geometrically_verified(s) for s in samples)
                if full
                else None,
                previous_inputs_geometrically_verified=sum(
                    geometrically_verified(s.get("previous_input") or {})
                    or s.get("previous_input_matches_reference") is True
                    for s in samples
                )
                if full
                else None,
                failures=sorted({s["error"] for s in samples if s["error"] is not None}),
                throttled_samples=sum(s["cpu_throttled"] for s in samples),
                median_update_ms=float(np.median(medians)) * 1000 if complete else None,
                sum_pair_medians_ms=float(sum(medians)) * 1000 if complete else None,
                p95_sample_ms=float(np.percentile([s["seconds"] for s in successful], 95)) * 1000
                if successful
                else None,
                max_hull_cache_bytes=max(
                    (s["diagnostics"].get("hull_cache_bytes", 0) for s in successful), default=0
                ),
            )
        output["groups"][group] = dict(
            pairs=len(rows),
            variants=variants,
            peer_output_sets_agree=all(
                len(
                    {
                        s["output_sha256"]
                        for values in row["variants"].values()
                        for s in values
                        if s["repeat"] >= 0 and s["error"] is None
                    }
                )
                == 1
                for row in rows
            ),
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in sorted(args.directory.glob("*.json")):
        if path.name == "invocations.json":
            continue
        results.append(summarize(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                aggregation="Median across pair medians; p95 across measured samples. Sum is the selected updates' medians, not a trajectory/training time. Failed rows are never assigned zero time.",
                cases=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
