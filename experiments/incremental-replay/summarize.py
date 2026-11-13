"""Derive compact first-milestone evidence from retained local replay reports."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def measured(record, name):
    return [s for s in record["variants"][name] if s["repeat"] >= 0]


def aggregate(report):
    names = next(iter(report["records"].values()))["variants"]
    result = {}
    for name in names:
        samples = [s for r in report["records"].values() for s in measured(r, name)]
        successful = [s for s in samples if s["error"] is None]
        milliseconds = np.array([s["seconds"] * 1000 for s in successful])
        result[name] = dict(
            measured_samples=len(samples),
            successful_samples=len(successful),
            failed_samples=len(samples) - len(successful),
            reference_matches=sum(s.get("matches_reference") is True for s in samples),
            reference_mismatches=sum(s.get("matches_reference") is False for s in samples),
            previous_input_reference_matches=sum(
                s.get("previous_input_matches_reference") is True for s in samples
            ),
            exact_valid_alternatives=sum(
                s.get("exact_geometric_check", {}).get("valid") is True for s in samples
            ),
            cpu_throttled_samples=sum(s["cpu_throttled"] for s in samples),
            median_ms=float(np.median(milliseconds)) if len(milliseconds) else None,
            p95_ms=float(np.percentile(milliseconds, 95)) if len(milliseconds) else None,
            maximum_ms=float(np.max(milliseconds)) if len(milliseconds) else None,
            sum_of_transition_medians_ms=sum(
                r["summary"][name]["median_seconds"] * 1000
                for r in report["records"].values()
                if r["summary"][name]["median_seconds"] is not None
            )
            if successful
            else None,
            failure_kinds=dict(Counter(s["failure_kind"] for s in samples if s["error"])),
        )
    return result


def crossover(report):
    rows = []
    for record in report["records"].values():
        times = {
            name: values["reference_validated_median_seconds"]
            for name, values in record["summary"].items()
        }
        if any(value is None for value in times.values()):
            raise ValueError("crossover requires previous/current reference-validated samples")
        ratio = times["vtm-fragment"] / times["radfoam-incremental"]
        sample = measured(record, "vtm-fragment")[0]
        rows.append(
            dict(
                frame=record["frame"],
                event=record["metadata"]["event"],
                moved_fraction=record["motion"]["moved_fraction"],
                maximum_displacement=record["motion"]["maximum_displacement"],
                spacing_normalized_p95=record["motion"]["displacement_over_spacing_p95"],
                topology_removed_fraction=sample["topology_churn"]["removed_fraction"],
                vtm_reuse_accepted=sample["diagnostics"]["reuse_accepted"],
                vtm_wall_attempted=sample["diagnostics"]["wall_attempted"],
                vtm_wall_accepted=sample["diagnostics"]["wall_accepted"],
                **{name.replace("-", "_") + "_ms": value * 1000 for name, value in times.items()},
                vtm_over_native_incremental=ratio,
                winner_with_5_percent_margin=(
                    "native-reference"
                    if ratio > 1.05
                    else "vtm"
                    if ratio < 1 / 1.05
                    else "near-tie"
                ),
            )
        )
    return rows


def control_outcomes(report):
    return [
        dict(
            frame=record["frame"],
            event=record["metadata"]["event"],
            variants={
                name: dict(
                    successful_samples=summary["successful_samples"],
                    reference_match=summary["all_comparable_outputs_match_reference"],
                    exact_valid_alternatives=sum(
                        s.get("exact_geometric_check", {}).get("valid") is True
                        for s in measured(record, name)
                    ),
                    failures=summary["failures"],
                )
                for name, summary in record["summary"].items()
            },
        )
        for record in report["records"].values()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for stem in ["control-qualified", "vtm-pairs", "vtm-sequence", "radfoam-pairs", "capacity"]:
        path = args.raw / f"{stem}.json"
        report = json.loads(path.read_text())
        if report.get("complete") is not True:
            raise ValueError(f"incomplete report: {path}")
        summary = dict(
            report=dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path)),
            fixture_manifest_sha256=report["fixture_manifest_sha256"],
            fixture_limitations=report["fixture_limitations"],
            contract=report["contract"],
            mode=report["mode"],
            transitions=len(report["records"]),
            warmups=report["warmups"],
            repeats=report["repeats"],
            reference=report["reference"],
            reference_binary_sha256=report["reference_binary_sha256"],
            exact_checker_sha256=report["exact_checker_sha256"],
            timing_scope=report["timing_scope"],
            validation_scope=report["validation_scope"],
            warpower_commit=report["warpower_commit"],
            source_sha256=report["source_sha256"],
            versions=report["versions"],
            backends=aggregate(report),
        )
        if stem.startswith("vtm-"):
            rows = crossover(report)
            summary["winner_counts_with_5_percent_margin"] = dict(
                Counter(row["winner_with_5_percent_margin"] for row in rows)
            )
            summary["scope"] = (
                "Exploratory native-reference crossover on this captured trajectory, with "
                "unchanged VTM certification. Native does not provide the same certificate. "
                "Offline motion/churn analysis is not a measured routing policy. "
                "Sum of transition medians excludes initialization and offline work."
            )
            with (args.output / f"{stem}.csv").open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        if stem in ["control-qualified", "capacity"]:
            summary["outcomes"] = control_outcomes(report)
        if stem == "radfoam-pairs":
            summary["output_comparisons"] = [
                dict(
                    frame=record["frame"],
                    source_step=record["metadata"]["source_step"],
                    sites=record["metadata"]["active_count"],
                    all_samples_successful=all(
                        s["error"] is None
                        for name in record["variants"]
                        for s in measured(record, name)
                    ),
                    identical_output_hashes=len(
                        {
                            s["output_sha256"]
                            for name in record["variants"]
                            for s in measured(record, name)
                        }
                    )
                    == 1,
                    backend_summary=record["summary"],
                )
                for record in report["records"].values()
            ]
            summary["scope"] = (
                "Bounded adjacency only; cross-backend hash equality and zero statuses "
                "are not an independent exact completeness certificate."
            )
        summaries[stem] = summary
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
