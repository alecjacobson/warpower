"""Write paired benchmark tables and exportable SVG figures without using the GPU."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from benchmarks.fetch_scenes import SCENES

DISTRIBUTIONS = ("uniform", "gradient", "cluster5", "cluster10")
SIZES = (100000, 500000, 1000000, 2000000, 5000000, 10000000, 15000000)
MODES = ("voronoi", "power")


def expected_cases():
    synthetic = {(d, n, mode) for d in DISTRIBUTIONS for n in SIZES for mode in MODES}
    return synthetic, {(scene, mode) for scene in SCENES for mode in MODES}


def load_pairs(root):
    rows, provenance = [], []
    paths = sorted((root / "warpower-l40/matrix").glob("*.json"))
    for path in paths:
        other = root / "paragram-l40/matrix" / path.name
        if not other.exists():
            continue
        w, p = json.loads(path.read_text()), json.loads(other.read_text())
        if (w["warmup"], w["repeats"]) != (3, 10):
            raise ValueError(f"{path.name}: expected three warm-ups and ten measured calls")
        for field in (
            "points_sha256",
            "weights_sha256",
            "distribution",
            "n",
            "seed",
            "weighted",
            "warmup",
            "repeats",
            "gpu",
            "timing",
        ):
            if p[field] != w[field]:
                raise ValueError(f"{path.name}: mismatched {field}")
        if p["versions"]["benchmark_source_sha256"] != w["versions"]["benchmark_source_sha256"]:
            raise ValueError(f"{path.name}: different benchmark harnesses")
        row = {
            "distribution": w["distribution"],
            "n": w["n"],
            "mode": "power" if w["weighted"] else "voronoi",
            "paragram_ms": p["mean_seconds"] * 1000,
            "warpower_ms": w["mean_seconds"] * 1000,
            "speedup": p["mean_seconds"] / w["mean_seconds"],
        }
        for backend, record in (("paragram", p), ("warpower", w)):
            row[f"{backend}_failed"] = sum(
                count
                for status, count in record["status_counts"].items()
                if status not in ("0", "3")
            )
            row[f"{backend}_empty"] = record["status_counts"].get("3", 0)
            row[f"{backend}_peak_gib"] = record["memory"]["peak_bytes"] / 1024**3
            provenance.append(
                {
                    "path": str((path if backend == "warpower" else other).relative_to(root)),
                    "sha256": hashlib.sha256(
                        (path if backend == "warpower" else other).read_bytes()
                    ).hexdigest(),
                    "versions": record["versions"],
                    "mempool_retain_mb": record["mempool_retain_mb"],
                }
            )
        rows.append(row)
    if not rows:
        raise ValueError("No matching completed benchmark pairs")
    sources = {
        item["versions"]["warpower_source_sha256"]
        for item in provenance
        if "warpower_source_sha256" in item["versions"]
    }
    if len(sources) != 1:
        raise ValueError("Results mix Warpower implementations; resume the matrix first")
    pools = {
        item["mempool_retain_mb"]
        for item in provenance
        if "warpower_source_sha256" in item["versions"]
    }
    if len(pools) != 1:
        raise ValueError("Results mix Warp allocator policies; resume the matrix first")
    rows.sort(key=lambda row: (row["distribution"], row["n"], row["mode"]))
    return rows, provenance


def plots(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    colors = {"paragram": "#666666", "warpower": "#1769aa"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, distribution in zip(axes.flat, DISTRIBUTIONS):
        for backend in colors:
            for mode, style in (("voronoi", "-"), ("power", "--")):
                group = sorted(
                    (r for r in rows if r["distribution"] == distribution and r["mode"] == mode),
                    key=lambda r: r["n"],
                )
                if group:
                    axis.loglog(
                        [r["n"] for r in group],
                        [r[f"{backend}_ms"] for r in group],
                        style,
                        marker="o",
                        markersize=4,
                        color=colors[backend],
                        label=f"{backend.title()} {mode}",
                    )
        axis.set(title=distribution, xlabel="Sites", ylabel="Construction time (ms)")
        axis.grid(True, which="both", alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle("NVIDIA L40 · fresh construction · mean of 10 runs")
    figure.savefig(output / "synthetic-time.svg")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, mode in zip(axes, MODES):
        for distribution in DISTRIBUTIONS:
            group = sorted(
                (r for r in rows if r["distribution"] == distribution and r["mode"] == mode),
                key=lambda r: r["n"],
            )
            if group:
                axis.semilogx(
                    [r["n"] for r in group],
                    [r["speedup"] for r in group],
                    marker="o",
                    label=distribution,
                )
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        axis.set(title=mode.title(), xlabel="Sites", ylabel="Paragram time / Warpower time")
        axis.grid(True, alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Synthetic speed ratios · above 1 favors Warpower")
    figure.savefig(output / "synthetic-speedup.svg")
    plt.close(figure)

    scenes = sorted({r["distribution"] for r in rows if r["distribution"] in SCENES})
    if scenes:
        figure, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
        lookup = {(r["distribution"], r["mode"]): r for r in rows}
        for mode, shift, color in (("voronoi", -0.2, "#1769aa"), ("power", 0.2, "#e68a2e")):
            values = [lookup.get((scene, mode), {}).get("speedup", math.nan) for scene in scenes]
            axis.bar(
                [i + shift for i in range(len(scenes))],
                values,
                width=0.4,
                label=mode.title(),
                color=color,
            )
        axis.axhline(1, color="black", linestyle="--", linewidth=1)
        axis.set(
            xticks=range(len(scenes)),
            xticklabels=scenes,
            ylabel="Paragram time / Warpower time",
            title="Public Radiant Foam scenes · above 1 favors Warpower",
        )
        axis.legend()
        figure.savefig(output / "scene-speedup.svg")
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/summary"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    rows, provenance = load_pairs(args.root)
    synthetic, scenes = expected_cases()
    found = {(r["distribution"], r["n"], r["mode"]) for r in rows}
    found_scenes = {(r["distribution"], r["mode"]) for r in rows}
    missing = [list(case) for case in sorted(synthetic - found)]
    missing += [list(case) for case in sorted(scenes - found_scenes)]
    if args.require_complete and missing:
        raise ValueError(f"Missing {len(missing)} paper benchmark pairs: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    pool = next(
        item["mempool_retain_mb"]
        for item in provenance
        if "warpower_source_sha256" in item["versions"]
    )
    with (args.output / "measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "complete": not missing,
        "paired_cases": len(rows),
        "missing_cases": missing,
        "geometric_mean_speedup": math.exp(sum(math.log(r["speedup"]) for r in rows) / len(rows)),
        "warpower_faster_cases": sum(r["speedup"] > 1 for r in rows),
        "paragram_failed_cells_total": sum(r["paragram_failed"] for r in rows),
        "warpower_failed_cells_total": sum(r["warpower_failed"] for r in rows),
        "provenance": provenance,
        "rows": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# NVIDIA L40 benchmark comparison",
        "",
        f"{'Complete' if not missing else 'Partial'} matrix: {len(rows)} paired cases; "
        f"{len(missing)} missing. Geometric mean time ratio: "
        f"{summary['geometric_mean_speedup']:.3f}×; "
        f"Warpower is faster in {summary['warpower_faster_cases']} cases.",
        "",
        "Ratios divide Paragram construction time by Warpower construction time; above 1 "
        "favors Warpower. Timings include allocation calls, hierarchy building, retries, "
        "and CSR output. Each case uses resident GPU inputs, three warm-ups, and ten "
        f"measured calls. Warp retains {pool / 1024:g} GiB in its allocator; "
        "PyTorch uses its normal cache.",
        "",
        "Failure counts exclude legitimately empty cells. A successful status does not "
        "prove geometric correctness. The methods and numerical limits are documented in "
        "[benchmark reproduction](../../docs/benchmarks.md) and "
        "[independent audits](../../docs/numerical-audit.md). "
        "Memory is an approximate device-wide NVML peak including contexts and allocator reserves.",
        "",
        "![Synthetic construction times](synthetic-time.svg)",
        "",
        "![Synthetic speed ratios](synthetic-speedup.svg)",
        "",
    ]
    if any(r["distribution"] in SCENES for r in rows):
        lines += ["![Scene speed ratios](scene-speedup.svg)", ""]
    lines += [
        "| Dataset | Sites | Mode | Paragram ms | Warpower ms | Ratio | Failed P / W | Empty P / W | Peak GiB P / W |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['distribution']} | {r['n']:,} | {r['mode']} | {r['paragram_ms']:.2f} | "
            f"{r['warpower_ms']:.2f} | {r['speedup']:.3f}× | {r['paragram_failed']} / "
            f"{r['warpower_failed']} | {r['paragram_empty']} / {r['warpower_empty']} | "
            f"{r['paragram_peak_gib']:.2f} / {r['warpower_peak_gib']:.2f} |"
        )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    plots(rows, args.output)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("rows", "provenance")}
        )
    )


if __name__ == "__main__":
    main()
