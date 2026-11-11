"""Plot full-fit quality versus elapsed training-loop time outside Git evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--warp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--radfoam", type=Path, default=Path("results/validation/radfoam/full-budget/analysis.json")
    )
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(9.4, 5.8), layout="constrained")
    baseline = json.loads(args.radfoam.read_text())["learning_curves"]
    curves = []
    for name, label, color in [
        ("native", "RadFoam / native", "#9298a1"),
        ("warpower", "RadFoam / Warp geometry", "#c38435"),
    ]:
        rows = baseline[name]
        curves.append(
            (
                label,
                color,
                "--",
                [r["elapsed_seconds"] / 60 for r in rows],
                [r["official_psnr"] for r in rows],
            )
        )
    for path, label, color in [
        (args.native, "VoroTracing / native", "#316bb5"),
        (args.warp, "VoroTracing / Warp", "#148563"),
    ]:
        report = json.loads((path / "report.json").read_text())
        assert report["complete"], "Only plot complete fits"
        rows = report["evaluations"]
        curves.append(
            (
                label,
                color,
                "-",
                [r["seconds"] / 60 for r in rows],
                [r["metrics"]["psnr"] for r in rows],
            )
        )
    for label, color, style, times, quality in curves:
        ax.plot(times, quality, style, color=color, linewidth=2.1, label=label)
        ax.scatter(times[-1], quality[-1], s=38, color=color, zorder=4)
    ax.set(
        xlabel="Training-loop time (minutes)",
        ylabel="Held-out PSNR (dB; higher is better)",
        ylim=(28, max(max(c[4]) for c in curves) + 0.35),
        xlim=(0, max(max(c[3]) for c in curves) * 1.03),
    )
    ax.set_title("Bonsai scene fitting on one NVIDIA L40", loc="left", weight="bold", pad=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", alpha=0.18)
    ax.legend(loc="lower right", frameon=False)
    fig.supxlabel(
        "37 half-resolution test views · one fit per backend · original PSNR metric\n"
        "Time includes validation; RoMa preprocessing is separate. Recipes differ between methods.",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
