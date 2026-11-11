"""Assemble the completed real-scene integration and update comparison results."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("results/local/radfoam-fitting-20261110")


def read(path):
    return json.loads(path.read_text())


def main():
    runs = {
        name: read(ROOT / f"bonsai-{name}-20k" / "report.json")
        for name in ["native", "warpower", "warpower-morton"]
    }
    ordering = read(ROOT / "ordering-validated.json")
    geometry = read(ROOT / "geometry-replay.json")
    updates = read(ROOT / "vtm-updates.json")
    assert len({r["initial_sha256"] for r in runs.values()}) == 1
    fitting = []
    for name, run in runs.items():
        audit = run["raw_render_gradient_checks"]
        assert run["complete"] and run["checkpoint_continuation"]
        assert all(v == 0 for k, v in audit.items() if k.startswith("nonfinite"))
        fitting.append(
            dict(
                backend=name,
                seconds=run["fit_wall_seconds"],
                test_psnr=run["evaluations"][-1]["mean_test_psnr"],
                final_sites=run["final_points"],
                geometry_seconds=run["stages_seconds"]["geometry"],
                forward_seconds=run["stages_seconds"]["forward"],
                backward_seconds=run["stages_seconds"]["backward"],
                cpu_throttled=run["cpu_throttled"],
                raw_checks=audit,
            )
        )
    with (ROOT / "fitting-summary.csv").open("w") as stream:
        fields = [k for k in fitting[0] if k != "raw_checks"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(fitting)
    report = dict(
        scope="Complete reduced-budget Mip-NeRF360 Bonsai fitting: 20,000 optimizer steps, 65,536 rays/step, SH3, 8,192 SfM sites plus 1,024 initial guards, 262,144 target sites, 255 training views and 37 held-out views at 780x520. One end-to-end run per backend; synchronized profiling and raw numerical checks enabled equally. Native incremental geometry versus fresh Warpower geometry; repeated component measurements are separate.",
        fitting=fitting,
        native_over_warp_morton_wall_speedup=(
            runs["native"]["fit_wall_seconds"] / runs["warpower-morton"]["fit_wall_seconds"]
        ),
        unordered_over_warp_morton_wall_speedup=(
            runs["warpower"]["fit_wall_seconds"] / runs["warpower-morton"]["fit_wall_seconds"]
        ),
        exact_same_scene_ordering=ordering,
        geometry_replay=geometry,
        vtm_updates=updates,
        limitations=[
            "Reduced site/iteration/ray budgets and eight-view densification error maps; this is not a paper-quality reproduction.",
            "The same initial physical sites and ray RNG seed do not force identical optimization: reordering changes floating-point reductions and later stochastic densification choices. Final site counts and quality are reported separately.",
            "One full run per backend cannot establish a general end-to-end speedup. Repeated same-input component benchmarks provide stronger evidence for their measured portions.",
            "Warpower supplies bounded Voronoi adjacency; native RadFoam also constructs an unbounded full-hull tetrahedralization. Previous exact checkpoint audits found seven incorrect undirected Warpower pairs across four larger scenes. Successful fitting and zero numerical failures do not close that correctness issue.",
            "RadFoam incremental updates do not supply VTM's complete exact/symbolic certificate. CGAL comparisons on these recorded transitions are independent evidence for these inputs, not equivalence of runtime correctness contracts.",
            "CUDA default stream only. The experiment does not establish correctness for arbitrary stream interleaving.",
            "Density is an optical attribute, not a power-diagram weight; weighted-cell scene fitting is not tested.",
        ],
        commands=[
            ".venv/bin/python experiments/radfoam-integration/fetch_bonsai.py",
            "CUDA_HOME=/usr/local/cuda-12.6 .venv/bin/python experiments/radfoam-integration/build_headless.py",
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python experiments/radfoam-integration/fit_scene.py --backend warpower-morton --output results/local/my-bonsai-fit --steps 20000 --initial-sites 8192 --final-sites 262144 --rays 65536 --eval-every 5000 --save-trajectory",
        ],
    )
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    labels = ["Native\n(incremental)", "Warpower\n(input order)", "Warpower\n(Morton order)"]
    bottom = [0.0] * 3
    for stage in ["forward", "backward", "geometry", "optimizer", "other"]:
        values = []
        for run in runs.values():
            measured = run["stages_seconds"]
            values.append(
                run["fit_wall_seconds"]
                - sum(measured[s] for s in ["forward", "backward", "geometry", "optimizer"])
                if stage == "other"
                else measured[stage]
            )
        axes[0].bar(labels, values, bottom=bottom, label=stage)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[0].set_ylabel("Complete fit wall time (seconds)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Same fitting budget; one run per backend")
    for label, run in zip(labels, runs.values()):
        axes[1].plot(
            [item["step"] + 1 for item in run["evaluations"]],
            [item["mean_test_psnr"] for item in run["evaluations"]],
            marker="o",
            label=label.replace("\n", " "),
        )
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Mean held-out PSNR (dB)")
    axes[1].set_title("37 held-out views, full 780 × 520 images")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle("Bonsai scene fitting on NVIDIA L40 — reduced budget")
    fig.savefig(ROOT / "fitting-comparison.png", dpi=180)
    print(json.dumps({k: v for k, v in report.items() if k.endswith("speedup") or k == "fitting"}))


if __name__ == "__main__":
    main()
