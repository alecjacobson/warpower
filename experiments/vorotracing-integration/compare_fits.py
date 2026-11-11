"""Summarize completed paired VoroTracing fits and the retained RadFoam baseline."""

import argparse
import json
import statistics
from pathlib import Path

from evaluate_inference import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--warp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--radfoam",
        type=Path,
        default=Path("results/validation/radfoam/full-budget/comparison.json"),
    )
    args = parser.parse_args()
    reports = {
        name: json.loads((directory / "report.json").read_text())
        for name, directory in [("native", args.native), ("warp", args.warp)]
    }
    native, warp = reports["native"], reports["warp"]
    assert all(r.get("complete") for r in reports.values()), "Both full fits must finish"
    for key in [
        "initial_base_sha256",
        "warmup_batch_sha256",
        "source_commit",
        "gpu",
        "torch",
        "cpu_affinity",
    ]:
        assert native[key] == warp[key], f"Unmatched {key}"
    for key in ["sites", "rays", "steps", "downsample", "eval_every"]:
        assert native["options"][key] == warp["options"][key], f"Unmatched workload: {key}"
    rows = []
    for name, report in reports.items():
        directory = args.native if name == "native" else args.warp
        process_path = directory / "process.json"
        process = json.loads(process_path.read_text()) if process_path.exists() else None
        last = report["evaluations"][-1]
        assert last["step"] + 1 == report["options"]["steps"]
        assert len(last["per_view"]) == 37
        rows.append(
            {
                "backend": name,
                "loop_seconds": report["loop_seconds"],
                "reported_trainer_wall_seconds": report["total_seconds"],
                "process": process,
                "sites": report["sites"],
                **last["metrics"],
                "pooled_psnr": sum(v["pooled_psnr"] for v in last["per_view"]) / 37,
                "geometry_update_seconds": sum(v["seconds"] for v in report["geometry"]),
                "geometry_updates": len(report["geometry"]),
                "numerics": report["numerics"],
                "cpu_quota": report["cpu_quota"],
                "native_extension_loaded": report["native_extension_loaded"],
                "checkpoint_sha256": digest(report["checkpoint"]),
            }
        )
    native_last, warp_last = [r["evaluations"][-1] for r in [native, warp]]
    per_view = [
        {
            "view": a["view"],
            "native_psnr": a["psnr"],
            "warp_psnr": b["psnr"],
            "warp_minus_native_db": b["psnr"] - a["psnr"],
        }
        for a, b in zip(native_last["per_view"], warp_last["per_view"])
    ]
    baselines = json.loads(args.radfoam.read_text())
    targets = sorted(
        {r["official_psnr"] for r in baselines["runs"]} | {native_last["metrics"]["psnr"]}
    )
    crossings = []
    for target in targets:
        row = {"target_psnr": target}
        for name, report in reports.items():
            crossed = next(
                (e for e in report["evaluations"] if e["metrics"]["psnr"] >= target), None
            )
            row[name] = (
                {
                    "step": crossed["step"] + 1,
                    "seconds": crossed["seconds"],
                    "psnr": crossed["metrics"]["psnr"],
                }
                if crossed
                else None
            )
        crossings.append(row)
    output = {
        "complete": True,
        "paired_initialization_and_warmup": True,
        "fits": rows,
        "radfoam_baselines": baselines["runs"],
        "native_over_warp_loop_speedup": native["loop_seconds"] / warp["loop_seconds"],
        "warp_minus_native_final_psnr_db": warp_last["metrics"]["psnr"]
        - native_last["metrics"]["psnr"],
        "per_view": per_view,
        "per_view_delta_summary": {
            "mean_db": statistics.mean(v["warp_minus_native_db"] for v in per_view),
            "median_db": statistics.median(v["warp_minus_native_db"] for v in per_view),
            "warp_better_views": sum(v["warp_minus_native_db"] > 0 for v in per_view),
            "native_better_views": sum(v["warp_minus_native_db"] < 0 for v in per_view),
            "largest_warp_deficit": min(per_view, key=lambda v: v["warp_minus_native_db"]),
            "largest_warp_gain": max(per_view, key=lambda v: v["warp_minus_native_db"]),
        },
        "time_to_quality": crossings,
        "limitations": [
            "One full fit per backend; no estimate of seed variance.",
            "Threshold crossings are first recorded observations, with no interpolation or intermediate checkpoint guarantee.",
            "VoroTracing and RadFoam use different published recipes and validation schedules; only native-vs-Warp within VoroTracing is the matched implementation comparison.",
            "Numerical masking is counted for both VoroTracing runs. The older full RadFoam runs did not record raw training-gradient nonfinites.",
            "Successful fitting does not certify all WarPower adjacency edges.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: output[k]
                for k in ["native_over_warp_loop_speedup", "warp_minus_native_final_psnr_db"]
            }
        )
    )


if __name__ == "__main__":
    main()
