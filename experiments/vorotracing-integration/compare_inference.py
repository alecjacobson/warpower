"""Summarize the completed, serial full-checkpoint inference evaluations."""

import argparse
import json
from pathlib import Path


def read(root, stem):
    value = json.loads((root / (stem + ".json")).read_text())
    assert value["complete"], stem
    return value


def aggregate(rows):
    ms = sum(row["gpu_ms"] for row in rows) / len(rows)
    return {
        "views": len(rows),
        "gpu_ms": ms,
        "fps": 1000 / ms,
        "psnr": sum(row["psnr"] for row in rows) / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "complete": False,
        "gpu": "NVIDIA L40",
        "inference_matrix": [],
        "paired_kernel_timings": [],
        "camera_pipeline": [],
        "radfoam": [],
        "high_fidelity": [],
        "limitations": [
            "One fitted model per backend and one scene; no estimate of seed variance.",
            "Compare renderers on the same checkpoint to isolate implementation performance. Different fitted checkpoints also differ in geometry, appearance and memory layout.",
            "All-view inference includes camera lookup, but excludes ray generation and RGB compositing, as in the upstream benchmark. Four-view paired timings alternate backend order; all-view benchmark jobs run serially.",
            "Camera pipeline includes ray generation and RGB8 packing. VoroTracing uses its original white residual background, while RadFoam's native fused kernel uses black. Quality is measured on the actual output in each row.",
            "FPS is the reciprocal of mean frame time, not the mean of per-view FPS. GPU clocks and workload affect timings; small differences are not established across repeated independent sessions.",
        ],
    }
    for checkpoint in ["native", "warp"]:
        for backend, suffix in [("native", "native-inference"), ("warp", "tuned-inference")]:
            stem = f"{checkpoint}-checkpoint-{suffix}"
            report = read(args.results, stem)
            assert len(report["per_view"]) == 37
            assert all(v["nonfinite_rgba"] == 0 for v in report["per_view"])
            assert report["gpu"] == result["gpu"]
            assert backend != "warp" or not report["native_extension_loaded"]
            result["inference_matrix"].append(
                {
                    "checkpoint": checkpoint,
                    "renderer": backend,
                    "evidence": stem + ".json",
                    **report["result"],
                }
            )
        report = read(args.results, f"{checkpoint}-checkpoint-tuned-renderer")
        for index in [0, 1]:
            rows = [view["inference"][index] for view in report["views"]]
            assert all(row["parity"]["max_abs"] == 0 for row in rows)
            times = {
                backend: sum(row[backend]["gpu_ms"] for row in rows) / len(rows)
                for backend in ["native", "warp"]
            }
            result["paired_kernel_timings"].append(
                {
                    "checkpoint": checkpoint,
                    "views": [v["view"] for v in report["views"]],
                    "threshold": rows[0]["transmittance_threshold"],
                    "skip": rows[0]["cell_skip_threshold"],
                    "bit_exact": True,
                    "gpu_ms": times,
                    "native_over_warp_speedup": times["native"] / times["warp"],
                }
            )
        report = read(args.results, f"{checkpoint}-camera-pipeline-white")
        assert len(report["views"]) == 37
        assert all(v["raw_rgba_nonfinite"] == 0 for v in report["views"])
        result["camera_pipeline"].append(
            {
                "checkpoint": checkpoint,
                "renderer": checkpoint,
                "background": report["background"],
                **aggregate(report["views"]),
            }
        )
        report = read(args.results, f"{checkpoint}-high-fidelity-inference")
        assert all(v["nonfinite_rgba"] == 0 for v in report["per_view"])
        result["high_fidelity"].append(
            {"checkpoint": checkpoint, "renderer": checkpoint, **report["result"]}
        )
        report = read(args.results, f"radfoam-{checkpoint}-inference")
        assert len(report["views"]) == 37
        assert all(v["generic"]["nonfinite"] == 0 for v in report["views"])
        result["radfoam"].append(
            {
                "checkpoint": checkpoint,
                "renderer": "native",
                "generic": aggregate([v["generic"] for v in report["views"]]),
                "fast_camera_pipeline": [
                    {
                        "threshold": report["views"][0]["fast"][i]["threshold"],
                        **aggregate([v["fast"][i] for v in report["views"]]),
                    }
                    for i in range(3)
                ],
            }
        )
    result["complete"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
