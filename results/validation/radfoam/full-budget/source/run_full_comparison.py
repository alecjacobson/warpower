"""Sequential, persistent full-budget Bonsai comparison using the official loop."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpus", default="0-3")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "status.json").exists():
        parser.error("A status file already exists; choose a new output directory")
    snapshot = output / "driver-snapshot"
    snapshot.mkdir()
    for name in [
        "official_reduced.py",
        "integration.py",
        "spatial.py",
        "evaluate_shared.py",
        "run_full_comparison.py",
    ]:
        shutil.copyfile(Path(__file__).with_name(name), snapshot / name)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(repo),
    )
    executable = str(Path(sys.executable).absolute())
    common = [
        "taskset",
        "-c",
        args.cpus,
        executable,
        str(snapshot / "official_reduced.py"),
        "--budget",
        "full",
        "--skip-ply",
        "--data",
        str(repo / "data/radfoam-training"),
        "--build",
        str(repo / ".work/radfoam-fitting-20261110/native"),
    ]
    stages = [
        (
            "preflight",
            common
            + [
                "--backend",
                "warpower-morton",
                "--steps",
                "2",
                "--output",
                str(output / "preflight"),
            ],
        ),
        ("native", common + ["--backend", "native", "--output", str(output / "native")]),
        (
            "warpower",
            common + ["--backend", "warpower-morton", "--output", str(output / "warpower")],
        ),
        (
            "evaluation",
            [
                "taskset",
                "-c",
                args.cpus,
                executable,
                str(snapshot / "evaluate_shared.py"),
                "--downsample",
                "2",
                "--data",
                str(repo / "data/radfoam-training"),
                "--build",
                str(repo / ".work/radfoam-fitting-20261110/native"),
                "--checkpoint",
                "native",
                str(output / "native/output/fit/model.pt"),
                "--checkpoint",
                "warpower",
                str(output / "warpower/output/fit/model.pt"),
                "--output",
                str(output / "shared-evaluation.json"),
            ],
        ),
    ]
    plan = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "Match PSNR while reducing training time at the official full budget.",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.iterdir()
        },
        "dataset_manifest_sha256": {
            name: hashlib.sha256((repo / "data/radfoam-training" / name).read_bytes()).hexdigest()
            for name in ["bonsai-manifest.json", "bonsai-manifest-2.json"]
        },
        "stages": [{"name": name, "command": command} for name, command in stages],
        "workload": {
            "steps": 20000,
            "rays_per_step": 1000000,
            "target_sites": 2097152,
            "training_downsample": [4, 2],
            "downsample_iterations": [0, 5000],
            "evaluation_downsample": 2,
            "evaluation_every": 100,
            "seed": 42,
        },
        "reference": {
            "paper_bonsai_psnr": 32.15,
            "paper_bonsai_training_seconds": 4200,
            "paper_gpu": "RTX 4090",
            "comparison_gpu": "L40",
            "note": "Primary speed comparison is native versus Warp on this L40. Comparing to 70 minutes is cross-hardware context only.",
        },
        "export_scope": "Both runs keep the original renderable PT checkpoint and final images. Redundant PLY exports are skipped to fit persistent storage; exports are outside the loop timer.",
    }
    write(output / "plan.json", plan)
    state = {
        "supervisor_pid": os.getpid(),
        "complete": False,
        "started_utc": plan["started_utc"],
        "completed_stages": [],
    }
    try:
        for name, command in stages:
            if shutil.disk_usage(output).free < 200 * 1024**2:
                raise RuntimeError("Insufficient persistent storage before starting " + name)
            with (output / f"{name}.log").open("w") as log:
                child = subprocess.Popen(
                    command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT
                )
                state.update(
                    stage=name,
                    child_pid=child.pid,
                    stage_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    progress=None,
                )
                write(output / "status.json", state)
                while child.poll() is None:
                    progress = output / name / "progress.json"
                    if progress.exists():
                        try:
                            state["progress"] = json.loads(progress.read_text())
                        except json.JSONDecodeError:
                            pass
                    state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    write(output / "status.json", state)
                    time.sleep(5)
                if child.returncode != 0:
                    raise RuntimeError(
                        f"{name} exited with code {child.returncode}; see {name}.log"
                    )
            if name != "evaluation":
                report = json.loads((output / name / "report.json").read_text())
                if not report["complete"]:
                    raise RuntimeError(name + " did not produce a complete report")
                if name != "preflight":
                    assert report["configuration"]["iterations"] == 20000
                    assert report["configuration"]["final_points"] == 2097152
                    assert report["configuration"]["downsample"] == [4, 2]
                    assert report["rays_per_batch"] == 1000000
            state["completed_stages"].append(name)
            write(output / "status.json", state)
        runs = {
            name: json.loads((output / name / "report.json").read_text())
            for name in ["native", "warpower"]
        }
        metrics = json.loads((output / "shared-evaluation.json").read_text())
        assert (
            runs["native"]["initial_physical_sha256"] == runs["warpower"]["initial_physical_sha256"]
        )
        by_name = {row["name"]: row for row in metrics["models"]}
        comparison = {
            "complete": True,
            "matched_initial_parameters": True,
            "native_over_warpower_loop_speedup": runs["native"]["loop_seconds"]
            / runs["warpower"]["loop_seconds"],
            "warpower_minus_native_official_psnr": by_name["warpower"]["official_psnr"]
            - by_name["native"]["official_psnr"],
            "runs": [
                {
                    "backend": name,
                    "loop_seconds": r["loop_seconds"],
                    "total_seconds": r["total_seconds"],
                    "final_points": r["final_points"],
                    "official_psnr": by_name[name]["official_psnr"],
                    "pooled_rgb_psnr": by_name[name]["pooled_rgb_psnr"],
                    "raw_final_rgba_nonfinite": by_name[name]["nonfinite_rgba"],
                    "final_traversal_cap_hits": by_name[name]["traversal_limit_rays"],
                    "cpu_quota": r["cpu_quota"],
                }
                for name, r in runs.items()
            ],
            "paper_reference": plan["reference"],
            "limitations": [
                "Single full run per backend.",
                "Training uses original upstream gradient masking; numerical counters here cover final evaluation only.",
                "Do not interpret a shorter time than 70 minutes as a hardware-controlled paper speedup.",
            ],
        }
        write(output / "comparison.json", comparison)
        state.update(
            complete=True,
            stage="complete",
            child_pid=None,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
    except BaseException as error:
        state.update(
            stage="failed",
            error=repr(error),
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        raise
    finally:
        write(output / "status.json", state)


if __name__ == "__main__":
    main()
