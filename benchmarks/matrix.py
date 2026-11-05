"""Run/resume the paper's synthetic sizes and nine public scenes, sequentially."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.datasets import generate, generate_weights
from benchmarks.fetch_scenes import SCENES


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def source_digest():
    return hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(Path("warpower").glob("*.py")))
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", choices=["paragram", "warpower"], default=["paragram", "warpower"]
    )
    parser.add_argument(
        "--sizes",
        nargs="*",
        type=int,
        default=[100000, 500000, 1000000, 2000000, 5000000, 10000000, 15000000],
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "gradient", "cluster5", "cluster10"],
        default=["uniform", "gradient", "cluster5", "cluster10"],
    )
    parser.add_argument("--scenes", nargs="*", choices=SCENES, default=list(SCENES))
    parser.add_argument(
        "--modes", nargs="+", choices=["voronoi", "power"], default=["voronoi", "power"]
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--mempool-retain-mb", type=int, default=32768)
    parser.add_argument("--job-timeout", type=int, default=900)
    args = parser.parse_args()
    source = source_digest()
    benchmark_source = hashlib.sha256(
        b"".join(
            (Path("benchmarks") / name).read_bytes()
            for name in ("datasets.py", "memory.py", "run.py")
        )
    ).hexdigest()
    cases = [(d, n, None) for n in args.sizes for d in args.distributions]
    cases += [(scene, None, Path("data/radfoam") / f"{scene}.npz") for scene in args.scenes]
    cache_dir = Path("data/matrix")
    cache_dir.mkdir(parents=True, exist_ok=True)
    logs = Path(".work/matrix")
    logs.mkdir(parents=True, exist_ok=True)
    progress = logs / "progress.jsonl"
    for distribution, n, scene in cases:
        if source_digest() != source:
            raise RuntimeError("Warpower source changed during the matrix; resume after committing")
        print(f"Preparing {distribution} {n or 'scene'}", flush=True)
        if scene is None:
            points, weights = generate(n, distribution, args.seed, weighted="power" in args.modes)
        else:
            points = np.load(scene)["points"]
            weights = generate_weights(points, args.seed) if "power" in args.modes else None
            n = len(points)
        points_hash = digest(points)
        weights_hash = digest(weights) if weights is not None else None
        dataset = cache_dir / "current.npz"
        np.savez(dataset, points=points, **({"weights": weights} if weights is not None else {}))
        del points, weights
        for mode in args.modes:
            for backend in args.backends:
                if source_digest() != source:
                    raise RuntimeError(
                        "Warpower source changed during the matrix; resume after committing"
                    )
                name = f"{distribution}-{n}-{mode}"
                output = Path(f"results/{backend}-l40/matrix/{name}.json")
                pool = args.mempool_retain_mb if backend == "warpower" else None
                expected = dict(
                    points_sha256=points_hash,
                    weights_sha256=weights_hash if mode == "power" else None,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    mempool_retain_mb=pool,
                )
                if output.exists():
                    old = json.loads(output.read_text())
                    matches = all(old.get(k) == v for k, v in expected.items())
                    matches &= not old.get("cpu_throttled", True)
                    matches &= (
                        old.get("versions", {}).get("benchmark_source_sha256") == benchmark_source
                    )
                    if backend == "warpower":
                        matches &= old.get("versions", {}).get("warpower_source_sha256") == source
                    if matches:
                        print(f"Skip {backend} {name}", flush=True)
                        continue
                    archive = output.parent / "archive"
                    archive.mkdir(exist_ok=True)
                    output.rename(archive / f"{name}-{time.time_ns()}.json")
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.run",
                    "--backend",
                    backend,
                    "--distribution",
                    distribution,
                    "--input",
                    str(dataset),
                    "--seed",
                    str(args.seed),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--mempool-retain-mb",
                    str(args.mempool_retain_mb),
                    "--output",
                    str(output),
                ]
                if mode == "power":
                    command.append("--weighted")
                env = os.environ.copy()
                env.setdefault("CUDA_HOME", "/usr/local/cuda-12.6")
                env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
                env.setdefault("MAX_JOBS", "4")
                started = time.perf_counter()
                log = logs / f"{backend}-{name}.log"
                try:
                    with log.open("w") as stream:
                        process = subprocess.run(
                            command,
                            env=env,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            timeout=args.job_timeout,
                        )
                    result = {"backend": backend, "case": name, "exit_code": process.returncode}
                    if process.returncode == 0:
                        measured = json.loads(output.read_text())
                        result.update(
                            seconds=measured["mean_seconds"],
                            status_counts=measured["status_counts"],
                            memory=measured["memory"],
                            cpu_throttled=measured["cpu_throttled"],
                        )
                except subprocess.TimeoutExpired:
                    result = {"backend": backend, "case": name, "timeout": args.job_timeout}
                result["wall_seconds"] = time.perf_counter() - started
                with progress.open("a") as stream:
                    stream.write(json.dumps(result) + "\n")
                print(json.dumps(result), flush=True)
        # This is a reproducible temporary input; provenance remains in results.
        dataset.unlink()


if __name__ == "__main__":
    main()
