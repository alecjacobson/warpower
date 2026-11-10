"""Fresh-process Radiant Foam adjacency timings and full-hull graph comparisons.

GPU outputs are clipped Voronoi adjacency; CGAL outputs full Delaunay adjacency.
Graph differences are observations, not automatically geometric errors.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.compare import edges, unpack
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2) + "\n")


def cpu_run(binary, points, folder, warmups, repeats, threads, env):
    name = f"cgal-{threads}thread"
    raw = folder / f"{name}.bin"
    command = [str(binary), str(points), str(raw), str(warmups), str(repeats), str(threads)]
    samples, final = [], None
    before = cpu_quota_snapshot()
    started = time.monotonic()
    with (folder / f"{name}.log").open("w") as log, (folder / f"{name}.stderr").open("w") as err:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err, text=True, env=env)
        for line in process.stdout:
            log.write(line)
            log.flush()
            row = json.loads(line)
            after = cpu_quota_snapshot()
            row["cpu_quota_since_previous_sample"] = cpu_quota_delta(before, after)
            row["cpu_throttled"] = any(
                x["throttled_periods"] for x in row["cpu_quota_since_previous_sample"].values()
            )
            before = after
            if row.get("final"):
                final = row
            else:
                samples.append(row)
                print(json.dumps({"scene": folder.name, "backend": name, **row}), flush=True)
            write_json(
                folder / f"{name}-partial.json",
                {"command": command, "samples": samples, "final": final},
            )
        code = process.wait()
    if code or final is None:
        raise RuntimeError(f"{name} failed ({code}); see {folder}")
    with raw.open("rb") as stream:
        n, nnz = map(int, np.fromfile(stream, dtype="<u8", count=2))
        offsets = np.fromfile(stream, dtype="<i4", count=n + 1)
        adjacency = np.fromfile(stream, dtype="<i4", count=nnz)
        if stream.read(1) or len(adjacency) != nnz or offsets[-1] != nnz:
            raise ValueError("invalid native CSR serialization")
    saved = folder / f"{name}-graph.npz"
    np.savez_compressed(saved, adjacency=adjacency, offsets=offsets, status=np.zeros(n, np.int32))
    with np.load(saved) as verified:
        assert np.array_equal(verified["adjacency"], adjacency)
        assert np.array_equal(verified["offsets"], offsets)
    raw_hash = sha(raw)
    raw.unlink()  # Verified lossless NPZ retains the complete native output.
    measured = [r for r in samples if r["iteration"] >= 0]
    record = {
        "backend": name,
        "command": command,
        "binary_sha256": sha(binary),
        "input_f32_sha256": sha(points),
        "warmups": warmups,
        "repeats": repeats,
        "samples": samples,
        "final": final,
        "mean_seconds": float(np.mean([r["total_seconds"] for r in measured])),
        "median_seconds": float(np.median([r["total_seconds"] for r in measured])),
        "mean_build_seconds": float(np.mean([r["build_seconds"] for r in measured])),
        "cpu_throttled": any(r["cpu_throttled"] for r in measured),
        "process_wall_seconds": time.monotonic() - started,
        "serialized_csr_sha256": raw_hash,
        "graph_npz_sha256": sha(saved),
        "scope": "Native CPU exact-predicate full-hull Delaunay plus two-pass CSR export; allocations included. Input decoding, final is_valid and disk export excluded. Sequential CSR export also used with parallel Delaunay.",
    }
    write_json(folder / f"{name}.json", record)
    return record


def compare_graphs(folder, names):
    packed, statuses, duplicate_counts = {}, {}, {}
    for name in names:
        with np.load(folder / f"{name}-graph.npz") as data:
            packed[name], duplicate_counts[name] = edges(data)
            statuses[name] = data["status"].copy()
    result = {
        "scope": "Full directed adjacency sets. GPU cells are box-clipped; CGAL is full-hull and can include zero-area dual faces at exact ties. Differences alone do not establish errors.",
        "pairs": {},
    }
    for other in names[1:]:
        a, b = packed["warpower"], packed[other]
        only_a = np.setdiff1d(a, b, assume_unique=True)
        only_b = np.setdiff1d(b, a, assume_unique=True)
        row = {
            "warpower_edges": len(a),
            "other_edges": len(b),
            "shared_edges": len(a) - len(only_a),
            "only_warpower": len(only_a),
            "only_other": len(only_b),
            "warpower_duplicate_edges": duplicate_counts["warpower"],
            "other_duplicate_edges": duplicate_counts[other],
            "only_warpower_examples": unpack(only_a[:16]),
            "only_other_examples": unpack(only_b[:16]),
            "affected_rows": int(len(np.union1d(only_a >> np.uint64(32), only_b >> np.uint64(32)))),
            "warpower_failed_cells": int(np.count_nonzero(~np.isin(statuses["warpower"], [0, 3]))),
            "other_failed_cells": int(np.count_nonzero(~np.isin(statuses[other], [0, 3]))),
        }
        result["pairs"][other] = row
        # Retain reproducible diagnostic edge samples for an offline exact facet audit.
        rng = np.random.default_rng(2026)
        categories = {
            "only_warpower": only_a,
            "only_other": only_b,
            "shared": np.intersect1d(a, b, assume_unique=True),
        }
        samples = {
            key: unpack(
                values[np.sort(rng.choice(len(values), min(64, len(values)), replace=False))]
            )
            for key, values in categories.items()
        }
        write_json(folder / f"samples-vs-{other}.json", samples)
    write_json(folder / "graph-comparison.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cgal", type=Path, required=True)
    parser.add_argument("--cgal-parallel", type=Path)
    parser.add_argument("--scenes", nargs="+", default=["bonsai", "garden", "bicycle", "playroom"])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        CUDA_HOME="/usr/local/cuda-12.6",
        TORCH_CUDA_ARCH_LIST="8.9",
        MAX_JOBS="2",
    )
    report = {
        "scope": "Unweighted construction on full public RadFoam v1 site arrays. Separate sequential backend processes; no rendering or training speed claim.",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": sha(__file__),
        "cgal_source_sha256": sha("benchmarks/cpu/radfoam_adjacency.cpp"),
        "platform": subprocess.check_output(["lscpu"], text=True),
        "cpu_limits": cpu_quota_snapshot(),
        "scenes": {},
    }
    write_json(args.output / "report.json", report)
    for scene in args.scenes:
        folder = args.output / scene
        folder.mkdir()
        source = Path("data/radfoam") / f"{scene}.npz"
        provenance = json.loads(source.with_suffix(".json").read_text())
        points = np.load(source)["points"]
        assert points.dtype == np.float32 and points.shape == (provenance["n"], 3)
        assert sha(source) == provenance["npz_sha256"]
        assert hashlib.sha256(points.tobytes()).hexdigest() == provenance["positions_sha256"]
        assert np.isfinite(points).all()
        input_bin = folder / "input.f32"
        points.tofile(input_bin)
        del points
        scene_report = {"input": provenance, "backends": {}}
        report["scenes"][scene] = scene_report
        for name in ["warpower", "paragram"]:
            command = [
                sys.executable,
                "-m",
                "benchmarks.run",
                "--backend",
                name,
                "--input",
                str(source),
                "--distribution",
                scene,
                "--warmup",
                str(args.warmups),
                "--repeats",
                str(args.repeats),
                "--output",
                str(folder / f"{name}.json"),
                "--save-diagram",
                str(folder / f"{name}-graph.npz"),
            ]
            if name == "warpower":
                command += ["--mempool-retain-mb", "32768"]
            print(json.dumps({"scene": scene, "starting": name, "command": command}), flush=True)
            with (folder / f"{name}.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            record = json.loads((folder / f"{name}.json").read_text())
            record["command"] = command
            record["graph_npz_sha256"] = sha(folder / f"{name}-graph.npz")
            write_json(folder / f"{name}.json", record)
            scene_report["backends"][name] = record
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "finished": name,
                        "mean_seconds": record["mean_seconds"],
                        "cpu_throttled": record["cpu_throttled"],
                        "statuses": record["status_counts"],
                    }
                ),
                flush=True,
            )
            write_json(args.output / "report.json", report)
        names = ["warpower", "paragram", "cgal-1thread"]
        scene_report["backends"]["cgal-1thread"] = cpu_run(
            args.cgal.resolve(), input_bin, folder, args.warmups, args.repeats, 1, env
        )
        if args.cgal_parallel:
            names.append("cgal-4thread")
            scene_report["backends"]["cgal-4thread"] = cpu_run(
                args.cgal_parallel.resolve(), input_bin, folder, args.warmups, args.repeats, 4, env
            )
        scene_report["comparison"] = compare_graphs(folder, names)
        input_bin.unlink()  # Reproducible byte-for-byte from the preserved original NPZ.
        write_json(args.output / "report.json", report)
        print(json.dumps({"scene": scene, "complete": True}), flush=True)
    report["complete"] = True
    write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
