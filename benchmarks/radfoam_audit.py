"""Sample exact bounded Voronoi facets using a validated CGAL neighbor star."""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np

from benchmarks.compare import edges, unpack
from benchmarks.radfoam_compare import sha, write_json


def write_query(stream, owner, candidate, lower, upper, neighbors):
    for point in [owner, candidate, lower, upper]:
        stream.write(" ".join(format(float(x), ".17g") for x in point) + "\n")
    stream.write(str(len(neighbors)) + "\n")
    for point in neighbors:
        stream.write(" ".join(format(float(x), ".17g") for x in point) + "\n")


def controls(binary, root):
    cube = np.array(list(itertools.product([-1, 1], repeat=3)), dtype=np.float32)
    cases = [
        ([0, 0, 0], [2, 0, 0], [[2, 0, 0]], True),
        ([0, 0, 0], [4, 0, 0], [[2, 0, 0], [4, 0, 0]], False),
        ([0, 0, 0], [30, 0, 0], [[30, 0, 0]], False),
        (cube[0], cube[1], cube[1:], True),
        (cube[0], cube[3], cube[1:], False),
        (cube[0], cube[7], cube[1:], False),
    ]
    query = root / "facet-controls.txt"
    with query.open("w") as stream:
        stream.write(str(len(cases)) + "\n")
        for owner, candidate, neighbors, expected in cases:
            write_query(stream, owner, candidate, [-10] * 3, [10] * 3, neighbors)
    result = subprocess.run([str(binary), str(query)], capture_output=True, text=True, check=True)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == len(cases)
    assert [r["positive_area_face"] for r in rows] == [c[-1] for c in cases]
    write_json(root / "facet-controls.json", {"passed": len(cases), "results": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--warpower-disagreement-limit", type=int, default=20000)
    args = parser.parse_args()
    binary = args.binary.resolve()
    controls(binary, args.root)
    for folder in sorted(args.root.iterdir()):
        if not folder.is_dir() or not (folder / "cgal-1thread.json").exists():
            continue
        cg = json.loads((folder / "cgal-1thread.json").read_text())
        assert cg["final"]["valid"]
        points = np.load(Path("data/radfoam") / f"{folder.name}.npz")["points"]
        provenance = json.loads((Path("data/radfoam") / f"{folder.name}.json").read_text())
        assert hashlib.sha256(points.tobytes()).hexdigest() == provenance["positions_sha256"]
        with np.load(folder / "cgal-1thread-graph.npz") as reference:
            cgal_edges, duplicates = edges(reference)
            assert duplicates == 0
            adjacency, offsets = reference["adjacency"], reference["offsets"]
        for backend in ["warpower", "paragram"]:
            with np.load(folder / f"{backend}-graph.npz") as result:
                gpu_edges, duplicates = edges(result)
                assert duplicates == 0
                gpu_status = result["status"]
            categories = {
                "gpu_only": np.setdiff1d(gpu_edges, cgal_edges, assume_unique=True),
                "cgal_only": np.setdiff1d(cgal_edges, gpu_edges, assume_unique=True),
                "shared": np.intersect1d(gpu_edges, cgal_edges, assume_unique=True),
            }
            rng = np.random.default_rng(2026)
            samples, population = [], {}
            for category, values in categories.items():
                population[category] = len(values)
                limit = (
                    args.warpower_disagreement_limit
                    if backend == "warpower" and category != "shared"
                    else args.sample_count
                )
                selected = values[
                    np.sort(rng.choice(len(values), min(limit, len(values)), replace=False))
                ]
                samples += [
                    {
                        "category": category,
                        "owner": i,
                        "candidate": j,
                        "owner_status": int(gpu_status[i]),
                        "candidate_status": int(gpu_status[j]),
                    }
                    for i, j in unpack(selected)
                ]
            del categories, gpu_edges
            # Match each GPU implementation's actual float32 domain arithmetic.
            padding = np.float32(1.00001 if backend == "warpower" else 1.0)
            lower = points.min(axis=0) - padding
            upper = points.max(axis=0) + padding
            query = folder / f"facet-{backend}-queries.txt"
            with query.open("w") as stream:
                stream.write(str(len(samples)) + "\n")
                for sample in samples:
                    i, j = sample["owner"], sample["candidate"]
                    neighbors = adjacency[offsets[i] : offsets[i + 1]]
                    write_query(stream, points[i], points[j], lower, upper, points[neighbors])
            completed = subprocess.run([str(binary), str(query)], capture_output=True, text=True)
            (folder / f"facet-{backend}.log").write_text(completed.stdout + completed.stderr)
            completed.check_returncode()
            answers = [json.loads(line) for line in completed.stdout.splitlines()]
            assert len(answers) == len(samples)
            for sample, answer in zip(samples, answers):
                sample.update(answer)
            summary = {}
            for category in population:
                rows = [s for s in samples if s["category"] == category]
                summary[category] = {
                    "population_directed_edges": population[category],
                    "sampled": len(rows),
                    "positive_area_face": sum(r["positive_area_face"] for r in rows),
                    "no_positive_area_face": sum(not r["positive_area_face"] for r in rows),
                    "owner_non_success": sum(r["owner_status"] != 0 for r in rows),
                    "candidate_non_success": sum(r["candidate_status"] != 0 for r in rows),
                }
            report = {
                "scope": "All Warpower disagreements when their category has at most 20000 directed edges; otherwise uniform fixed-seed samples. Paragram disagreements and shared edges are sampled. Not a complete graph audit. Exact rational LP uses all CGAL neighbors of the owner; native CGAL full-hull triangulation has passed is_valid. Strict positive slack on the candidate bisector proves a positive-area facet inside the backend's float32 clipping box. Zero-area contacts are not adjacency.",
                "backend": backend,
                "seed": 2026,
                "summary": summary,
                "samples": samples,
                "box": [lower.tolist(), upper.tolist()],
                "binary_sha256": sha(binary),
                "source_sha256": sha("benchmarks/cpu/voronoi_facet_audit.cpp"),
                "queries_sha256": sha(query),
                "input_points_sha256": provenance["positions_sha256"],
            }
            write_json(folder / f"facet-{backend}.json", report)
            print(
                json.dumps({"scene": folder.name, "backend": backend, "summary": summary}),
                flush=True,
            )


if __name__ == "__main__":
    main()
