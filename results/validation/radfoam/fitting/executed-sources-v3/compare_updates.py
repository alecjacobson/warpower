"""Compare native incremental Delaunay with current VTM reuse on identical motion."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import warp as wp
from integration import initialize_runtime, native
from scipy.spatial import Delaunay

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from benchmarks.tetrahedra import canonical
from warpower._wall_reuse import WallReuseDelaunayWorkspace


def time_call(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    function()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames",
        type=Path,
        default=Path("results/local/neighbor-clipping-v2/actual-trials/frames.npy"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    initialize_runtime()
    frames = np.load(args.frames)
    motion = np.max(np.linalg.norm(np.diff(frames, axis=0).astype(np.float64), axis=2), axis=1)
    order = np.argsort(motion, kind="stable")
    indices = sorted(
        set(
            [
                1,
                2,
                len(frames) - 1,
                *[int(order[k]) + 1 for k in [0, len(order) // 2, len(order) - 1]],
            ]
        )
    )
    mesh = WallReuseDelaunayWorkspace(frames[0])
    records = []
    for index in indices:
        a, b = frames[index - 1], frames[index]
        ta, tb = torch.tensor(a, device="cuda"), torch.tensor(b, device="cuda")
        wa, wb = wp.from_torch(ta, dtype=wp.vec3), wp.from_torch(tb, dtype=wp.vec3)
        samples = {
            name: [] for name in ["radfoam-full", "radfoam-incremental", "vtm-fragment-reuse"]
        }
        outputs = {}
        errors = {}
        native_owner = None
        quota_before = cpu_quota_snapshot()
        for repeat in range(args.repeats + 1):
            names = list(samples)
            if repeat % 2:
                names.reverse()
            for name in names:
                if name == "radfoam-full":
                    box = []
                    duration = time_call(lambda: box.append(native.Triangulation(tb)))
                    tri = box[0]
                    perm = tri.permutation().long()
                    outputs[name] = canonical(perm[tri.tets().long()].cpu().numpy())
                    errors[name] = 0
                    native_owner = tri
                elif name == "radfoam-incremental":
                    tri = native.Triangulation(ta)
                    perm = tri.permutation().long().clone()
                    current = tb[perm].contiguous()
                    flag = []
                    duration = time_call(
                        lambda: flag.append(tri.rebuild(current, incremental=True))
                    )
                    assert not flag[0], "unexpected permutation on same-size incremental input"
                    outputs[name] = canonical(perm[tri.tets().long()].cpu().numpy())
                    errors[name] = 0
                    native_owner = tri
                else:
                    mesh.clear_reuse()
                    wp.copy(mesh.points, wa)
                    mesh.run()
                    wp.synchronize()
                    assert mesh.error.numpy()[0] == 0
                    wp.copy(mesh.points, wb)
                    wp.synchronize()
                    duration = time_call(mesh.run)
                    count = int(mesh.tetrahedra_count.numpy()[0])
                    errors[name] = int(mesh.error.numpy()[0])
                    outputs[name] = canonical(mesh.tetrahedra.numpy()[:count])
                if repeat > 0:
                    samples[name].append(duration)
        quota = cpu_quota_delta(quota_before, cpu_quota_snapshot())
        expected = canonical(Delaunay(b.astype(np.float64)).simplices)
        artifacts = args.output.parent / f"vtm-frame-{index}"
        artifacts.mkdir(parents=True, exist_ok=True)
        points_path, tets_path = artifacts / "points.txt", artifacts / "cgal-tets.txt"
        with points_path.open("w") as stream:
            stream.write(f"{len(b)}\n")
            np.savetxt(stream, b, fmt="%.17g")
        reference = subprocess.run(
            [".work/vtm/cgal-delaunay-final", str(points_path), str(tets_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        cgal = canonical(np.loadtxt(tets_path, skiprows=1, dtype=np.int64))
        np.savez_compressed(artifacts / "outputs.npz", points=b, **outputs)
        row = {
            "frame": index,
            "sites": len(b),
            "maximum_site_displacement": float(motion[index - 1]),
            "cpu_quota": quota,
            "cpu_throttled": any(x["throttled_periods"] for x in quota.values()),
            "variants": {
                name: {
                    "seconds": values,
                    "median_seconds": float(np.median(values)),
                    "tetrahedra": len(outputs[name]),
                    "error": errors[name],
                    "matches_scipy": bool(np.array_equal(outputs[name], expected)),
                    "matches_cgal": bool(np.array_equal(outputs[name], cgal)),
                }
                for name, values in samples.items()
            },
            "vtm_reuse_accepted": int(mesh.reuse_accepted.numpy()[0]),
            "vtm_reuse_rounds": int(mesh.reuse_rounds.numpy()[0]),
            "cgal_reference": json.loads(reference.stderr),
        }
        records.append(row)
        report = {
            "scope": "Isolated transitions from an actual 8192-site Armadillo VTM trajectory. One warm-up plus five alternating-order wall-time samples. Previous-state construction, transfers and independent SciPy checking are outside timing. Native outputs tetrahedra AND adjacency; VTM produces an exactly certified tetrahedralization. Different correctness contracts; this is not adoption evidence for replacing VTM.",
            "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "records": records,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cgal_binary_sha256": hashlib.sha256(
                Path(".work/vtm/cgal-delaunay-final").read_bytes()
            ).hexdigest(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    del native_owner


if __name__ == "__main__":
    main()
