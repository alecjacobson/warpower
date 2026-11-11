"""Cache upstream RoMa initialization once for identical paired experiments."""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path("/home/horde/references/vorotracing")
    roma = Path("/home/horde/references/RoMaV2")
    sys.path[:0] = ["/opt/warpower-generated-cache/voro-deps", str(source), str(roma / "src")]
    os.environ.setdefault("TORCH_HOME", "/opt/warpower-generated-cache/torch-models")
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", "/opt/warpower-generated-cache/vorotracing-extensions"
    )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    import vorotracing.datasets.feature_matching as matching
    from vorotracing.datasets.datasets import ColmapConfig

    native = matching.get_pointcloud_from_images
    state = {}

    def recorded(*a, **kw):
        start = time.perf_counter()
        points, colors = native(*a, **kw)
        torch.cuda.synchronize()
        state["matching_seconds"] = time.perf_counter() - start
        return points, colors

    matching.get_pointcloud_from_images = recorded
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    start = time.perf_counter()
    dataset = ColmapConfig(data_path=str(args.data.resolve()), scene="bonsai").load("train", 2)
    points, colors = dataset.get_initial_points()
    assert len(points) >= 1995000, f"Dense initialization too small: {len(points)}"
    assert points.isfinite().all() and colors.isfinite().all()
    checkpoint = args.output / "dense-points.pt"
    torch.save(
        {
            "points": points.cpu(),
            "colors": colors.cpu(),
            "poses": dataset.poses.cpu(),
            "K": dataset.K.cpu(),
        },
        checkpoint,
    )
    report = {
        "complete": True,
        "sites": len(points),
        "seed": 42,
        "scene": "bonsai",
        "split": "train only",
        "downsample": 2,
        "total_seconds": time.perf_counter() - start,
        **state,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "vorotracing_commit": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "roma_commit": subprocess.check_output(
            ["git", "-C", str(roma), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
