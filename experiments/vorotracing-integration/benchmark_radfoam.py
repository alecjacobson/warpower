"""Benchmark the retained RadFoam fit with its native FP16 inference paths."""

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pycolmap
import torch
import warp as wp
from benchmark_renderer import timed
from evaluate_inference import digest
from warp_renderer import prepared_edges


def make_camera(pose, intrinsics, width, height):
    """Express pixel-center pinhole rays using the native benchmark's basis."""
    fx, fy, cx, cy = intrinsics
    right, down, forward = pose[:3, 0], pose[:3, 1], pose[:3, 2]
    return {
        "position": pose[:3, 3].contiguous(),
        "forward": (
            forward + ((0.5 - cx + width / 2) / fx) * right + ((0.5 - cy + height / 2) / fy) * down
        ).contiguous(),
        "right": (right * (fy / fx)).contiguous(),
        "up": (-down).contiguous(),
        "fov": float(2 * math.atan(height / (2 * fy))),
        "width": width,
        "height": height,
        "model": "pinhole",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument("--views", nargs="+", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "radfoam-integration"))
    from official_reduced import load_upstream

    upstream = load_upstream(
        Path("/home/horde/references/radfoam"),
        Path(".work/radfoam-fitting-20261110/native").resolve(),
    )
    wp.init()
    test = upstream.DataHandler(
        SimpleNamespace(
            dataset="colmap", scene="bonsai", data_path=str(args.data.resolve()), patch_based=False
        ),
        rays_per_batch=0,
    )
    test.reload("test", downsample=2)
    reconstruction = pycolmap.Reconstruction(str(args.data / "bonsai/sparse/0"))
    camera = next(iter(reconstruction.cameras.values()))
    width, height = test.img_wh
    camera.rescale(width, height)
    assert camera.model.name == "PINHOLE", "Fast camera adaptation requires pinhole intrinsics"
    model = upstream.RadFoamScene(
        SimpleNamespace(sh_degree=3, init_points=64, final_points=2097152, activation_scale=1.0),
        device=torch.device("cuda"),
        attr_dtype=torch.float16,
    )
    model.load_pt(args.checkpoint)
    with torch.no_grad():
        points, attributes, adjacency, offsets = model.get_trace_data()
        # Native traversal may read a four-edge chunk at the end of a list.
        # Older WarPower checkpoints stored compact adjacency without a tail.
        adjacency = torch.nn.functional.pad(adjacency.view(torch.int32), (0, 32)).view(torch.uint32)
        edges = prepared_edges(points, adjacency, offsets)
        output = torch.empty((height, width), dtype=torch.uint32, device="cuda")
        report = {
            "complete": False,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": digest(args.checkpoint),
            "gpu": torch.cuda.get_device_name(),
            "views": [],
            "generic_scope": "FP16 cached attributes; original full RGBA forward, including nearest-camera query, edge preparation and traversal; input rays precomputed; white background for quality.",
            "fast_scope": "Original native FP16 benchmark kernel with cached edges, including per-frame nearest-camera query, ray generation and packed RGB8 output; native kernel uses black residual background. Camera forward/right vectors adapted algebraically to the dataset's pixel centers, principal point and fx/fy. No renderer source changes.",
        }
        for view in args.views if args.views is not None else range(len(test.rays)):
            rays, target = test.rays[view].cuda(), test.rgbs[view].cuda()
            origin = rays.reshape(-1, 6)[:1, :3].contiguous()

            def generic():
                start = upstream.radfoam.nn(points, model.aabb_tree, origin)
                start = start.expand(height, width).contiguous()
                return model.pipeline.trace_forward(
                    points,
                    attributes,
                    adjacency,
                    offsets,
                    rays,
                    start,
                    weight_threshold=0.001,
                    return_contribution=False,
                )["rgba"]

            rgba = generic().float()
            rgb = (rgba[..., :3] + 1 - rgba[..., 3:]).clamp(0, 1)
            row = {
                "view": view,
                "generic": {
                    **timed(generic, 20),
                    "psnr": float(upstream.psnr(rgb, target).mean()),
                    "pooled_psnr": float(-10 * torch.log10((rgb - target).square().mean())),
                    "nonfinite": int((~torch.isfinite(rgba)).sum()),
                },
                "fast": [],
            }
            fast_camera = make_camera(test.c2ws[view], camera.params, width, height)
            for threshold in [0.001, 0.01, 0.05]:

                def fast():
                    start = upstream.radfoam.nn(points, model.aabb_tree, origin)
                    model.pipeline.trace_benchmark(
                        points,
                        attributes,
                        adjacency,
                        offsets,
                        edges,
                        fast_camera,
                        start,
                        output,
                        weight_threshold=threshold,
                    )

                fast()
                packed = output.view(torch.int32).long()
                rgb = (
                    torch.stack([(packed >> shift) & 255 for shift in [0, 8, 16]], -1).float() / 255
                )
                row["fast"].append(
                    {
                        "threshold": threshold,
                        **timed(fast, 20),
                        "psnr": float(upstream.psnr(rgb, target).mean()),
                        "pooled_psnr": float(-10 * torch.log10((rgb - target).square().mean())),
                    }
                )
            report["views"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        report["complete"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
