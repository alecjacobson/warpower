"""Time camera rays, nearest-cell lookup, FP16 tracing and packed RGB8 together."""

import argparse
import json
import sys
from pathlib import Path

import torch
import warp as wp
from benchmark_renderer import timed
from evaluate_inference import digest
from upstream import load
from warp_renderer import stream, tensor, vec4h

wp.set_module_options({"enable_backward": False, "fast_math": True})
OPAQUE = wp.constant(wp.uint32(0xFF000000))


@wp.kernel
def camera_rays(
    origin: wp.vec3,
    right: wp.vec3,
    down: wp.vec3,
    forward: wp.vec3,
    intrinsics: wp.vec4,
    width: int,
    rays: wp.array2d[wp.vec3],
):
    i = wp.tid()
    x, y = float(i % width) + 0.5, float(i // width) + 0.5
    rays[i, 0] = origin
    rays[i, 1] = (
        ((x - intrinsics[2]) / intrinsics[0]) * right
        + ((y - intrinsics[3]) / intrinsics[1]) * down
        + forward
    )


@wp.kernel
def pack_rgb(rgba: wp.array[vec4h], background: float, packed: wp.array[wp.uint32]):
    i = wp.tid()
    p = rgba[i]
    residual = (1.0 - float(p[3])) * background
    r = wp.uint32(wp.clamp(float(p[0]) + residual, 0.0, 1.0) * 255.0)
    g = wp.uint32(wp.clamp(float(p[1]) + residual, 0.0, 1.0) * 255.0)
    b = wp.uint32(wp.clamp(float(p[2]) + residual, 0.0, 1.0) * 255.0)
    packed[i] = r | (g << wp.uint32(8)) | (b << wp.uint32(16)) | OPAQUE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "warp"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument("--views", nargs="+", type=int)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--skip", type=float, default=0.001)
    parser.add_argument("--background", choices=["white", "black"], default="white")
    args = parser.parse_args()
    upstream = load(args.backend)
    import benchmark_eval
    from vorotracing.datasets.datasets import ColmapConfig

    config, _ = benchmark_eval.load_config(args.checkpoint)
    model = upstream.VoroTracingInfer.from_pretrained(args.checkpoint, config)
    dataset = ColmapConfig(
        data_path=str(args.data.resolve()), scene="bonsai", init_points="original"
    ).load("test", 2)
    assert dataset.camera.model.name == "PINHOLE"
    width, height = dataset.img_wh
    rays = torch.empty((height, width, 6), device="cuda", dtype=torch.float32)
    packed = torch.empty((height, width), device="cuda", dtype=torch.uint32)
    intrinsics = wp.vec4(*map(float, dataset.camera.params))
    ray_view = tensor(rays.reshape(-1, 2, 3), wp.vec3)
    packed_view = tensor(packed.reshape(-1))
    model._tile_perm(height, width)
    report = {
        "complete": False,
        "backend": args.backend,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "gpu": torch.cuda.get_device_name(),
        "native_extension_loaded": "vorotracing.cuda._backend" in sys.modules,
        "scope": "Per frame: generate pinhole pixel-center rays, query nearest camera cell, FP16 tracing, composite background, clamp and truncate RGB8. Cached model/edges and output ray storage; no display or host image transfer. RadFoam's native benchmark fuses ray generation/tracing/packing and uses black residual background. VoroTracing's original evaluation uses white.",
        "background": args.background,
        "threshold": args.threshold,
        "skip": args.skip,
        "views": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for view in args.views if args.views is not None else range(len(dataset)):
            pose = dataset.poses[view]
            basis = [wp.vec3(*pose[:, j].tolist()) for j in [3, 0, 1, 2]]

            def generate():
                wp.launch(
                    camera_rays,
                    height * width,
                    inputs=[*basis, intrinsics, width, ray_view],
                    stream=stream(),
                )

            def render():
                generate()
                rgba = model.render(
                    rays, weight_threshold=args.threshold, cell_skip_threshold=args.skip
                )
                wp.launch(
                    pack_rgb,
                    height * width,
                    inputs=[
                        tensor(rgba.reshape(-1, 4), vec4h),
                        float(args.background == "white"),
                        packed_view,
                    ],
                    stream=stream(),
                )
                return rgba

            batch = dataset[view]
            target = batch["rgbs"].cuda()
            expected = batch["rays"].cuda()
            generate()
            direction_error = float(
                (
                    torch.nn.functional.normalize(rays[..., 3:], dim=-1)
                    - torch.nn.functional.normalize(expected[..., 3:], dim=-1)
                )
                .abs()
                .max()
            )
            origin_error = float((rays[..., :3] - expected[..., :3]).abs().max())
            assert direction_error < 2e-6 and origin_error < 1e-6
            raw = render()
            nonfinite = int((~torch.isfinite(raw)).sum())
            assert nonfinite == 0
            values = packed.view(torch.int32).long()
            rgb = torch.stack([(values >> shift) & 255 for shift in [0, 8, 16]], -1).float() / 255
            reference = raw.float()
            reference_rgb = reference[..., :3]
            if args.background == "white":
                reference_rgb = reference_rgb + 1 - reference[..., 3:]
            reference_rgb = (reference_rgb.clamp(0, 1) * 255).to(torch.uint8).float() / 255
            packing_error = float((rgb - reference_rgb).abs().max())
            assert packing_error <= 1 / 255 + 1e-7
            row = {
                "view": view,
                "image_name": dataset.image_names[view],
                "ray_direction_max_abs": direction_error,
                "ray_origin_max_abs": origin_error,
                "raw_rgba_nonfinite": nonfinite,
                "packing_max_abs": packing_error,
                "psnr": float(benchmark_eval.psnr(rgb, target).mean()),
                "pooled_psnr": float(-10 * torch.log10((rgb - target).square().mean())),
                **timed(render, args.repeats),
            }
            report["views"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    ms = sum(v["gpu_ms"] for v in report["views"]) / len(report["views"])
    report.update(
        complete=True,
        gpu_ms=ms,
        fps=1000 / ms,
        psnr=sum(v["psnr"] for v in report["views"]) / len(report["views"]),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
