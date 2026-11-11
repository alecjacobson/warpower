"""Re-evaluate a saved checkpoint with the FP32 training renderer and raw audits."""

import argparse
import json
import sys
from pathlib import Path

import numerics
import torch
from evaluate_inference import digest
from upstream import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "warp"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument("--views", type=int, nargs="+")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    upstream = load(args.backend, audit=True)
    import benchmark_eval
    from vorotracing.datasets.datasets import ColmapConfig

    config, _ = benchmark_eval.load_config(args.checkpoint)
    model = upstream.VoroTracing.from_pretrained(args.checkpoint, config)
    dataset = ColmapConfig(
        data_path=str(args.data.resolve()), scene="bonsai", init_points="original"
    ).load("test", 2)
    report = {
        "complete": False,
        "backend": args.backend,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "gpu": torch.cuda.get_device_name(),
        "native_extension_loaded": "vorotracing.cuda._backend" in sys.modules,
        "protocol": "Saved FP32 checkpoint, half-resolution held-out images, original training renderer with transmittance threshold 0.001 and cap 1024; white background. Capped rays count includes natural termination exactly at the cap.",
        "views": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        starts = model.get_starting_point(dataset.poses[:, :, 3].cuda())
        for view in args.views if args.views is not None else range(len(dataset)):
            batch = dataset[view]
            rays, target = batch["rays"].cuda(), batch["rgbs"].cuda()
            numerics.reset()
            rgba, _, _, counts, _, _ = model(rays, starts[view])
            rgb = benchmark_eval.output_to_rgb(rgba)
            row = {
                "view": view,
                "image_name": dataset.image_names[view],
                "psnr": float(benchmark_eval.psnr(rgb, target).mean()),
                "pooled_psnr": float(-10 * torch.log10((rgb - target).square().mean())),
                "max_intersections": int(counts.to(torch.int64).max()),
                "numerics": numerics.snapshot(),
            }
            report["views"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if args.preview_dir is not None and view in [0, 10, 36]:
                from PIL import Image

                args.preview_dir.mkdir(parents=True, exist_ok=True)
                for name, pixels in [("render", rgb), ("target", target)]:
                    Image.fromarray((pixels.clamp(0, 1) * 255).round().byte().cpu().numpy()).save(
                        args.preview_dir / f"{view:02d}-{name}.png"
                    )
    report.update(
        complete=True,
        psnr=sum(v["psnr"] for v in report["views"]) / len(report["views"]),
        pooled_psnr=sum(v["pooled_psnr"] for v in report["views"]) / len(report["views"]),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
