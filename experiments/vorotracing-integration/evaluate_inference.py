"""Run the original all-view VoroTracing inference benchmark on either backend."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from upstream import load


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["native", "warp"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--skip", type=float, default=0.001)
    args = parser.parse_args()
    load(args.backend)
    import benchmark_eval as upstream
    from vorotracing.datasets.datasets import ColmapConfig

    dataset = ColmapConfig(
        data_path=str(args.data.resolve()), scene="bonsai", init_points="original"
    ).load("test", 2)
    config = upstream.BenchmarkConfig(
        scene="bonsai", downsample=2, weight_threshold=args.threshold, cell_skip_threshold=args.skip
    )
    device = torch.device("cuda")
    ssim = upstream.SSIM(data_range=1.0, size_average=True, channel=3)
    lpips = upstream.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(
        device
    )
    lpips_3dgs = upstream.LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(
        device
    )
    per_view = []
    original_rgb = upstream.output_to_rgb
    original_psnr = upstream.psnr

    def audited_rgb(rgba, white):
        view = len(per_view)
        per_view.append(
            {
                "view": view,
                "image_name": dataset.image_names[view],
                "nonfinite_rgba": int((~torch.isfinite(rgba)).sum()),
            }
        )
        return original_rgb(rgba, white)

    def recorded_psnr(rgb, target):
        psnr = original_psnr(rgb, target)
        per_view[-1].update(
            psnr=float(psnr.mean()),
            pooled_psnr=float(-10 * torch.log10((rgb - target).square().mean())),
        )
        return psnr

    upstream.output_to_rgb = audited_rgb
    upstream.psnr = recorded_psnr
    result = upstream.benchmark_checkpoint(
        args.checkpoint, dataset, config, ssim, lpips, lpips_3dgs, device
    )
    report = {
        "complete": True,
        "backend": args.backend,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "gpu": torch.cuda.get_device_name(),
        "native_extension_loaded": "vorotracing.cuda._backend" in sys.modules,
        "result": result,
        "per_view": per_view,
        "protocol": "Unmodified upstream benchmark loop, all 37 half-resolution test images, ten warmups and twenty timed renders/view, including per-frame nearest-camera lookup; precomputed rays; FP16 textures, 4x8 screen tiling, Morton cell order; metrics use the actual timed inference settings.",
        "upstream_benchmark_sha256": digest(upstream.__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
