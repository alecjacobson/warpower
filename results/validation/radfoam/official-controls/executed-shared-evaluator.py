"""Evaluate saved fits with both upstream and pooled-RGB PSNR on identical rays."""

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from official_reduced import load_upstream


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", nargs=2, metavar=("NAME", "PATH"), action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("RADFOAM_SOURCE_DIR", "/home/horde/references/radfoam")),
    )
    parser.add_argument(
        "--build",
        type=Path,
        default=Path(os.environ.get("RADFOAM_BUILD_DIR", ".work/radfoam-fitting-20261110/native")),
    )
    args = parser.parse_args()
    upstream = load_upstream(args.source.resolve(), args.build.resolve())
    test = upstream.DataHandler(
        SimpleNamespace(
            dataset="colmap", scene="bonsai", data_path=str(args.data.resolve()), patch_based=False
        ),
        rays_per_batch=0,
    )
    test.reload("test", downsample=4)
    report = {
        "resolution": test.img_wh,
        "views": len(test.rays),
        "protocol": "Official COLMAP every-eighth-view held-out split, images_4, full-image rays, explicit precomputed starting cells, white background and [0,1] clamp; each checkpoint retains its saved graph.",
        "metrics": {
            "official": "Mean over images of mean over channels of -10 log10(mean pixel squared error in each channel).",
            "pooled_rgb": "Mean over images of -10 log10(mean squared error over all pixels and RGB channels).",
        },
        "evaluator_sha256": digest(__file__),
        "upstream_psnr_source_sha256": digest(args.source / "radfoam_model/utils.py"),
        "models": [],
    }
    for name, checkpoint in args.checkpoint:
        model_args = SimpleNamespace(
            sh_degree=3, init_points=64, final_points=262144, activation_scale=1.0
        )
        model = upstream.RadFoamScene(model_args, device=torch.device("cuda"))
        model.load_pt(checkpoint)
        ray_fetcher = upstream.radfoam.BatchFetcher(test.rays, batch_size=1, shuffle=False)
        rgb_fetcher = upstream.radfoam.BatchFetcher(test.rgbs, batch_size=1, shuffle=False)
        rows = []
        nonfinite = 0
        started = time.perf_counter()
        with torch.no_grad():
            starts = model.get_starting_point(
                test.rays[:, 0, 0].cuda(), model.primal_points, model.aabb_tree
            )
            for i in range(len(test.rays)):
                rays, target = ray_fetcher.next()[0], rgb_fetcher.next()[0]
                rgba = model(rays, starts[i])[0]
                nonfinite += int((~torch.isfinite(rgba)).sum())
                rgb = (rgba[..., :3] + (1 - rgba[..., -1:])).reshape(target.shape).clamp(0, 1)
                channel_psnr = upstream.psnr(rgb, target).flatten()
                pooled_psnr = -10 * torch.log10((rgb - target).square().mean())
                rows.append(
                    {
                        "view": i,
                        "channel_psnr": channel_psnr.tolist(),
                        "official_psnr": float(channel_psnr.mean()),
                        "pooled_rgb_psnr": float(pooled_psnr),
                    }
                )
                torch.cuda.synchronize()
        row = {
            "name": name,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": digest(checkpoint),
            "sites": len(model.primal_points),
            "official_psnr": sum(r["official_psnr"] for r in rows) / len(rows),
            "pooled_rgb_psnr": sum(r["pooled_rgb_psnr"] for r in rows) / len(rows),
            "nonfinite_rgba": nonfinite,
            "per_view": rows,
            "evaluation_seconds": time.perf_counter() - started,
        }
        report["models"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "per_view"}), flush=True)
        del model, ray_fetcher, rgb_fetcher
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
