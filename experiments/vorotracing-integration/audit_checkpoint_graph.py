"""Isolate adjacency effects by rebuilding a fixed checkpoint and re-rendering it."""

import argparse
import json
import time
from pathlib import Path

import numerics
import torch
from evaluate_inference import digest
from upstream import load


def edge_keys(adjacency, offsets):
    offsets = offsets.long()
    degrees = offsets[1:] - offsets[:-1]
    sources = torch.repeat_interleave(torch.arange(len(degrees), device=offsets.device), degrees)
    return ((sources << 32) | adjacency[: len(sources)].long()).sort().values


def missing(a, b):
    if len(b) == 0:
        return a
    indices = torch.searchsorted(b, a)
    return a[(indices == len(b)) | (b[indices.clamp_max(len(b) - 1)] != a)]


def sample_edges(keys):
    return [[int(k >> 32), int(k & 0xFFFFFFFF)] for k in keys[:32].cpu().tolist()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replacement", choices=["paragram", "warpower"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    args = parser.parse_args()
    upstream = load("native", audit=True)
    import benchmark_eval
    from vorotracing import cuda
    from vorotracing.datasets.datasets import ColmapConfig

    config, _ = benchmark_eval.load_config(args.checkpoint)
    model = upstream.VoroTracing.from_pretrained(args.checkpoint, config)
    original = (model.point_adjacency, model.point_adjacency_offsets)
    torch.cuda.synchronize()
    start = time.perf_counter()
    if args.replacement == "paragram":
        adjacency, offsets, status = cuda.build_voronoi_adjacency(model.primal_points)
    else:
        import warpower

        result = warpower.voronoi_diagram(model.primal_points.detach())
        adjacency = torch.nn.functional.pad(result.adjacency, (0, 32)).view(torch.uint32)
        offsets, status = result.offsets.view(torch.uint32), result.status
    torch.cuda.synchronize()
    build_seconds = time.perf_counter() - start
    values, counts = torch.unique(status, return_counts=True)
    a, b = edge_keys(*original), edge_keys(adjacency, offsets)
    removed, added = missing(a, b), missing(b, a)
    report = {
        "complete": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "replacement": args.replacement,
        "rebuild_seconds_including_first_call_setup": build_seconds,
        "status_counts": dict(zip(map(str, values.cpu().tolist()), counts.cpu().tolist())),
        "original_directed_edges": len(a),
        "replacement_directed_edges": len(b),
        "original_duplicate_edges": int((a[1:] == a[:-1]).sum()),
        "replacement_duplicate_edges": int((b[1:] == b[:-1]).sum()),
        "removed_directed_edges": len(removed),
        "added_directed_edges": len(added),
        "removed_sample": sample_edges(removed),
        "added_sample": sample_edges(added),
        "protocol": "Fixed positions and appearance; only CSR adjacency changes. Native FP32 renderer and nearest queries, all 37 half-resolution held-out views, threshold 0.001, cap 1024, white background. Directed edge sets ignore row ordering. Agreement does not constitute exact geometric certification.",
        "views": [],
    }
    del a, b, removed, added
    dataset = ColmapConfig(
        data_path=str(args.data.resolve()), scene="bonsai", init_points="original"
    ).load("test", 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        starts = model.get_starting_point(dataset.poses[:, :, 3].cuda())
        for view in range(len(dataset)):
            batch = dataset[view]
            rays, target = batch["rays"].cuda(), batch["rgbs"].cuda()
            row = {"view": view, "image_name": dataset.image_names[view]}
            images = []
            for name, graph in [("original", original), ("replacement", (adjacency, offsets))]:
                model.point_adjacency, model.point_adjacency_offsets = graph
                numerics.reset()
                rgba, _, _, _, _, _ = model(rays, starts[view])
                rgb = benchmark_eval.output_to_rgb(rgba)
                images.append(rgb)
                row[name] = {
                    "psnr": float(benchmark_eval.psnr(rgb, target).mean()),
                    "numerics": numerics.snapshot(),
                }
            delta = (images[1] - images[0]).abs()
            row.update(
                rgb_max_abs=float(delta.max()),
                rgb_mean_abs=float(delta.mean()),
                pixels_over_1e_minus_3=int((delta.max(dim=-1).values > 1e-3).sum()),
                psnr_delta=row["replacement"]["psnr"] - row["original"]["psnr"],
            )
            report["views"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    report.update(
        complete=True,
        mean_psnr_delta=sum(v["psnr_delta"] for v in report["views"]) / len(report["views"]),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
