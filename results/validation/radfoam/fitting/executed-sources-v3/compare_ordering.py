"""Measure spatial ordering with identical RadFoam parameters and graph edges."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from integration import AuditedPipeline, COLMAPDataset, initialize_runtime, native
from radfoam_model.render import TraceRays
from spatial import morton_permutation, permute_graph

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot


def timed(function):
    torch.cuda.synchronize()
    t = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return time.perf_counter() - t, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initialize_runtime()
    saved = torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    points = saved["xyz"]
    attrs = torch.cat(
        [saved["color_dc"], saved["color_sh"], F.softplus(saved["density"], beta=10)], dim=1
    )
    adjacency = saved["adjacency"].to(torch.uint32)
    offsets = saved["adjacency_offsets"].to(torch.uint32)
    train = COLMAPDataset("data/radfoam-training/bonsai", "train", 4)
    rng = np.random.default_rng(2026)
    indices = rng.choice(train.all_rays.numel() // 6, 65536, replace=False)
    rays = train.all_rays.reshape(-1, 6)[indices].cuda()
    cameras, inverse = torch.unique(rays[:, :3], dim=0, return_inverse=True)
    ordering_seconds, permutation = timed(lambda: morton_permutation(points))
    relabel_seconds, graph = timed(lambda: permute_graph(adjacency, offsets, permutation))
    variants = {
        "input_order": (points, attrs, adjacency, offsets),
        "warp_morton": (points[permutation].contiguous(), attrs[permutation].contiguous(), *graph),
    }
    pipeline = native.create_pipeline(3, torch.float32)
    trees = {key: native.build_aabb_tree(value[0]) for key, value in variants.items()}

    def query(key):
        return native.nn(variants[key][0], trees[key], cameras)

    starts = {key: query(key).long()[inverse].to(torch.uint32) for key in variants}
    original_indices = starts["input_order"].long()
    sorted_indices = permutation[starts["warp_morton"].long()]
    assert torch.equal(original_indices, sorted_indices)

    def trace(key):
        return pipeline.trace_forward(*variants[key], rays, starts[key])["rgba"]

    def full_forward(key):
        unique, inv = torch.unique(rays[:, :3], dim=0, return_inverse=True)
        first = native.nn(variants[key][0], trees[key], unique).long()[inv].to(torch.uint32)
        return pipeline.trace_forward(*variants[key], rays, first)["rgba"]

    rgba = {key: trace(key) for key in variants}
    difference = float((rgba["input_order"] - rgba["warp_morton"]).abs().max())
    assert difference < 1e-5, difference
    # Check row-wise graph equality after undoing the permutation; exact integer
    # relabeling must not be mistaken for changed adjacency or clipping.
    deg0 = offsets.view(torch.int32)[1:] - offsets.view(torch.int32)[:-1]
    deg1 = graph[1].view(torch.int32)[1:] - graph[1].view(torch.int32)[:-1]
    assert torch.equal(deg1, deg0[permutation])
    old_edges = saved["adjacency"].cpu().numpy()
    old_off = saved["adjacency_offsets"].cpu().numpy()
    perm = permutation.cpu().numpy()
    new_edges = graph[0].cpu().numpy()
    new_off = graph[1].cpu().numpy()
    assert all(
        np.array_equal(
            perm[new_edges[new_off[i] : new_off[i + 1]]], old_edges[old_off[j] : old_off[j + 1]]
        )
        for i, j in enumerate(perm)
    )
    samples = {
        key: {
            stage: [] for stage in ["aabb", "nearest_cameras", "trace_known_starts", "full_forward"]
        }
        for key in variants
    }
    quota_before = cpu_quota_snapshot()
    for repeat in range(23):
        keys = list(variants)
        if repeat % 2:
            keys.reverse()
        for key in keys:
            for stage, fn in [
                ("aabb", lambda: native.build_aabb_tree(variants[key][0])),
                ("nearest_cameras", lambda: query(key)),
                ("trace_known_starts", lambda: trace(key)),
                ("full_forward", lambda: full_forward(key)),
            ]:
                seconds, result = timed(fn)
                if repeat >= 3:
                    samples[key][stage].append(seconds)
    quota = cpu_quota_delta(quota_before, cpu_quota_snapshot())
    preparation = {"ordering": [], "graph_relabel": []}
    for repeat in range(23):
        seconds, _ = timed(lambda: morton_permutation(points))
        if repeat >= 3:
            preparation["ordering"].append(seconds)
        seconds, _ = timed(lambda: permute_graph(adjacency, offsets, permutation))
        if repeat >= 3:
            preparation["graph_relabel"].append(seconds)
    audited = AuditedPipeline(pipeline)
    gradients = {}
    for key, (p, a, edges, off) in variants.items():
        p, a = p.detach().requires_grad_(), a.detach().requires_grad_()
        output = TraceRays.apply(audited, p, a, edges, off, rays, starts[key], None, False)[0]
        output.square().mean().backward()
        gradients[key] = [p.grad, a.grad]
    gradient_checks = {}
    for name, original, ordered in zip(
        ["position", "attribute"], gradients["input_order"], gradients["warp_morton"]
    ):
        original = original[permutation]
        torch.testing.assert_close(ordered, original, atol=1e-6, rtol=1e-3)
        gradient_checks[name] = {
            "max_abs_difference": float((ordered - original).abs().max()),
            "relative_l2_difference": float((ordered - original).norm() / original.norm()),
        }
    report = {
        "scope": "Same fitted Bonsai parameters and exactly the same directed graph relabeled by Warp Morton order. Upstream RadFoam AABB/NN/renderer; 65536 identical rays. Three warm-ups, twenty alternating-order samples. Ordering and graph relabel preparation reported separately.",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "sites": len(points),
        "rays": len(rays),
        "cameras": len(cameras),
        "ordering_seconds": ordering_seconds,
        "relabel_seconds": relabel_seconds,
        "preparation_note": "The first preparation calls above include module loading; warm medians below exclude it.",
        "warm_preparation_median_seconds": {
            key: float(np.median(values)) for key, values in preparation.items()
        },
        "gradient_checks": gradient_checks,
        "raw_render_gradient_checks": audited.counts,
        "same_start_sites": True,
        "same_directed_graph": True,
        "rgba_max_abs_difference": difference,
        "cpu_quota": quota,
        "cpu_throttled": any(x["throttled_periods"] for x in quota.values()),
        "samples_seconds": samples,
        "median_seconds": {
            key: {stage: float(np.median(values)) for stage, values in data.items()}
            for key, data in samples.items()
        },
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("spatial.py")]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "samples_seconds"}), flush=True)


if __name__ == "__main__":
    main()
