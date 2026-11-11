"""Matched-checkpoint inference and replay timings, excluding ray generation."""

import argparse
import json
import time
from pathlib import Path

import torch
from evaluate_inference import digest
from upstream import load
from validate_renderer import difference
from warp_renderer import Nearest, Trace, infer


def timed(call, repeats):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    a.record()
    for _ in range(repeats):
        call()
    b.record()
    b.synchronize()
    return {
        "gpu_ms": a.elapsed_time(b) / repeats,
        "wall_ms": (time.perf_counter() - wall) * 1000 / repeats,
    }


def paired_timing(calls, repeats, rounds=3):
    samples = {name: [] for name in calls}
    names = list(calls)
    for round_index in range(rounds):
        order = names if round_index % 2 == 0 else names[::-1]
        for name in order:
            samples[name].append(timed(calls[name], repeats))
    return {
        name: {
            "samples": rows,
            "gpu_ms": sum(r["gpu_ms"] for r in rows) / len(rows),
            "wall_ms": sum(r["wall_ms"] for r in rows) / len(rows),
        }
        for name, rows in samples.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/radfoam-training"))
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--views", type=int, nargs="+", default=[0, 10, 20, 36])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    upstream = load("native")
    import numerics
    from vorotracing import cuda
    from vorotracing.datasets.datasets import ColmapConfig

    numerics.instrument_native(cuda._wrapper)

    config = upstream.VoroTracingConfig()
    model = upstream.VoroTracing.from_pretrained(args.checkpoint, config)
    renderer = upstream.VoroTracingInfer(model)
    dataset = ColmapConfig(data_path=str(args.data), scene="bonsai", init_points="original").load(
        "test", downsample_factor=args.downsample
    )
    native_nn = renderer.bvh
    warp_nn = Nearest()
    warp_nn.build(renderer.primal_points)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "gpu": torch.cuda.get_device_name(),
        "views": [],
        "complete": False,
        "scope": "Same Morton-sorted checkpoint, padded FP16 octmaps, 4x8 ray tiling; each timed call includes nearest-camera lookup and tracing but excludes ray generation and RGB compositing.",
    }
    report["starting_cells"] = []
    training = ColmapConfig(data_path=str(args.data), scene="bonsai", init_points="original").load(
        "train", downsample_factor=args.downsample
    )
    cameras = [("test", dataset.poses[:, :, 3].cuda()), ("train", training.poses[:, :, 3].cuda())]
    del training
    for split, origins in cameras:
        for count in [65536, 1000000]:
            torch.manual_seed(77)
            queries = origins[torch.randint(len(origins), (count,), device="cuda")].contiguous()

            def original_starts():
                unique, inverse = torch.unique(queries, dim=0, return_inverse=True)
                return (
                    native_nn.query(renderer.primal_points, unique).long()[inverse].to(torch.uint32)
                )

            def warp_starts():
                return warp_nn.query(renderer.primal_points, queries)

            mismatch = int((original_starts().long() != warp_starts().long()).sum())
            assert mismatch == 0
            report["starting_cells"].append(
                {
                    "split": split,
                    "rays": count,
                    "cameras": len(origins),
                    "mismatches": mismatch,
                    **paired_timing({"native": original_starts, "warp": warp_starts}, args.repeats),
                }
            )
    for view in args.views:
        batch = dataset[view]
        rays, target = batch["rays"].cuda(), batch["rgbs"].cuda()
        origins = rays.reshape(-1, 6)[:1, :3]
        assert torch.equal(
            native_nn.query(renderer.primal_points, origins).long(),
            warp_nn.query(renderer.primal_points, origins).long(),
        )
        row = {"view": view, "shape": list(rays.shape[:-1]), "inference": []}
        for threshold, skip in [(0.001, 0.0), (0.01, 0.001)]:

            def native():
                return renderer.render(rays, weight_threshold=threshold, cell_skip_threshold=skip)

            def warp():
                start = (
                    warp_nn.query(renderer.primal_points, origins)
                    .expand(rays.shape[:-1])
                    .contiguous()
                )
                return infer(
                    renderer.primal_points,
                    renderer.diffuse_fp16,
                    renderer.specular_fp16,
                    renderer.density,
                    renderer.point_adjacency,
                    renderer.point_adjacency_offsets,
                    rays,
                    start,
                    renderer.adjacent_diff,
                    renderer._tile_perm(*rays.shape[:-1]),
                    threshold,
                    1024,
                    skip,
                )

            with torch.no_grad():
                ref, got = native(), warp()
                item = {
                    "transmittance_threshold": threshold,
                    "cell_skip_threshold": skip,
                    "parity": difference(got, ref),
                }
                timing = paired_timing({"native": native, "warp": warp}, args.repeats)
                for name, rgba, call in [("native", ref, native), ("warp", got, warp)]:
                    rgb = (rgba[..., :3].float() + 1 - rgba[..., 3:].float()).clamp(0, 1)
                    item[name] = {
                        **timing[name],
                        "psnr": float(-10 * torch.log10((rgb - target).square().mean())),
                    }
                row["inference"].append(item)
        if args.training:
            sample = rays.reshape(-1, 6)[:: max(rays.numel() // 6 // 65536, 1)][:65536].contiguous()
            training_nn = Nearest()
            training_nn.build(model.primal_points)
            starts = training_nn.query(model.primal_points, sample[:, :3])
            density = model.density.exp().detach().requires_grad_()
            params = [model.primal_points, model.att_diffuse, model.att_specular, density]

            def replay(backend):
                for parameter in params:
                    parameter.grad = None
                common = [
                    *params,
                    model.point_adjacency,
                    model.point_adjacency_offsets,
                    sample,
                    starts,
                ]
                if backend == "native":
                    rgba, _, _, _, _, dist = cuda.TraceVoroTracing.apply(
                        *common, None, 0.001, 1024, False
                    )
                else:
                    rgba, dist, _ = Trace.apply(*common, 0.001, 1024)
                (rgba.mean() + 0.002 * dist.mean()).backward()

            row["training"] = paired_timing(
                {"native": lambda: replay("native"), "warp": lambda: replay("warp")}, args.repeats
            )
        report["views"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    report.update(complete=True, numerics=numerics.snapshot())
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
