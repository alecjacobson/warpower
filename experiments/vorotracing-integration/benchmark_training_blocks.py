"""Compare launch block sizes on a full checkpoint and uniformly sampled training rays."""

import argparse
import json
import sys
from pathlib import Path

import numerics
import torch
import warp_renderer as wr
from benchmark_renderer import timed
from upstream import load

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/radfoam-training")
    parser.add_argument("--rays", type=int, default=1000000)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "experiments/radfoam-integration"))
    from spatial import morton_permutation, permute_graph

    upstream = load("native")
    from vorotracing import cuda
    from vorotracing.datasets.datasets import ColmapConfig

    numerics.instrument_native(cuda._wrapper)
    torch.manual_seed(2026)
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    model = upstream.VoroTracing.from_pretrained(args.checkpoint, upstream.VoroTracingConfig())
    with torch.no_grad():
        perm = morton_permutation(model.primal_points)
        model.point_adjacency, model.point_adjacency_offsets = permute_graph(
            model.point_adjacency, model.point_adjacency_offsets, perm
        )
        for name in ["primal_points", "density", "att_diffuse", "att_specular"]:
            setattr(model, name, torch.nn.Parameter(getattr(model, name)[perm].contiguous()))
        model.bvh.build(model.primal_points)
    dataset = ColmapConfig(
        data_path=str(args.data.resolve()), scene="bonsai", init_points="original"
    ).load("train", 2)
    rays = dataset.sample_batch(args.rays)["rays"]
    starts = model.get_starting_point(rays)
    density = model.density.exp().detach().requires_grad_()
    params = [model.primal_points, model.att_diffuse, model.att_specular, density]
    original_launch = wr.wp.launch
    selection = [256, 256]

    def launch(kernel, *args, **kwargs):
        if kernel is wr.trace_forward:
            kwargs["block_dim"] = selection[0]
        if kernel is wr.trace_backward:
            kwargs["block_dim"] = selection[1]
        return original_launch(kernel, *args, **kwargs)

    wr.wp.launch = launch

    def replay(backend):
        for p in params:
            p.grad = None
        common = [*params, model.point_adjacency, model.point_adjacency_offsets, rays, starts]
        if backend == "native":
            rgba, _, _, _, _, dist = cuda.TraceVoroTracing.apply(*common, None, 0.001, 1024, False)
        else:
            rgba, dist, _ = wr.Trace.apply(*common, 0.001, 1024)
        (rgba.mean() + 0.002 * dist.mean()).backward()

    configs = [(256, 256), (128, 128), (64, 64), (256, 128), (128, 64), (64, 128), (128, 256)]
    report = {
        "complete": False,
        "scope": "Same full 2m-site native checkpoint with identical Morton-relabeled graph and uniformly sampled rays from all 255 training cameras. Timed FP32 forward + backward + gradient initialization and common counted sanitization. No optimizer or ray generation.",
        "samples": [],
        "native": [],
        "rays": args.rays,
        "checkpoint": str(args.checkpoint),
        "gpu": torch.cuda.get_device_name(),
    }
    for round_index in range(args.rounds):
        native = timed(lambda: replay("native"), args.repeats)
        report["native"].append(native)
        print("native", round_index, native, flush=True)
        for config in configs if round_index % 2 == 0 else configs[::-1]:
            selection[:] = config
            row = {
                "round": round_index,
                "forward_block": config[0],
                "backward_block": config[1],
                **timed(lambda: replay("warp"), args.repeats),
            }
            report["samples"].append(row)
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    report.update(complete=True, numerics=numerics.snapshot())
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
