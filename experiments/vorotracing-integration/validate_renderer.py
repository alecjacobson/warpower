"""Compare Warp against the pinned upstream CUDA renderer on identical inputs."""

import argparse
import json
from pathlib import Path

import numerics
import numpy as np
import torch
import warp as wp
from scipy.spatial import Delaunay
from upstream import load
from warp_renderer import Nearest, Trace, infer, prepared_edges


def graph(points):
    indptr, indices = Delaunay(points.cpu().numpy()).vertex_neighbor_vertices
    return (
        torch.from_numpy(np.pad(indices, (0, 32))).cuda().to(torch.uint32),
        torch.from_numpy(indptr).cuda().to(torch.uint32),
    )


def difference(a, b):
    a, b = a.detach().double(), b.detach().double()
    error = a - b
    return dict(
        max_abs=float(error.abs().max()),
        relative_l2=float(error.norm() / b.norm().clamp_min(1e-20)),
        nonfinite=int((~torch.isfinite(a)).sum()),
        reference_nonfinite=int((~torch.isfinite(b)).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load("native")
    from vorotracing import cuda
    from vorotracing.cuda._wrapper import TraceVoroTracing, trace_octmap_infer

    numerics.instrument_native(cuda._wrapper)

    wp.init()
    torch.manual_seed(127)
    points = torch.randn(512, 3) * 2
    adj, offsets = graph(points)
    points = points.cuda()
    diffuse = torch.randn(512, 192, device="cuda") * 0.5
    specular = torch.randn_like(diffuse) * 0.2
    density = torch.rand(512, 1, device="cuda") * 3
    nearest = Nearest()
    nearest.build(points)
    output = {
        "reference_commit": "75910484d6d46af25e180044ca6b9091204fe6fa",
        "cases": [],
        "complete": False,
    }
    for case in ["outside", "inside", "axes", "transparent", "opaque", "cap", "small_scale"]:
        torch.manual_seed(213)
        o = torch.randn(2048, 3, device="cuda") * (4 if case == "outside" else 0.4)
        d = torch.nn.functional.normalize(torch.randn_like(o) - o, dim=-1)
        if case == "axes":
            d[:6] = torch.tensor(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], device="cuda"
            )
            d[6:12] = torch.nn.functional.normalize(d[:6] + 1e-7, dim=-1)
        case_points = points
        sigma = density.clone()
        if case == "transparent":
            sigma.zero_()
        if case == "opaque":
            sigma.fill_(1e4)
        if case == "small_scale":
            case_points = points * 1.0e-5
            o *= 1.0e-5
            sigma *= 1.0e5
            nearest.build(case_points)
        rays = torch.cat([o, d], dim=1)
        starts = nearest.query(case_points, o)
        brute = torch.cdist(o, case_points).argmin(1)
        assert torch.equal(starts.long(), brute)
        cap = 2 if case == "cap" else 1024
        params = [t.clone().requires_grad_() for t in [case_points, diffuse, specular, sigma]]
        ref_params = [t.detach().clone().requires_grad_() for t in params]
        got = Trace.apply(*params, adj, offsets, rays, starts, 0.001, cap)
        ref = TraceVoroTracing.apply(
            *ref_params, adj, offsets, rays, starts, None, 0.001, cap, False
        )
        rgba_grad = torch.randn_like(got[0])
        dist_grad = torch.randn_like(got[1])
        torch.autograd.backward([got[0], got[1]], [rgba_grad, dist_grad])
        torch.autograd.backward([ref[0], ref[5]], [rgba_grad, dist_grad.reshape(ref[5].shape)])
        row = {
            "name": case,
            "rgba": difference(got[0], ref[0]),
            "distortion": difference(got[1].flatten(), ref[5].flatten()),
            "traversal_count_mismatches": int(
                (got[2].long().flatten() != ref[3].long().flatten()).sum()
            ),
            "gradients": {
                name: difference(a.grad, b.grad)
                for name, a, b in zip(
                    ["points", "diffuse", "specular", "density"], params, ref_params
                )
            },
        }
        packed = [
            torch.nn.functional.pad(t.reshape(-1, 64, 3), (0, 1)).half().flatten(1)
            for t in [diffuse, specular]
        ]
        edges = prepared_edges(case_points, adj, offsets)
        permutation = torch.randperm(len(rays), device="cuda", dtype=torch.int32)
        row["inference"] = []
        for skip in [0.0, 0.001]:
            call = [
                case_points,
                *packed,
                sigma,
                adj,
                offsets,
                rays,
                starts,
                edges,
                permutation,
                0.01,
                cap,
                skip,
            ]
            row["inference"].append(
                {"skip": skip, **difference(infer(*call), trace_octmap_infer(*call))}
            )
        output["cases"].append(row)
        print(json.dumps(row), flush=True)
    output["complete"] = True
    output["passed"] = all(
        r["rgba"]["max_abs"] < 2e-5
        and r["distortion"]["max_abs"] < 2e-5
        and r["traversal_count_mismatches"] == 0
        and all(x["max_abs"] <= 0.001 and x["nonfinite"] == 0 for x in r["inference"])
        and all(g["nonfinite"] == 0 and g["relative_l2"] < 2e-3 for g in r["gradients"].values())
        for r in output["cases"]
    )
    output["numerics"] = numerics.snapshot()
    output["passed"] = output["passed"] and all(
        count == 0 for name, count in output["numerics"].items() if name.endswith("gradient")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    if not output["passed"]:
        raise RuntimeError("Renderer parity failed; inspect the saved report")


if __name__ == "__main__":
    main()
