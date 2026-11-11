"""Independent directional derivatives and exact camera-cache reuse checks."""

import argparse
import json
from pathlib import Path

import numerics
import torch
import warp as wp
from validate_renderer import graph
from warp_renderer import Nearest, Trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wp.init()
    torch.manual_seed(551)
    points = torch.randn(512, 3)
    adjacency, offsets = graph(points)
    points = points.cuda()
    nearest = Nearest()
    nearest.build(points)
    camera_cases = []
    for name, cameras in [("repeated", 255), ("colliding_unique", 15000)]:
        origins = torch.randn(cameras, 3, device="cuda")
        queries = origins[torch.arange(32768, device="cuda") % cameras].contiguous()
        # Small batches exercise the uncached hierarchy search. The large
        # batch must give exactly these answers even with hash collisions.
        expected = torch.cat([nearest.query(points, q) for q in queries.split(2048)])
        actual = nearest.query(points, queries)
        mismatch = int((expected.long() != actual.long()).sum())
        camera_cases.append(
            {"name": name, "queries": len(queries), "cameras": cameras, "mismatches": mismatch}
        )
        assert mismatch == 0

    origins = torch.randn(1024, 3, device="cuda") * 2
    directions = torch.nn.functional.normalize(torch.randn_like(origins) - origins, dim=-1)
    rays = torch.cat([origins, directions], -1)
    starts = nearest.query(points, origins)
    params = [
        points,
        torch.randn(512, 192, device="cuda") * 0.2,
        torch.randn(512, 192, device="cuda") * 0.2,
        0.5 + torch.rand(512, 1, device="cuda"),
    ]
    params = [p.detach().requires_grad_() for p in params]
    rgba_weights = torch.randn(1024, 4, device="cuda")
    distortion_weights = torch.randn(1024, device="cuda") * 0.1

    def loss(values):
        # A negligible positive tail threshold avoids the reference backward's
        # undefined division after exactly-zero transmittance. Positions and
        # the graph remain fixed; omitted color weight is at most 1e-10.
        rgba, distortion, _ = Trace.apply(*values, adjacency, offsets, rays, starts, 1.0e-10, 1024)
        return (rgba.double() * rgba_weights).sum() + (
            distortion.double() * distortion_weights
        ).sum()

    loss(params).backward()
    derivatives = []
    for index, name in [(1, "diffuse"), (2, "specular"), (3, "density")]:
        for probe in range(3):
            direction = torch.randn_like(params[index])
            analytic = float((params[index].grad.double() * direction).sum())
            approximations = []
            for epsilon in [0.01, 0.005]:
                minus, plus = [p.detach() for p in params], [p.detach() for p in params]
                minus[index] = minus[index] - epsilon * direction
                plus[index] = plus[index] + epsilon * direction
                numeric = float((loss(plus) - loss(minus)) / (2 * epsilon))
                approximations.append(numeric)
            # Richardson extrapolation suppresses the O(epsilon^2) term.
            extrapolated = (4 * approximations[1] - approximations[0]) / 3
            error = abs(extrapolated - analytic) / max(abs(analytic), 0.1)
            derivatives.append(
                {
                    "parameter": name,
                    "probe": probe,
                    "analytic": analytic,
                    "central_differences": approximations,
                    "extrapolated": extrapolated,
                    "relative_error": error,
                }
            )
    raw = numerics.snapshot()
    output = {
        "complete": True,
        "camera_cache": camera_cases,
        "derivatives": derivatives,
        "numerics": raw,
        "position_derivatives": "Not finite-differenced: matching the upstream custom backward intentionally stops texture-coordinate derivatives and omits position derivatives of distortion.",
        "passed": all(d["relative_error"] < 0.005 for d in derivatives)
        and all(v == 0 for k, v in raw.items() if k.endswith("gradient")),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)
    assert output["passed"], "See the saved contract report"


if __name__ == "__main__":
    main()
