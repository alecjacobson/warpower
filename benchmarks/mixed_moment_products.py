"""Experimental FP32 moment products with unchanged FP64 sums (#274).

Production does not import this module. Near-range failures in the product
proposal retain the FP64 product; ordinary weights, owners, masses, energy and
all global acceptance/geometry checks remain unchanged.
"""

import argparse
import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

import warp as wp

from warpower import _bvh, _mixed_boundary
from warpower._mixed_boundary import nearest_site_mixed

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.func
def moment_product(weight: wp.float64, delta: wp.vec3d):
    proposal = wp.vec3(delta) * wp.float32(weight)
    safe = True
    for k in range(3):
        value = proposal[k]
        if not wp.isfinite(value):
            safe = False
        if value == wp.float32(0.0):
            if weight != wp.float64(0.0) and delta[k] != wp.float64(0.0):
                safe = False
        elif wp.abs(value) < wp.float32(1.0e-37):
            safe = False
    if safe:
        return wp.vec3d(proposal)
    return weight * delta


@wp.kernel
def accumulate_products32(
    tree: _bvh.Tree,
    points: wp.array[wp.vec3],
    active: wp.array[int],
    occupied: wp.array[int],
    samples: wp.array[wp.vec3],
    base_weights: wp.array[wp.float64],
    sizing: wp.array[float],
    dimension: int,
    owners: wp.array[int],
    mass: wp.array[wp.float64],
    moment: wp.array[wp.vec3d],
    energies: wp.array[wp.float64],
    total: wp.array[wp.float64],
    maximum: wp.array[wp.float64],
    component: int,
    error: wp.array[int],
):
    s = wp.tid()
    owners[s] = -1
    if error[0] != 0 or base_weights[s] == wp.float64(0.0):
        return
    h = wp.float64(sizing[s])
    if h <= wp.float64(0.0) or not wp.isfinite(h):
        wp.atomic_or(error, 0, 8)
        return
    denominator = h * h * h
    if dimension == 2:
        denominator *= h
    weight = base_weights[s] / denominator
    sample = wp.vec3d(samples[s])
    hit = nearest_site_mixed(tree, points, samples[s], active, occupied, False)
    if hit.index < 0 or not wp.isfinite(weight):
        wp.atomic_or(error, 0, 16)
        return
    owners[s] = hit.index
    delta = sample - wp.vec3d(points[hit.index])
    distance2 = wp.dot(delta, delta)
    wp.atomic_add(mass, hit.index, weight)
    wp.atomic_add(moment, hit.index, moment_product(weight, delta))
    wp.atomic_add(energies, component, weight * distance2)
    wp.atomic_add(total, component, weight)
    wp.atomic_max(maximum, component, wp.sqrt(distance2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--products", choices=["baseline", "float32"], required=True)
    parser.add_argument("--report", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.report.exists():
        parser.error("report exists")
    if (
        "--boundary-query-precision" not in remaining
        or remaining[remaining.index("--boundary-query-precision") + 1] != "mixed"
    ):
        parser.error("experiment requires explicit mixed boundary queries")
    source = Path(__file__)
    report = dict(
        scope=__doc__,
        products=args.products,
        revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver="examples.vtm_gallery",
        arguments=remaining,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    original = _mixed_boundary.accumulate_samples_mixed
    if args.products == "float32":
        wp.load_module(module=__name__, device="cuda:0")
        _mixed_boundary.accumulate_samples_mixed = accumulate_products32
    try:
        sys.argv = ["examples.vtm_gallery", *remaining]
        runpy.run_module("examples.vtm_gallery", run_name="__main__")
    finally:
        _mixed_boundary.accumulate_samples_mixed = original


if __name__ == "__main__":
    main()
