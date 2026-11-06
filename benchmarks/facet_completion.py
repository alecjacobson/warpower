"""Exercise experimental facet completion with a resident producer/consumer loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from warpower import DelaunayWorkspace
from warpower._facet_completion import FacetCompletion
from warpower._mesh_topology import identity
from warpower._site_bvh import SiteHierarchy

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def produce_points(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[i] = source[i] + wp.vec3(wp.where(phase[0] == 2, 64.0, 0.0), 0.0, 0.0)


@wp.kernel
def produce_seed(
    source: wp.array2d[int], seeds: wp.array2d[int], count: wp.array[int], phase: wp.array[int]
):
    i = wp.tid()
    for k in range(4):
        seeds[i, k] = source[i, k]
    if i == 0:
        count[0] = wp.where(phase[0] == 3, source.shape[0], 1)
        if phase[0] == 1:
            seeds[0, 0] = -1


@wp.kernel
def consume(
    tets: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    rounds: wp.array[int],
    phase: wp.array[int],
    records: wp.array2d[wp.int64],
):
    row = phase[0]
    records[row, 0] = wp.int64(count[0])
    records[row, 1] = wp.int64(error[0])
    records[row, 2] = wp.int64(rounds[0])
    checksum = wp.int64(0)
    for i in range(count[0]):
        for k in range(4):
            checksum += wp.int64(tets[i, k])
    records[row, 3] = checksum
    phase[0] = (row + 1) % 4


def audit(profile=False, dot=None):
    host = np.random.default_rng(108).integers(-16, 17, (32, 3)).astype(np.float32)
    assert len(np.unique(host, axis=0)) == len(host)
    reference = DelaunayWorkspace(host).run()
    assert reference.error.numpy()[0] == 0
    full = reference.tetrahedra.numpy()[: reference.tetrahedra_count.numpy()[0]]
    source = wp.array(host, dtype=wp.vec3, device=reference.device)
    points = wp.clone(source)
    seed_source = wp.array(full, dtype=int, device=points.device)
    seeds = wp.empty_like(seed_source)
    count = wp.zeros(1, dtype=int, device=points.device)
    hierarchy = SiteHierarchy(points)
    wp.launch(identity, len(points), inputs=[hierarchy.tree.owners])
    completion = FacetCompletion(points, seeds, count, hierarchy.tree, capacity=512)
    ready = wp.ones(1, dtype=int, device=points.device)
    phase = wp.zeros(1, dtype=int, device=points.device)
    records = wp.zeros((4, 4), dtype=wp.int64, device=points.device)
    wp.load_module(module=__name__, device=points.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce_points, len(points), inputs=[source, points, phase])
        wp.launch(produce_seed, len(seeds), inputs=[seed_source, seeds, count, phase])
        # SiteHierarchy requires a child graph so CUB sort scratch is owned at
        # capture setup instead of becoming allocation/free nodes on replay.
        # This diagnostic producer always supplies finite, unique input sites.
        wp.capture_if(ready, hierarchy.rebuild)
        completion.run()
        wp.launch(
            consume,
            1,
            inputs=[
                completion.tetrahedra,
                completion.tetrahedra_count,
                completion.error,
                completion.rounds,
                phase,
                records,
            ],
        )
    if dot:
        wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    for _ in range(4):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    if profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(4):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = records.numpy()
    np.testing.assert_array_equal(values[:, 0], [len(full), 0, len(full), len(full)])
    np.testing.assert_array_equal(values[:, 1], [0, 1, 0, 0])
    np.testing.assert_array_equal(values[:, 3], [full.sum(), 0, full.sum(), full.sum()])
    assert values[0, 2] > 0 and values[2, 2] > 0 and values[3, 2] == 0
    assert phase.numpy()[0] == 0
    return dict(
        scope="Four warmed resident producer/facet-completion/consumer replays: one seed, invalid seed, translated points with one seed, complete seed. Hierarchy rebuilt within the graph. No host phase changes; one final sync/readback outside the measured native replay range. Experimental internal path, not full VTM or performance comparison.",
        columns=["count", "error", "rounds", "checksum"],
        records=values.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.profile, args.dot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
