"""Audit a producer/mesher/consumer graph with changing geometry, target and seed."""

import argparse
import json
from pathlib import Path

import warp as wp

import warpower
from tests.test_domain import cube
from tests.test_mesher import produce
from tests.test_vtm_core import consume


@wp.kernel
def record(
    phase: wp.array[int],
    error: wp.array[int],
    counts: wp.array[int],
    volume: wp.array[wp.float64],
    log: wp.array2d[int],
    volumes: wp.array[wp.float64],
):
    i = phase[0]
    log[i, 0] = error[0]
    for k in range(4):
        log[i, k + 1] = counts[k]
    volumes[i] = volume[0]
    phase[0] = (i + 1) % 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    source = wp.clone(d.points)
    target = wp.full(1, 96, dtype=int, device=d.device)
    seed = wp.full(1, 2026, dtype=int, device=d.device)
    phase = wp.zeros(1, dtype=int, device=d.device)
    volume = wp.zeros(1, dtype=wp.float64, device=d.device)
    log = wp.zeros((5, 5), dtype=int, device=d.device)
    volumes = wp.zeros(5, dtype=wp.float64, device=d.device)
    mesh = warpower.VariationalMeshingWorkspace(
        d, 128, target_count=target, initial_count=64, seed=seed, quality_rounds=1
    )
    for module in (__name__, "tests.test_mesher", "tests.test_vtm_core"):
        wp.load_module(module=module, device=d.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, len(source), inputs=[source, d.points, phase, target, seed])
        d.refit()
        mesh.run()
        volume.zero_()
        wp.launch(
            consume,
            len(mesh.tetrahedra),
            inputs=[mesh.points, mesh.tetrahedra, mesh.tetrahedra_count, mesh.error, volume],
        )
        wp.launch(record, 1, inputs=[phase, mesh.error, mesh.counts, volume, log, volumes])
    for _ in range(5):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(5):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    values, consumed = log.numpy(), volumes.numpy()
    assert values[:, 0].tolist() == [0, 0, 1, 1, 0]
    assert values[:, 1].tolist() == [96, 128, 0, 0, 96]
    assert not values[2:4, 1:].any() and not consumed[2:4].any()
    assert all(consumed[i] > 0 for i in [0, 1, 4])
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(cap.graph, str(args.dot), verbose=True)
    result = dict(
        phases=["valid96", "deformed128", "invalid_target", "collapsed_domain", "recovered96"],
        error_and_counts=values.tolist(),
        consumed_volumes=consumed.tolist(),
        device=d.device.name,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
