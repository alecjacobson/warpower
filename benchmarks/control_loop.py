"""Audit six device-produced VTM budget/mode phases in one reusable graph."""

import argparse
import json
from pathlib import Path

import warp as wp

from tests.test_selective import initial
from tests.test_vtm_controls import produce_controls
from tests.test_vtm_core import consume
from warpower._vtm import VTMWorkspace


@wp.kernel
def record(
    phase: wp.array[int],
    error: wp.array[int],
    iterations: wp.array[int],
    count: wp.array[int],
    volume: wp.array[wp.float64],
    log: wp.array2d[int],
    volumes: wp.array[wp.float64],
):
    i = phase[0]
    log[i, 0] = error[0]
    log[i, 1] = iterations[0]
    log[i, 2] = count[0]
    volumes[i] = volume[0]
    phase[0] = (i + 1) % 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fitted = initial(128, 8)
    source = wp.clone(fitted.points)
    phase = wp.zeros(1, dtype=int, device=fitted.device)
    budget = wp.zeros(1, dtype=int, device=fitted.device)
    mode = wp.zeros(1, dtype=int, device=fitted.device)
    volume = wp.zeros(1, dtype=wp.float64, device=fitted.device)
    log = wp.zeros((6, 3), dtype=int, device=fitted.device)
    volumes = wp.zeros(6, dtype=wp.float64, device=fitted.device)
    core = VTMWorkspace(
        fitted.domain,
        wp.clone(source),
        max_iterations=5,
        iteration_limit=budget,
        selective_threshold=0.3,
        selective_enabled=mode,
    )
    for module in (__name__, "tests.test_vtm_controls", "tests.test_vtm_core"):
        wp.load_module(module=module, device=core.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce_controls, core.n, inputs=[source, core.points, phase, budget, mode])
        core.run()
        volume.zero_()
        wp.launch(
            consume,
            len(core.full_tetrahedra),
            inputs=[
                core.points,
                core.full_tetrahedra,
                core.full_tetrahedra_count,
                core.error,
                volume,
            ],
        )
        wp.launch(
            record,
            1,
            inputs=[
                phase,
                core.error,
                core.iterations,
                core.full_tetrahedra_count,
                volume,
                log,
                volumes,
            ],
        )
    for _ in range(6):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.profile:
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(6):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = log.numpy()
    assert values[:, 0].tolist() == [0, 0, 32, 0, 32, 0]
    assert values[0, 1] == 0 and values[1, 1] == 2 and values[5, 1] == 1
    assert values[2, 2] == values[4, 2] == 0
    assert volumes.numpy()[2] == volumes.numpy()[4] == 0
    if args.dot:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        wp.capture_debug_dot_print(cap.graph, str(args.dot), verbose=True)
    result = dict(
        phases=["refresh", "global2", "invalid_budget", "selective2", "invalid_mode", "global1"],
        error_iterations_tetrahedra=values.tolist(),
        consumed_volumes=volumes.numpy().tolist(),
        device=core.device.name,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
