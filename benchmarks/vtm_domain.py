"""Native audit of the resident VTM material-direction guard."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_domain import cube
from tests.test_vtm_domain import arrays, consume, produce
from warpower import SurfaceDomain
from warpower._vtm_domain import constrain


def audit(profile=False, dot=None):
    domain = SurfaceDomain(*cube())
    points, roles, delta, edge, error = arrays(np.zeros((2, 3)), np.zeros((2, 3)))
    phase = wp.zeros(1, dtype=int)
    output = wp.empty((4, 3), dtype=wp.float64)
    for module in ["tests.test_vtm_domain", "warpower._vtm_domain"]:
        wp.load_module(module=module, device=points.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, 1, inputs=[phase, points, delta])
        error.zero_()
        wp.launch(constrain, 2, inputs=[domain.tree, points, roles, delta, edge, 0.25, 0.5, error])
        wp.launch(consume, 1, inputs=[phase, delta, error, output])
    if dot:
        wp.capture_debug_dot_print(capture.graph, str(dot), verbose=True)
    if profile:
        for _ in range(4):
            wp.capture_launch(capture.graph)
        wp.synchronize()
        import torch

        torch.cuda.cudart().cudaProfilerStart()
    for _ in range(4):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStop()
    values = output.numpy()
    np.testing.assert_array_equal(values[:, 0], [0, 64, 0, 0])
    assert 0 < values[0, 1] < 0.02 and values[3, 1] == values[0, 1]
    assert values[1, 1] == 0 and values[2, 1] == -0.25
    np.testing.assert_array_equal(values[:, 2], 0)
    assert phase.numpy()[0] == 0
    return dict(
        scope="Four resident producer/guard/consumer phases: interior, nonfinite, exterior, restored interior. A zero-direction inactive NaN slot remains untouched. One final sync/readback; native audit warms one complete cycle. This is not full meshing or a throughput benchmark.",
        columns=["error", "direction_x", "inactive_direction_squared"],
        records=values.tolist(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--dot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.profile, args.dot)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
