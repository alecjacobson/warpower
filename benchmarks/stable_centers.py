"""Native replay audit for generated-center rounding and invalid recovery."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from tests.test_sphere_bounds import exact_sphere
from tests.test_stable_center import consume, evaluate, produce, rounded_fraction


def audit(profile=False, dot=None):
    path = Path(__file__).resolve().parents[1] / "tests/data/hand-fit256.npz"
    hand = np.load(path)["points"][[14, 50, 96, 233]]
    cases = np.stack(
        [hand, hand, np.random.default_rng(1109).normal(size=(4, 3)), hand * 2]
    ).astype(np.float32)
    cases[1, 0, 0] = np.nan
    source = wp.array(cases, dtype=wp.vec3, device="cuda:0")
    current = wp.empty((1, 4), dtype=wp.vec3, device=source.device)
    outputs = [
        wp.empty(1, dtype=t, device=source.device)
        for t in [wp.vec3, wp.vec3d, wp.float64, int, int]
    ]
    history = wp.empty(4, dtype=wp.vec3, device=source.device)
    states = wp.empty((4, 2), dtype=int, device=source.device)
    phase = wp.zeros(1, dtype=int, device=source.device)
    wp.load_module(module="tests.test_stable_center", device=source.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce, 4, inputs=[source, current, phase])
        wp.launch(evaluate, 1, inputs=[current, False, *outputs])
        wp.launch(consume, 1, inputs=[outputs[0], outputs[3], outputs[4], phase, history, states])
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
    result, values = states.numpy(), history.numpy()
    np.testing.assert_array_equal(result, [[1, 1], [0, 1], [1, 0], [1, 1]])
    for i in [0, 2, 3]:
        np.testing.assert_array_equal(
            values[i], [rounded_fraction(c) for c in exact_sphere(cases[i])[1]]
        )
    assert phase.numpy()[0] == 0
    return dict(
        scope="Four resident producer/center/consumer phases: a real hand cell requiring exact rounding, nonfinite input, an ordinary cell using the fast path, restored hand cell scaled by two. Device phase advance, final synchronization/readback only. Native profiling first warms one complete cycle outside the trace. Not full meshing or a throughput comparison.",
        warmup_replays=4 if profile else 0,
        columns=["valid", "exact_fallback"],
        states=result.tolist(),
        centers=values.tolist(),
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
