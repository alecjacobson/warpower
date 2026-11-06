"""Measure captured automatic-sizing refits on structured/enriched torus controls."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_delaunay_recovery import exact_empty
from tests.test_domain import torus
from tests.test_tetrahedra import check_mesh, reference


@wp.kernel
def deform(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    points[i] = source[i] * (1.0 + 0.01 * float(phase[0] % 5))


@wp.kernel
def advance(phase: wp.array[int]):
    phase[0] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--major", type=int, default=16)
    parser.add_argument("--minor", type=int, default=8)
    parser.add_argument("--subdivisions", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    start = time.perf_counter()
    domain = warpower.SurfaceDomain(*torus(a.major, a.minor))
    auto = warpower.AutomaticSizingWorkspace(domain, control_subdivisions=a.subdivisions)
    source = wp.clone(domain.points)
    phase = wp.zeros(1, dtype=int, device=domain.device)
    wp.load_module(module=__name__, device=domain.device)
    graphs = []
    for changing in [False, True]:
        with wp.ScopedCapture() as cap:
            if changing:
                wp.launch(deform, len(source), inputs=[source, domain.points, phase])
                domain.refit()
            auto.refit()
            if changing:
                wp.launch(advance, 1, inputs=[phase])
        graphs.append(cap.graph)
    wp.synchronize()
    setup = time.perf_counter() - start
    first, last = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    timings = []
    for graph in graphs:
        for _ in range(2):
            wp.capture_launch(graph)
        wp.synchronize()
        quota = cpu_quota_snapshot()
        samples, strategies = [], []
        for _ in range(a.repeats):
            wp.record_event(first)
            wp.capture_launch(graph)
            wp.record_event(last)
            wp.synchronize_event(last)
            samples.append(wp.get_event_elapsed_time(first, last))
            strategies.append(
                [
                    int(v.numpy()[0])
                    for v in (
                        auto.mesh.base_attempted,
                        auto.mesh.attempts,
                        auto.mesh.symbolic_recovered,
                        auto.error,
                    )
                ]
            )
        delta = cpu_quota_delta(quota, cpu_quota_snapshot())
        timings.append(
            dict(
                samples_ms=samples,
                median_ms=float(np.median(samples)),
                cpu_quota_delta=delta,
                strategy_samples_base_attempted_auxiliary_attempts_symbolic_success_error=strategies,
            )
        )
        assert domain.error.numpy()[0] == auto.error.numpy()[0] == 0
    points = auto.control_points.numpy()
    cells = auto.mesh.tetrahedra.numpy()[: auto.mesh.tetrahedra_count.numpy()[0]]
    check_mesh(points, cells)
    differing = None
    if a.reference:
        expected = reference(points)
        different = np.array([t for t in cells if tuple(sorted(t)) not in expected], np.int32)
        differing = len(different)
        if differing:
            exact_empty(points, different)
    result = dict(
        scope="Automatic sizing only: control-site Delaunay, Voronoi poles and field refit. Changing-input graph also includes a device surface producer and domain refit. No material-site sampling or VTM optimization.",
        device=domain.device.name,
        warp=wp.__version__,
        parameters=vars(a) | {"output": str(a.output)},
        setup_seconds_including_initial_automatic_refit=setup,
        controls=auto.n,
        tetrahedra=len(cells),
        finite_poles=int(auto.pole_count.numpy()[0]),
        static_refit=timings[0],
        changing_input_refit=timings[1],
        symbolic_recovered=int(auto.mesh.symbolic_recovered.numpy()[0]),
        attempts=int(auto.mesh.attempts.numpy()[0]),
        local_feature_size_quantiles=np.quantile(
            auto.local_feature_size.numpy(), [0, 0.5, 1]
        ).tolist(),
        independent_full_hull_check=True,
        scipy_reference_and_rational_tie_checks=a.reference,
        scipy_differing_cells=differing,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        a.output.with_suffix(".npz"),
        points=points,
        tetrahedra=cells,
        lfs=auto.local_feature_size.numpy(),
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
