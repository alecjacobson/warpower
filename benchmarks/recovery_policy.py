"""Compare reconstruction-cache policies on the same changing-input meshing loop."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import warpower
from benchmarks.mesher_loop import record
from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot
from tests.test_domain import cube
from tests.test_mesher import produce, validate
from tests.test_vtm_core import consume
from warpower._mesher import start_growth


class HistoricalMesher(warpower.VariationalMeshingWorkspace):
    """Reproduce the pre-#81 solve entry from 6750927 for a fixed comparison.

    This small copy isolates cache policy from the production default. Numerical
    stages and publication are inherited. The enclosing graph applies a reset
    when requested, including the two invalid-input phases in the measurements.
    """

    def _solve(self):
        wp.copy(self.points, self.sampler.points, count=self.initial_count)
        self._launch(start_growth, 1, [self._state])
        if self.refinement is not None:
            wp.capture_while(self.grow, self._growth)
        wp.capture_if(self.ready, self._polish)


@wp.kernel
def probe_ready(preferred: wp.array[int], flag: wp.array[int]):
    flag[0] = int(preferred[0] == 2)


@wp.kernel
def preferred_needed(error: wp.array[int], flag: wp.array[int], base_attempted: wp.array[int]):
    flag[0] = int(error[0] == 2)
    if flag[0] != 0:
        # Preserve the original auxiliary fallback if the cached strategy also
        # fails. Repeating the base in that rare fallback is intentional here.
        base_attempted[0] = 0


@wp.kernel
def log_base(log: wp.array[int]):
    log[4] += 1


@wp.kernel
def log_strategy(
    base: wp.array[int], attempts: wp.array[int], symbolic: wp.array[int], log: wp.array[int]
):
    log[0] += 1
    log[1] += base[0]
    log[2] += int(attempts[0] > 0)
    log[3] += symbolic[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["cached", "uncached", "reset", "probe"], required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    a = parser.parse_args()
    d = warpower.SurfaceDomain(*cube(), surface_subdivisions=4)
    source = wp.clone(d.points)
    target = wp.full(1, 96, dtype=int, device=d.device)
    seed = wp.full(1, 2026, dtype=int, device=d.device)
    phase = wp.zeros(1, dtype=int, device=d.device)
    volume = wp.zeros(1, dtype=wp.float64, device=d.device)
    logs = wp.zeros((5, 5), dtype=int, device=d.device)
    volumes = wp.zeros(5, dtype=wp.float64, device=d.device)
    mesh = HistoricalMesher(
        d, 128, target_count=target, initial_count=64, seed=seed, quality_rounds=1
    )
    w = mesh.core.mesh
    if a.policy == "uncached":
        w.cache_strategy = False
    flag = wp.zeros(1, dtype=int, device=d.device)
    strategies = wp.zeros(5, dtype=int, device=d.device)
    original_base = w._base_path

    def base():
        wp.launch(log_base, 1, inputs=[strategies])
        original_base()

    w._base_path = base
    if a.policy == "probe":
        preferred = w._preferred_path

        def preferred_probe():
            wp.launch(probe_ready, 1, inputs=[w._preferred, flag])
            wp.capture_if(flag, w._base_path)
            wp.launch(preferred_needed, 1, inputs=[w.error, flag, w.base_attempted])
            wp.capture_if(flag, preferred)

        w._preferred_path = preferred_probe
    original_run = w.run

    def run(stream=None):
        original_run(stream)
        wp.launch(
            log_strategy, 1, inputs=[w.base_attempted, w.attempts, w.symbolic_attempted, strategies]
        )

    w.run = run
    for mod in [__name__, "tests.test_mesher", "tests.test_vtm_core", "benchmarks.mesher_loop"]:
        wp.load_module(module=mod, device=d.device)
    with wp.ScopedCapture() as cap:
        wp.launch(produce, len(source), inputs=[source, d.points, phase, target, seed])
        d.refit()
        if a.policy == "reset":
            w._preferred.zero_()
        mesh.run()
        volume.zero_()
        wp.launch(
            consume,
            len(mesh.tetrahedra),
            inputs=[mesh.points, mesh.tetrahedra, mesh.tetrahedra_count, mesh.error, volume],
        )
        wp.launch(record, 1, inputs=[phase, mesh.error, mesh.counts, volume, logs, volumes])
    for _ in range(5):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    samples = []
    quota = cpu_quota_snapshot()
    strategies.zero_()
    for _ in range(a.repeats):
        wp.record_event(begin)
        for _ in range(5):
            wp.capture_launch(cap.graph)
        wp.record_event(end)
        wp.synchronize_event(end)
        samples.append(wp.get_event_elapsed_time(begin, end))
    cpu = cpu_quota_delta(quota, cpu_quota_snapshot())
    if a.reference:
        validate(mesh)
    result = dict(
        scope="Five-phase producer/domain/full-mesher/consumer loop per sample, three valid solves and two invalid inputs. Experimental capture-time strategy wiring, outside library defaults.",
        policy=a.policy,
        device=d.device.name,
        warp=wp.__version__,
        repeats=a.repeats,
        gpu_milliseconds=samples,
        median_ms=float(np.median(samples)),
        cpu_quota_delta=cpu,
        cpu_throttled=any(v["throttled_periods"] for v in cpu.values()),
        strategy_calls_total_baseflag_auxiliary_symbolic_actualbase=strategies.numpy().tolist(),
        error_and_counts=logs.numpy().tolist(),
        consumed_volumes=volumes.numpy().tolist(),
        final_topology=mesh.topology.numpy().tolist(),
        final_metrics=mesh.metrics.numpy().tolist(),
        independent_final_validation=a.reference,
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert (
        result["error_and_counts"][0][0]
        == result["error_and_counts"][1][0]
        == result["error_and_counts"][4][0]
        == 0
    )
    assert result["error_and_counts"][2][0] == result["error_and_counts"][3][0] == 1
    np.savez_compressed(
        a.output.with_suffix(".npz"),
        points=mesh.points.numpy()[: mesh.point_count.numpy()[0]],
        tets=mesh.tetrahedra.numpy()[: mesh.tetrahedra_count.numpy()[0]],
        full=mesh.full_tetrahedra.numpy()[: mesh.full_tetrahedra_count.numpy()[0]],
    )


if __name__ == "__main__":
    main()
