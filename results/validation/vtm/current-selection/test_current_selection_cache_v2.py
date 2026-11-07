"""Force fresh geometry after each cache read and test explicit-call freshness."""
import numpy as np
import pytest
import warp as wp
import warpower
from current_selection_cache_v1 import CurrentSelection, install
from warpower import _selective

if _selective.SelectiveUpdate is not CurrentSelection:
    install()

from tests.test_selective import initial
from tests.test_vtm_core import valid_core
from tests.test_containment import cavity, fan
from warpower._vtm import VTMWorkspace

wp.set_module_options({'enable_backward': False})


@wp.kernel
def begin_audit(needed: wp.array[int], calls: wp.array[int]):
    calls[0] += 1
    if needed[0] == 0:
        calls[1] += 1


@wp.kernel
def compare_ints(a: wp.array[int], b: wp.array[int], count: wp.array[int], bounded: bool, bad: wp.array[int]):
    i = wp.tid()
    if not bounded or i < count[0]:
        if a[i] != b[i]:
            wp.atomic_or(bad, 0, 1)


@wp.kernel
def compare_longs(a: wp.array[wp.int64], b: wp.array[wp.int64], count: wp.array[int], bounded: bool, error: wp.array[int], bad: wp.array[int]):
    i = wp.tid()
    if error[0] == 0 and (not bounded or i < count[0]):
        if a[i] != b[i]:
            wp.atomic_or(bad, 0, 2)


class FreshOracle:
    def __init__(self, selection):
        self.selection = selection
        self.calls = wp.zeros(2, dtype=int, device=selection.device)
        self.bad = wp.zeros(1, dtype=int, device=selection.device)
        arrays = [(selection.keep, True), (selection.quality.view(wp.int64), True),
                  (selection.minimum.view(wp.int64), False), (selection.error, False),
                  (selection.retained_count, False)]
        self.arrays = [(a, wp.empty_like(a), bounded) for a, bounded in arrays]
        wp.load_module(module=__name__, device=selection.device)

    def select(self):
        s = self.selection
        wp.launch(begin_audit, 1, inputs=[s._current_needed, self.calls], device=s.device)
        s.select_current()
        for a, saved, _ in self.arrays:
            wp.copy(saved, a)
        s.evaluate()  # This call must recompute every ordinary/recovered result.
        for a, saved, bounded in self.arrays:
            args = [a, saved, s.core.full_tetrahedra_count, bounded]
            kernel = compare_ints
            if a.dtype == wp.int64:
                kernel = compare_longs
                args.append(s.error)
            wp.launch(kernel, len(a), inputs=args + [self.bad], device=s.device)


def test_every_optimizer_selection_matches_forced_fresh_evaluation(monkeypatch):
    fitted = initial(256, 20)
    core = VTMWorkspace(fitted.domain, wp.clone(fitted.points), max_iterations=8, selective_threshold=0.3)
    audit = FreshOracle(core.selection)
    original = core.selection.select_current

    def checked_select():
        # Avoid recursive interception while building the audit's first call.
        core.selection.select_current = original
        try:
            audit.select()
        finally:
            core.selection.select_current = checked_select

    monkeypatch.setattr(core.selection, 'select_current', checked_select)
    pointers = [core.points.ptr, core.full_tetrahedra.ptr, core.selection.keep.ptr]
    with wp.ScopedCapture() as capture:
        wp.copy(core.points, fitted.points)
        audit.calls.zero_()
        audit.bad.zero_()
        core.run()
    for _ in range(2):
        wp.capture_launch(capture.graph)
        assert core.error.numpy()[0] == 0
        assert audit.bad.numpy()[0] == 0
        assert audit.calls.numpy()[1] > 0
        assert [core.points.ptr, core.full_tetrahedra.ptr, core.selection.keep.ptr] == pointers
        assert core.selection._allow_current is False
        valid_core(core)


@wp.kernel
def produce_points(base: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int], sites: bool):
    i = wp.tid()
    scale = float(1.0)
    if phase[0] == 3:
        scale = 2.0
    points[i] = base[i] * scale
    if sites and phase[0] == 2:
        points[i] = wp.vec3(wp.nan)


@wp.kernel
def produce_controls(phase: wp.array[int], recovery: wp.array[int], error: wp.array[int]):
    recovery[0] = int(phase[0] == 1 or phase[0] == 3)
    error[0] = 0
    if phase[0] == 4:
        error[0] = 64


def recovered_selection():
    class Core:
        pass
    c = Core()
    p, t = fan()
    c.domain = warpower.SurfaceDomain(*cavity())
    c.device = c.domain.device
    c.points = wp.array(p, dtype=wp.vec3, device=c.device)
    c.full_tetrahedra = wp.array(t, device=c.device)
    c.full_tetrahedra_count = wp.full(1, len(t), dtype=int, device=c.device)
    c.n = len(p)
    c.mesh, c.boundary = Core(), Core()
    c.boundary.roles = wp.zeros(c.n, dtype=int, device=c.device)
    c.mesh.error = wp.zeros(1, dtype=int, device=c.device)
    c.error = wp.zeros(1, dtype=int, device=c.device)
    c.termination = wp.zeros(1, dtype=int, device=c.device)
    c._accepted = wp.ones(1, dtype=int, device=c.device)
    c._running = wp.ones(1, dtype=int, device=c.device)
    c.material_recovery = wp.zeros(1, dtype=int, device=c.device)
    return c, CurrentSelection(c, 0.4)


def test_recovered_current_results_match_fresh_with_device_failure_recovery(monkeypatch):
    c, s = recovered_selection()
    audit = FreshOracle(s)
    points, surface = wp.clone(c.points), wp.clone(c.domain.points)
    phase = wp.zeros(1, dtype=int, device=c.device)
    wp.load_module(module=__name__, device=c.device)

    def forbidden(*args, **kwargs):
        raise AssertionError('Host allocation/readback/synchronization in capture')

    with monkeypatch.context() as patch:
        for name in ['empty', 'zeros', 'ones', 'empty_like', 'clone', 'synchronize', 'synchronize_device']:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, 'numpy', forbidden)
        with wp.ScopedCapture() as capture:
            wp.launch(produce_points, c.n, inputs=[points, c.points, phase, True])
            wp.launch(produce_points, len(surface), inputs=[surface, c.domain.points, phase, False])
            wp.launch(produce_controls, 1, inputs=[phase, c.material_recovery, c.mesh.error])
            c.domain.refit()
            c._accepted.fill_(1)
            c._running.fill_(1)
            c.error.zero_()
            c.termination.zero_()
            s.reset()
            s.previous_minimum.zero_()
            audit.calls.zero_()
            audit.bad.zero_()
            s._check()
            audit.select()
    for value in range(6):
        phase.fill_(value)
        wp.capture_launch(capture.graph)
        assert audit.bad.numpy()[0] == 0
        if value in [2, 4]:
            assert s.error.numpy()[0] != 0 and audit.calls.numpy()[1] == 0
        else:
            assert s.error.numpy()[0] == 0 and audit.calls.numpy()[1] == 1
            assert s.retained_count.numpy()[0] == (5 if value in [1, 3] else 3)
            if value in [1, 3]:
                np.testing.assert_array_equal(s.keep.numpy(), s.recovery_extraction.selected.numpy())
                assert s.recovery_extraction.repaired_count.numpy()[0] == 2


@pytest.mark.parametrize('operation', ['select', 'evaluate'])
def test_direct_calls_remain_fresh_even_after_an_accepted_check(operation):
    c, s = recovered_selection()
    with wp.ScopedCapture() as capture:
        s.reset()
        s.previous_minimum.zero_()
        c._accepted.fill_(1)
        s._check()
        s.keep.fill_(-7)
        s.quality.fill_(-7)
        getattr(s, operation)()
    wp.capture_launch(capture.graph)
    assert s.error.numpy()[0] == 0 and s.retained_count.numpy()[0] == 3
    assert set(s.keep.numpy()).issubset({0, 1})
    assert np.all(s.quality.numpy() > 0)
    assert s._current_needed.numpy()[0] == 1
