"""Private candidate acceptance, capacity and changing-input graph checks."""
import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay
from local_dewall_integration_v3 import install
from warpower import DelaunayWorkspace
from warpower import _delaunay_recovery

wp.set_module_options({'enable_backward': False})


@pytest.fixture(autouse=True)
def private_builder():
    original = _delaunay_recovery.DiagramWorkspace
    original_attempt = _delaunay_recovery.DelaunayWorkspace._base_attempt
    install()
    yield
    _delaunay_recovery.DiagramWorkspace = original
    _delaunay_recovery.DelaunayWorkspace._base_attempt = original_attempt


def cloud(n=1000):
    return np.random.default_rng(2026).uniform(-1, 1, (n, 3)).astype(np.float32)


def canonical(t):
    return {tuple(sorted(row)) for row in t}


def check(w, p):
    assert w.error.numpy()[0] == 0
    count = int(w.tetrahedra_count.numpy()[0])
    actual = w.tetrahedra.numpy()[:count]
    assert count == len(canonical(actual))
    unique, indices = np.unique(p, axis=0, return_index=True)
    ref = indices[Delaunay(unique).simplices]
    assert canonical(actual) == canonical(ref)
    return actual


@wp.kernel
def producer(frames: wp.array2d[wp.vec3], phase: wp.array[int], points: wp.array[wp.vec3]):
    i = wp.tid()
    points[i] = frames[phase[0], i]


@wp.kernel
def consume(tets: wp.array2d[int], count: wp.array[int], error: wp.array[int], checksum: wp.array[wp.int64]):
    i = wp.tid()
    if error[0] == 0 and i < count[0]:
        v = wp.int64(0)
        for k in range(4):
            v += wp.int64(tets[i, k])
        wp.atomic_add(checksum, 0, v)


@wp.kernel
def advance(phase: wp.array[int], count: int):
    phase[0] = (phase[0] + 1) % count


def test_device_producer_consumer_owner_changes_invalid_recovery(monkeypatch):
    p = cloud()
    frames = np.tile(p, (7, 1, 1))
    frames[1, 0] += np.array([0.05, -0.02, 0.01], np.float32)
    frames[2, 128:] = p[0]
    frames[3, 0] = p[1]
    frames[4, 0, 1] = np.nan
    frames[5] = p * np.float32(2.0**20) + np.float32(2.0**30)
    w = DelaunayWorkspace(p, fast_candidates=False, cache_strategy=False)
    source = wp.array(frames, dtype=wp.vec3, device=w.device)
    phase = wp.zeros(1, dtype=int, device=w.device)
    checksum = wp.zeros(1, dtype=wp.int64, device=w.device)
    wp.load_module(module=__name__, device=w.device)
    pointers = (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)

    def forbidden(*args, **kwargs):
        raise AssertionError('allocation/readback/synchronization inside capture')

    with monkeypatch.context() as patch:
        for name in ['empty', 'empty_like', 'zeros', 'ones', 'full', 'clone', 'synchronize', 'synchronize_device', 'synchronize_stream']:
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, 'numpy', forbidden)
        with wp.ScopedCapture(device=w.device) as capture:
            wp.launch(producer, len(p), inputs=[source, phase, w.points])
            w.run()
            checksum.zero_()
            wp.launch(consume, len(w.tetrahedra), inputs=[w.tetrahedra, w.tetrahedra_count, w.error, checksum])
            wp.launch(advance, 1, inputs=[phase, len(frames)])
    for i, frame in enumerate(frames):
        wp.capture_launch(capture.graph)
        if i == 4:
            assert w.error.numpy()[0] != 0
            assert w.tetrahedra_count.numpy()[0] == checksum.numpy()[0] == 0
        else:
            tets = check(w, frame)
            assert checksum.numpy()[0] == int(tets.astype(np.int64).sum())
        assert pointers == (w.points.ptr, w.tetrahedra.ptr, w.tetrahedra_count.ptr)
    assert phase.numpy()[0] == 0


@pytest.mark.parametrize('failure', ['overflow', 'empty', 'filter', 'completion'])
def test_forced_failure_runs_ordinary_constructor(monkeypatch, failure):
    p = cloud()
    w = DelaunayWorkspace(p, cache_strategy=False, fast_candidates=False)
    base = w._base
    if failure in ['overflow', 'empty']:
        original = base._wall.run
        def corrupt():
            original()
            base._wall.raw_count.fill_(len(base._wall.raw_tets) + 1 if failure == 'overflow' else 0)
        monkeypatch.setattr(base._wall, 'run', corrupt)
    else:
        original = base._wall_completion.run
        def corrupt():
            original()
            if failure == 'filter':
                base._wall_filter.error.fill_(2)
            else:
                base._wall_completion.error.fill_(16)
        monkeypatch.setattr(base._wall_completion, 'run', corrupt)
    with wp.ScopedCapture(device=w.device) as capture:
        w.run()
    wp.capture_launch(capture.graph)
    check(w, p)
    assert base.wall_accepted.numpy()[0] == 0
    assert base.wall_fallback.numpy()[0] == 1


def test_output_capacity_fails_without_partial_output():
    w = DelaunayWorkspace(cloud(128), tetrahedra_capacity=1, cache_strategy=False, fast_candidates=False).run()
    assert w.error.numpy()[0] != 0
    assert w.tetrahedra_count.numpy()[0] == 0


@pytest.mark.parametrize('scale', [2.0**-60, 2.0**60])
def test_extreme_finite_scales_use_original_coordinate_certificate(scale):
    p = cloud(128) * np.float32(scale)
    w = DelaunayWorkspace(p, fast_candidates=False).run()
    check(w, p)
