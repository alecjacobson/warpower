import numpy as np
import pytest
import warp as wp

import warpower
from tests.test_domain import combine, cube, torus
from warpower._domain_topology import DomainTopology


@pytest.mark.parametrize("case", ["cube", "torus", "cavity-and-components"])
def test_input_shell_invariants_and_volumes(case):
    if case == "cube":
        p, f = cube()
        expected = [1, 1, 0, 2]
    elif case == "torus":
        p, f = torus(24, 12)
        expected = [1, 1, 0, 0]
    else:
        inner = cube(0.4)
        p, f = combine(cube(), (inner[0], inner[1][:, ::-1].copy()), cube(0.3, (4, 0, 0)))
        expected = [2, 3, 1, 6]
    d = warpower.SurfaceDomain(p, f)
    t = DomainTopology(d).run()
    assert t.error.numpy()[0] == 0
    np.testing.assert_array_equal(t.counts.numpy(), expected)
    tri = p.astype(np.float64)[f]
    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    np.testing.assert_allclose(t.volume.numpy()[0], volume, rtol=1e-12)
    labels = t.labels.numpy()
    assert len(np.unique(labels)) == expected[1]
    for i, shell in enumerate(labels):
        assert labels[shell] == shell


def test_unreferenced_vertices_do_not_change_euler_or_volume():
    p, f = cube()
    p = np.concatenate((p, [[10, 10, 10]])).astype(np.float32)
    t = DomainTopology(warpower.SurfaceDomain(p, f)).run()
    assert t.error.numpy()[0] == 0
    np.testing.assert_array_equal(t.counts.numpy(), [1, 1, 0, 2])
    np.testing.assert_allclose(t.volume.numpy(), 8.0, rtol=1e-12)


def test_invalid_immutable_surface_topology_is_gated():
    p, f = cube()
    t = DomainTopology(warpower.SurfaceDomain(p, f[:-1].copy())).run()
    assert t.error.numpy()[0] == 1
    np.testing.assert_array_equal(t.counts.numpy(), 0)
    np.testing.assert_array_equal(t.volume.numpy(), 0)


@wp.kernel
def produce(source: wp.array[wp.vec3], points: wp.array[wp.vec3], phase: wp.array[int]):
    i = wp.tid()
    p = source[i] * float(1 + phase[0])
    if phase[0] == 2:
        p = wp.vec3(0.0)
    points[i] = p


def test_captured_geometry_updates_and_invalid_recovery(monkeypatch, tmp_path):
    d = warpower.SurfaceDomain(*cube())
    t = DomainTopology(d)
    source = wp.clone(d.points)
    phase = wp.zeros(1, dtype=int, device=d.device)
    wp.load_module(module=__name__, device=d.device)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host allocation, readback, or synchronization")

    with monkeypatch.context() as patch:
        for name in (
            "empty",
            "zeros",
            "ones",
            "full",
            "empty_like",
            "clone",
            "synchronize",
            "synchronize_device",
            "synchronize_stream",
        ):
            patch.setattr(wp, name, forbidden)
        patch.setattr(wp.array, "numpy", forbidden)
        with wp.ScopedCapture() as cap:
            wp.launch(produce, len(source), inputs=[source, d.points, phase])
            d.refit()
            t.run()
        wp.capture_launch(cap.graph)
    labels = t.labels.numpy()
    for value in [0, 1, 2, 0]:
        phase.fill_(value)
        wp.capture_launch(cap.graph)
        if value == 2:
            assert t.error.numpy()[0] == 1
            np.testing.assert_array_equal(t.counts.numpy(), 0)
            assert t.volume.numpy()[0] == 0
        else:
            assert t.error.numpy()[0] == 0
            np.testing.assert_array_equal(t.counts.numpy(), [1, 1, 0, 2])
            np.testing.assert_allclose(t.volume.numpy(), 8.0 * (1 + value) ** 3, rtol=1e-12)
        np.testing.assert_array_equal(t.labels.numpy(), labels)
    path = tmp_path / "domain-topology.dot"
    wp.capture_debug_dot_print(cap.graph, str(path), verbose=True)
    assert all(node not in path.read_text() for node in ("MEM_ALLOC", "MEM_FREE", "HOST"))
