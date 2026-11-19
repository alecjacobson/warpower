"""Exercise the public geometry names and their compatibility aliases."""

import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import warpower
from tests.oracle import csr_sets, reference_cells
from tests.test_tetrahedra import reference, tet_set


@pytest.mark.parametrize(
    "previous,current",
    [
        ("DiagramWorkspace", "UpdatableDiagram"),
        ("DelaunayWorkspace", "UpdatableDelaunay"),
        ("BoundaryWorkspace", "BoundaryFitter"),
        ("VariationalMeshingWorkspace", "VariationalMesher"),
    ],
)
def test_deprecated_names_preserve_class_identity_and_signatures(previous, current):
    """Preserve old imports and identify their replacements in warnings."""
    replacement = getattr(warpower, current)
    with pytest.warns(
        DeprecationWarning, match=f"{previous} is deprecated; use warpower.{current}"
    ):
        original = getattr(warpower, previous)
    np.testing.assert_equal(original is replacement, True)
    np.testing.assert_equal(replacement.__name__, current)
    np.testing.assert_equal(previous in warpower.__all__ and current in warpower.__all__, True)
    np.testing.assert_equal(inspect.signature(original) == inspect.signature(replacement), True)
    if current.startswith("Updatable"):
        np.testing.assert_equal(replacement.run is replacement.update, True)


def test_public_names_preserve_lazy_imports():
    """Load only requested services when importing the new public names."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import unittest
import warnings
import warpower

checks = unittest.TestCase()
optional = ("_graph", "_delaunay_recovery", "_boundary", "_mesher")
for name in optional:
    checks.assertNotIn("warpower." + name, sys.modules)
with warnings.catch_warnings():
    warnings.simplefilter("error", DeprecationWarning)
    from warpower import UpdatableDiagram
checks.assertEqual(UpdatableDiagram.__name__, "UpdatableDiagram")
for name in optional[1:]:
    checks.assertNotIn("warpower." + name, sys.modules)
""",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        timeout=60,
    )


@pytest.mark.parametrize("mode", ["voronoi", "power", "delaunay"])
def test_captured_updates_refresh_geometry_in_stable_buffers(mode, monkeypatch):
    """Refresh changed inputs and recover from invalid input through update()."""
    rng = np.random.default_rng(93)
    positions = rng.normal(size=(32, 3)).astype(np.float32)
    weights = rng.normal(0, 0.05, len(positions)).astype(np.float32)
    stream = wp.Stream(wp.get_device("cuda:0"))
    with wp.ScopedStream(stream):
        points = wp.array(positions, dtype=wp.vec3, device="cuda:0")
        device_weights = wp.array(weights, device="cuda:0")
        if mode == "delaunay":
            geometry = warpower.UpdatableDelaunay(points, chunk_size=17)
        else:
            geometry = warpower.UpdatableDiagram(
                points, device_weights if mode == "power" else None, chunk_size=17
            )
        np.testing.assert_equal(geometry.update(stream=stream) is geometry, True)

        def forbidden(*args, **kwargs):
            raise AssertionError("Update allocated or read geometry back to the host")

        with monkeypatch.context() as guard:
            guard.setattr(wp.array, "numpy", forbidden)
            for name in (
                "empty",
                "zeros",
                "full",
                "synchronize",
                "synchronize_stream",
                "synchronize_device",
            ):
                guard.setattr(wp, name, forbidden)
            with wp.ScopedCapture(stream=stream) as capture:
                geometry.update(stream=stream)

        arrays = [geometry.points, geometry.error, geometry.tetrahedra, geometry.tetrahedra_count]
        if mode != "delaunay":
            arrays.extend([geometry.adjacency, geometry.offsets, geometry.adjacency_count])
        pointers = [array.ptr for array in arrays]
        topologies = []
        for step in range(2):
            if step:
                positions += rng.normal(0, 0.3, positions.shape).astype(np.float32)
                weights += rng.normal(0, 0.2, weights.shape).astype(np.float32)
            wp.copy(points, wp.array(positions, dtype=wp.vec3, device="cuda:0"))
            wp.copy(device_weights, wp.array(weights, device="cuda:0"))
            wp.capture_launch(capture.graph, stream=stream)
            np.testing.assert_array_equal(geometry.error.numpy(), [0])
            np.testing.assert_array_equal([array.ptr for array in arrays], pointers)
            if mode == "delaunay":
                count = int(geometry.tetrahedra_count.numpy()[0])
                topology = tet_set(geometry.tetrahedra[:count].numpy())
                np.testing.assert_equal(topology, reference(positions))
                np.testing.assert_array_equal(geometry.certificate_error.numpy(), [0])
            else:
                count = int(geometry.adjacency_count.numpy()[0])
                topology = csr_sets(geometry.adjacency[:count].numpy(), geometry.offsets.numpy())
                expected, _ = reference_cells(positions, weights if mode == "power" else None)
                np.testing.assert_equal(topology, expected)
            topologies.append(topology)
        np.testing.assert_equal(topologies[0] != topologies[1], True)

        invalid = positions.copy()
        invalid[0, 0] = np.nan
        wp.copy(points, wp.array(invalid, dtype=wp.vec3, device="cuda:0"))
        wp.capture_launch(capture.graph, stream=stream)
        np.testing.assert_equal(int(geometry.error.numpy()[0]) != 0, True)
        np.testing.assert_array_equal(geometry.tetrahedra_count.numpy(), [0])
        if mode != "delaunay":
            np.testing.assert_array_equal(geometry.adjacency_count.numpy(), [0])
            np.testing.assert_array_equal(geometry.offsets.numpy(), np.zeros(33, dtype=np.int32))
        wp.copy(points, wp.array(positions, dtype=wp.vec3, device="cuda:0"))
        np.testing.assert_equal(geometry.run(stream=stream) is geometry, True)
        np.testing.assert_array_equal(geometry.error.numpy(), [0])
