"""CPU regressions for exact initial-mesh ordering and source-capture provenance."""

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

from benchmarks.replay.fixture import Fixture, Writer, digest
from benchmarks.replay.package_pairs import package
from benchmarks.replay.runner import reference


def exact_orientation(points, tetrahedron):
    coordinates = [[Fraction(float(x)) for x in points[i]] for i in tetrahedron]
    a, b, c = [[row[j] - coordinates[0][j] for j in range(3)] for row in coordinates[1:]]
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


class ReplayPackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ids = np.array([101, 303, 707, 909], dtype=np.int64)

    def fake_cgal(self, order=(0, 1, 2, 3)):
        """A protocol test double, not a geometric validity oracle."""
        path = self.root / "fake-cgal"
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            "assert int(Path(sys.argv[1]).read_text().splitlines()[0]) == 4\n"
            f"order = {list(map(int, order))!r}\n"
            "Path(sys.argv[2]).write_text('1\\n' + ' '.join(map(str, order)) + '\\n')\n"
            "print(json.dumps({'CGAL': 'protocol-test-double', 'vertices': 4, "
            "'tetrahedra': 1, 'build_seconds': 0}), file=sys.stderr)\n"
        )
        path.chmod(0o755)
        return path

    def two_frame_fixture(self, points):
        rows = np.array([2, 0, 3, 1])
        writer = Writer(self.root / "source", name="orientation-source", provenance={})
        writer.append(points[rows], self.ids[rows], reset=True)
        writer.append(points[rows], self.ids[rows])
        writer.finalize()
        return Fixture(writer.root)

    def test_near_degenerate_positive_mesh_keeps_cgal_orientation(self):
        # Every coordinate is an exactly representable float32 integer. The
        # rational determinant is +1. NumPy's float64 determinant was -1.1918
        # on the qualification host, so the old "orientation repair" inverted it.
        points = np.array(
            [
                [0, 0, 0],
                [-4565, -333183, 16250],
                [-60519, -4416584, 2465378],
                [-101631, -7417202, 2610009],
            ],
            dtype=np.float32,
        )
        self.assertEqual(exact_orientation(points, [0, 1, 2, 3]), 1)
        fixture = self.two_frame_fixture(points)
        binary = self.fake_cgal()
        output = self.root / "packaged"
        result = package(fixture.root, [1], output, binary)
        self.assertEqual(result["transitions"], [1])
        packaged = Fixture(output)
        entry = packaged.document["provenance"]["initial_meshes"][0]
        self.assertEqual(entry["sha256"], digest(output / entry["path"]))
        with np.load(output / entry["path"], allow_pickle=False) as mesh:
            np.testing.assert_array_equal(mesh["points"], points)
            np.testing.assert_array_equal(mesh["site_ids"], self.ids)
            np.testing.assert_array_equal(mesh["tetrahedra"], [[0, 1, 2, 3]])
            self.assertEqual(exact_orientation(mesh["points"], mesh["tetrahedra"][0]), 1)

    def test_reference_and_package_preserve_noncanonical_positive_vertex_order(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        fixture = self.two_frame_fixture(points)
        order = np.array([1, 2, 0, 3])  # Even permutation; positive but not ID-sorted.
        self.assertGreater(exact_orientation(points, order), 0)
        binary = self.fake_cgal(order)
        frame = fixture.frame(0)
        oriented, error, _ = reference(frame, "cgal", binary, preserve_orientation=True)
        self.assertIsNone(error)
        np.testing.assert_array_equal(oriented, self.ids[order][None, :])
        canonical, error, _ = reference(frame, "cgal", binary)
        self.assertIsNone(error)
        np.testing.assert_array_equal(canonical, self.ids[None, :])
        output = self.root / "packaged"
        package(fixture.root, [1], output, binary)
        with np.load(output / "initial-000.npz", allow_pickle=False) as mesh:
            np.testing.assert_array_equal(mesh["tetrahedra"], order[None, :])
            np.testing.assert_array_equal(mesh["site_ids"][mesh["tetrahedra"]], oriented)

    def test_out_of_pair_restoration_and_native_pointers_keep_source_provenance(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
        proposed = points.copy()
        proposed[3, 2] = np.float32(1.25)
        rows = np.array([3, 1, 0, 2])
        metadata = dict(
            previous_native_output=dict(
                frame=11,
                payload=dict(path="connectivity/000011.npz", sha256="a" * 64),
            ),
            producer_event_indices=[19, 21],
            optimizer_updates=23,
        )
        writer = Writer(self.root / "source", name="restore-source", provenance={"scene": "test"})
        writer.append(points, self.ids, reset=True, event="initial", stage="initialization")
        writer.append(proposed[rows], self.ids[rows], event="proposal", source_step=22)
        writer.append(
            points[rows],
            self.ids[rows],
            event="restoration",
            stage="quality-restoration",
            restored_from=0,
            source_step=23,
            metadata=metadata,
        )
        writer.finalize()
        source = Fixture(writer.root)
        source.validate()
        output = self.root / "packaged"
        package(source.root, [2], output, self.fake_cgal())
        packaged = Fixture(output)
        restored = packaged.frame(1)
        self.assertEqual(packaged.document["provenance"]["original_manifest_sha256"], source.sha256)
        self.assertIsNone(restored.record["restored_from"])
        # Package frame zero is the rejected proposal, not the restoration target.
        self.assertNotEqual(
            packaged.frame(0).canonical()[0].tobytes(), restored.canonical()[0].tobytes()
        )
        self.assertEqual(
            restored.canonical()[0].tobytes(), source.frame(0).canonical()[0].tobytes()
        )
        saved = restored.record["metadata"]
        self.assertEqual(saved["original_frame"], 2)
        self.assertEqual(saved["original_restored_from"], 0)
        self.assertEqual(saved["original_restore_target_sha256"], source.records[0]["sha256"])
        self.assertEqual(saved["original_metadata"], metadata)
        self.assertNotIn("previous_native_output", saved)
        self.assertNotIn("producer_event_indices", saved)
        self.assertIn("original capture", saved["original_metadata_scope"])
        self.assertIn("not this pair package", saved["original_metadata_scope"])
        self.assertEqual(
            (restored.record["stage"], restored.record["source_step"]), ("quality-restoration", 23)
        )
        self.assertFalse((output / metadata["previous_native_output"]["payload"]["path"]).exists())


if __name__ == "__main__":
    unittest.main()
