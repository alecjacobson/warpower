"""CPU checks for live producer identity bookkeeping and observed row contracts."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.replay.fixture import Fixture
from benchmarks.replay.radfoam_capture import (
    CaptureBudgetExceeded,
    RadFoamRecorder,
    instrument_scene,
)
from benchmarks.replay.radfoam_capture_audit import audit


class ProducerCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.recorder = RadFoamRecorder(Path(self.temp.name) / "capture", provenance={})
        self.points = np.arange(15, dtype=np.float32).reshape(5, 3)
        self.recorder.initialize(5)

    def test_real_operations_preserve_survivor_ids_and_never_recycle_births(self):
        r = self.recorder
        r.input(self.points, incremental=False, constructor=True)
        permutation = np.array([4, 2, 0, 1, 3])
        r.permute(permutation)
        points = self.points[permutation]
        r.optimizer_step()
        r.input(points, incremental=True)
        added = np.array([[20, 21, 22], [30, 31, 32]], np.float32)
        r.insert(added)
        points = np.concatenate([points, added])
        mask = np.array([False, True, False, False, True, False, False])
        r.prune(mask)
        points = points[~mask]
        r.begin_geometry(step=7, stage="post-prune-densify", rebuild=True, incremental=False)
        r.input(points, incremental=False)
        r.finish()
        fixture = Fixture(r.writer.root)
        fixture.validate()
        np.testing.assert_array_equal(fixture.frame(2).inserted_ids, [5, 6])
        np.testing.assert_array_equal(fixture.frame(2).deleted_ids, [2, 3])
        np.testing.assert_array_equal(fixture.frame(2).site_ids, [4, 0, 1, 5, 6])
        self.assertEqual(fixture.frame(1).canonical()[0].tobytes(), self.points.tobytes())
        self.assertEqual(fixture.records[2]["identity_epoch"], 0)
        self.assertTrue(fixture.records[2]["reset"])
        evidence = audit(r.root)
        self.assertTrue(evidence["producer_identity_chain_valid"])
        self.assertEqual((evidence["inserted_sites"], evidence["deleted_sites"]), (2, 2))
        self.assertEqual(evidence["cgal_comparisons"], 0)

    def test_connectivity_sidecar_maps_internal_rows_to_stable_ids(self):
        r = self.recorder
        index, ids = r.input(self.points, incremental=False, constructor=True)
        permutation = np.array([4, 2, 0, 1, 3])
        r.success(index, ids, np.array([[0, 1, 2, 3]]), permutation)
        with np.load(r.root / r.outputs[0]["payload"]["path"]) as data:
            np.testing.assert_array_equal(data["site_ids"][data["tetrahedra"]], [[4, 2, 0, 1]])
        r.permute(permutation)
        r.input(self.points[permutation], incremental=True)
        r.finish()
        frame = Fixture(r.writer.root).frame(1)
        self.assertIsNone(frame.previous_tets)
        self.assertEqual(frame.record["metadata"]["previous_native_output"]["frame"], 0)
        self.assertEqual(audit(r.root)["native_outputs"], 1)

    def test_audit_rejects_native_connectivity_with_wrong_identity(self):
        r = self.recorder
        index, ids = r.input(self.points, incremental=False, constructor=True)
        r.success(index, ids + 100, np.array([[0, 1, 2, 3]]))
        r.finish()
        with self.assertRaisesRegex(ValueError, "current input IDs"):
            audit(r.root)

    def test_failed_retry_retains_actual_new_input_without_inventing_valid_state(self):
        r = self.recorder
        index, _ = r.input(self.points, incremental=False, constructor=True)
        r.failure(index, RuntimeError("native failure"))
        changed = self.points.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], np.float32(1))
        r.input(changed, incremental=False)
        r.finish()
        fixture = Fixture(r.writer.root)
        self.assertEqual(fixture.records[1]["event"], "native-retry")
        self.assertIsNone(fixture.records[1]["metadata"]["previous_native_output"])
        self.assertEqual(fixture.frame(1).points.tobytes(), changed.tobytes())

    def test_invalid_permutation_and_budget_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "bijection"):
            self.recorder.permute(np.array([0, 1, 2, 3, 3]))
        self.recorder.max_bytes = 1024**2
        with self.assertRaises(CaptureBudgetExceeded):
            self.recorder.input(self.points, incremental=False)
        self.assertFalse(self.recorder.writer.document["frames"])

    def test_hooks_follow_upstream_mutations_and_check_coordinates(self):
        class Scene:
            def __init__(self, points):
                self.primal_points = points

            def permute_points(self, rows):
                self.primal_points = self.primal_points[rows]

            def densification_postfix(self, values):
                self.primal_points = np.concatenate([self.primal_points, values["primal_points"]])

            def prune_points(self, mask):
                self.primal_points = self.primal_points[~mask]

        model = instrument_scene(Scene, self.recorder)(self.points.copy())
        model.permute_points(np.array([4, 3, 2, 1, 0]))
        model.densification_postfix({"primal_points": self.points[:1] + 50})
        model.prune_points(np.array([False, False, True, False, False, False]))
        np.testing.assert_array_equal(self.recorder.registry.ids, [4, 3, 1, 0, 5])

        class Wrong(Scene):
            def permute_points(self, rows):
                self.primal_points = self.primal_points[rows] + 1

        with self.assertRaisesRegex(RuntimeError, "coordinate values"):
            instrument_scene(Wrong, self.recorder)(model.primal_points.copy()).permute_points(
                np.arange(5)
            )


if __name__ == "__main__":
    unittest.main()
