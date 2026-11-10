"""CPU checks for replay identity, restoration, timing state, and malformed inputs."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.replay.backends import ScipyBackend
from benchmarks.replay.exporters import export_radfoam, synthetic
from benchmarks.replay.fixture import Fixture, SiteRegistry, Writer, motion_features
from benchmarks.replay.runner import run, supervise


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.points = np.random.default_rng(22).uniform(-1, 1, (32, 3)).astype(np.float32)
        self.ids = np.arange(32, dtype=np.int64) + 100

    def writer(self, name="fixture"):
        return Writer(self.root / name, name=name, provenance={"seed": 22})

    def fixture(self):
        writer = self.writer()
        writer.append(self.points, self.ids, reset=True)
        writer.append(self.points[::-1], self.ids[::-1], event="reorder")
        moved = self.points.copy()
        moved[0, 0] += 0.01
        writer.append(moved, self.ids, event="update")
        writer.append(self.points, self.ids, event="restore", restored_from=0)
        writer.finalize()
        return Fixture(writer.root)

    def test_reorder_is_not_motion_and_restoration_is_exact(self):
        fixture = self.fixture()
        self.assertTrue(fixture.validate()["correspondence_valid"])
        first, reordered = fixture.frame(0), fixture.frame(1)
        self.assertEqual(first.canonical()[0].tobytes(), reordered.canonical()[0].tobytes())
        self.assertEqual(motion_features(first, reordered)["moved_fraction"], 0)
        np.testing.assert_array_equal(reordered.rows_from_previous, np.arange(31, -1, -1))
        self.assertEqual(motion_features(reordered, fixture.frame(2))["moved_fraction"], 1 / 32)

    def test_active_prefix_insert_delete_and_inactive_nonfinite(self):
        writer = self.writer()
        points = self.points.copy()
        points[-1] = np.nan
        writer.append(points, self.ids, active_count=20)
        writer.append(points, self.ids, active_count=25)
        writer.append(points, self.ids, active_count=18)
        writer.finalize()
        fixture = Fixture(writer.root)
        fixture.validate()
        self.assertTrue(fixture.frame(0).record["active_finite"])
        self.assertEqual(fixture.frame(1).record["inserted_count"], 5)
        self.assertEqual(fixture.frame(2).record["deleted_count"], 7)

    def test_original_float32_and_unique_ids_are_required(self):
        writer = self.writer()
        with self.assertRaisesRegex(ValueError, "original float32"):
            writer.append(self.points.astype(np.float64), self.ids)
        bad_ids = self.ids.copy()
        bad_ids[1] = bad_ids[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            writer.append(self.points, bad_ids)
        with self.assertRaisesRegex(ValueError, "active_count"):
            writer.append(self.points, self.ids, active_count=33)

    def test_restoration_marker_cannot_hide_changed_coordinates(self):
        writer = self.writer()
        writer.append(self.points, self.ids)
        altered = self.points.copy()
        altered[0, 0] = np.nextafter(altered[0, 0], np.float32(np.inf))
        writer.append(altered, self.ids, restored_from=0)
        writer.finalize()
        with self.assertRaisesRegex(ValueError, "restoration does not"):
            Fixture(writer.root).validate()

    def test_checksum_and_path_escape_are_rejected(self):
        fixture = self.fixture()
        path = fixture.root / fixture.records[0]["path"]
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            Fixture(fixture.root)
        document = fixture.document
        document["frames"][0]["path"] = "../outside.npz"
        fixture.manifest_path.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            Fixture(fixture.root)

    def test_registry_prune_reorder_insert_and_rollback(self):
        registry = SiteRegistry(4)
        original = registry.ids.copy()
        registry.select([3, 1])
        np.testing.assert_array_equal(registry.insert(2), [4, 5])
        registry.restore(original)
        np.testing.assert_array_equal(registry.insert(1), [6])
        with self.assertRaises(ValueError):
            registry.select([0, 0])

    def test_previous_connectivity_has_explicit_mapping_and_allows_faults(self):
        writer = self.writer()
        bad = np.array([[0, 0, -1, 99]], np.int64)
        with self.assertRaisesRegex(ValueError, "explicit site-ID"):
            writer.append(self.points, self.ids, previous_tets=bad)
        writer.append(self.points, self.ids, previous_tets=bad, previous_tet_site_ids=self.ids)
        writer.finalize()
        np.testing.assert_array_equal(Fixture(writer.root).frame(0).previous_tets, bad)

    def test_sequence_retains_state_and_pairs_reset_previous_state(self):
        fixture = self.fixture()
        instances = []

        class Observed(ScipyBackend):
            def __init__(self, name, **options):
                super().__init__(name, **options)
                self.calls = []
                instances.append(self)

            def update(self, prepared):
                self.calls.append(prepared["frame"].record["index"])
                super().update(prepared)

        with patch("benchmarks.replay.runner.create_backend", Observed):
            report = run(fixture, names=["scipy-full"], mode="sequence", repeats=2)
        self.assertEqual([instance.calls for instance in instances], [[0, 1, 2, 3]] * 3)
        for record in report["records"].values():
            summary = record["summary"]["scipy-full"]
            self.assertEqual(summary["independently_comparable_samples"], 2)
        instances.clear()
        with patch("benchmarks.replay.runner.create_backend", Observed):
            run(fixture, names=["scipy-full"], mode="pairs", repeats=1)
        self.assertEqual([instance.calls for instance in instances], [[0, 1, 1, 2, 2, 3]] * 2)

    def test_contracts_and_sequence_gaps_cannot_be_silently_mixed(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(ValueError, "separate replay"):
            run(fixture, names=["scipy-full", "warp-adjacency"])
        with self.assertRaisesRegex(ValueError, "every frame"):
            run(fixture, names=["scipy-full"], transitions=[1, 3])

    def test_controlled_corpus_preserves_expected_failure_records(self):
        path = synthetic(self.root / "control", count=32)
        fixture = Fixture(path.parent)
        fixture.validate()
        report = run(fixture, names=["scipy-full"], repeats=1)
        nonfinite = next(
            record
            for record in report["records"].values()
            if record["metadata"]["event"] == "nonfinite"
        )
        summary = nonfinite["summary"]["scipy-full"]
        self.assertEqual(summary["successful_samples"], 0)
        self.assertIsNone(summary["reference_validated_median_seconds"])
        self.assertTrue(summary["failures"])

    def test_legacy_radfoam_resets_unrecoverable_identity(self):
        source, metadata, report = [
            self.root / name for name in ["trajectory.npz", "trajectory.json", "report.json"]
        ]
        np.savez(source, points_0=self.points, points_1=self.points[::-1], points_2=self.points)
        metadata.write_text(
            json.dumps(
                [
                    dict(step=0, incremental=True),
                    dict(step=10, incremental=True),
                    dict(step=11, incremental=True),
                ]
            )
        )
        report.write_text(
            json.dumps(
                dict(
                    backend="warpower",
                    events=[dict(step=10, kind="densify_prune")],
                    source_sha256={},
                )
            )
        )
        path = export_radfoam(source, metadata, self.root / "radfoam", report=report)
        fixture = Fixture(path.parent)
        fixture.validate()
        self.assertTrue(fixture.frame(1).record["reset"])
        self.assertTrue(np.all(fixture.frame(1).rows_from_previous == -1))
        self.assertFalse(fixture.frame(2).record["reset"])
        report.write_text(json.dumps(dict(backend="native", events=[])))
        with self.assertRaisesRegex(ValueError, "no permutations"):
            export_radfoam(source, metadata, self.root / "invalid", report=report)

    def test_inspection_imports_neither_optimizer_nor_gpu_runtime(self):
        code = "import sys; import benchmarks.incremental_replay; assert 'torch' not in sys.modules; assert 'warp' not in sys.modules; assert 'radfoam' not in sys.modules; assert 'pycolmap' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_process_timeout_preserves_partial_evidence(self):
        path = self.root / "partial.json"
        path.write_text(json.dumps(dict(records=["previous completed frame"])))
        status = supervise(
            [sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05, output_path=path
        )
        self.assertEqual(status, 124)
        report = json.loads(path.read_text())
        self.assertFalse(report["complete"])
        self.assertEqual(report["records"], ["previous completed frame"])


if __name__ == "__main__":
    unittest.main()
