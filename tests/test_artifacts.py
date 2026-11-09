import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

from benchmarks.artifacts import file_digest, members, pack, verify
from benchmarks.check_artifact_policy import LIMIT, check, violation


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = b"measured result\n"
        (self.root / "evidence.txt").write_bytes(self.data)
        self.row = dict(
            path="evidence.txt", bytes=len(self.data), sha256=hashlib.sha256(self.data).hexdigest()
        )
        self.inventory = dict(files=[self.row])
        self.archive, self.manifest = self.root / "bundle.tar.gz", self.root / "manifest.json"

    def test_round_trip_and_deterministic_archive(self):
        result = pack(self.root, self.inventory, self.archive, self.manifest)
        self.assertEqual(verify(self.archive, result)["verified_members"], 1)
        self.assertEqual(json.loads(self.manifest.read_text()), result)
        second = self.root / "second.tar.gz"
        pack(self.root, self.inventory, second, self.root / "second.json")
        self.assertEqual(self.archive.read_bytes(), second.read_bytes())

    def test_stale_source_does_not_create_bundle(self):
        (self.root / "evidence.txt").write_bytes(b"different bytes")
        with self.assertRaisesRegex(ValueError, "Source differs"):
            pack(self.root, self.inventory, self.archive, self.manifest)
        self.assertFalse(self.archive.exists())

    def test_bundle_corruption_is_detected(self):
        result = pack(self.root, self.inventory, self.archive, self.manifest)
        self.archive.write_bytes(self.archive.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "Archive size"):
            verify(self.archive, result)

    def test_unexpected_or_unsafe_members_are_rejected(self):
        for name in ["../escape", "/absolute", ".", "x/../escape", "a//b", "a\\b"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                members(dict(files=[dict(self.row, path=name)]))
        with self.assertRaises(ValueError):
            members(dict(files=[self.row, self.row]))
        with tarfile.open(self.archive, "w:gz") as tar:
            entry = tarfile.TarInfo("unexpected.txt")
            entry.size = len(self.data)
            tar.addfile(entry, io.BytesIO(self.data))
        document = dict(
            self.inventory,
            archive=dict(bytes=self.archive.stat().st_size, sha256=file_digest(self.archive)),
        )
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            verify(self.archive, document)

    def test_links_and_missing_members_are_rejected(self):
        with tarfile.open(self.archive, "w:gz") as tar:
            entry = tarfile.TarInfo(self.row["path"])
            entry.type, entry.linkname = tarfile.SYMTYPE, "elsewhere"
            tar.addfile(entry)
        document = dict(
            self.inventory,
            archive=dict(bytes=self.archive.stat().st_size, sha256=file_digest(self.archive)),
        )
        with self.assertRaisesRegex(ValueError, "non-regular"):
            verify(self.archive, document)
        with tarfile.open(self.archive, "w:gz"):
            pass
        document["archive"] = dict(
            bytes=self.archive.stat().st_size, sha256=file_digest(self.archive)
        )
        with self.assertRaisesRegex(ValueError, "Missing"):
            verify(self.archive, document)

    def test_existing_output_is_preserved(self):
        self.archive.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            pack(self.root, self.inventory, self.archive, self.manifest)
        self.assertEqual(self.archive.read_bytes(), b"existing")

    def test_forward_policy_keeps_small_fixtures_and_lfs_pointers(self):
        self.assertIsNone(violation("experiments/run/report.json", b"{}"))
        self.assertIsNone(violation("tests/data/tiny.npz", b"\0binary"))
        pointer = (
            b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"a" * 64 + b"\nsize 10\n"
        )
        self.assertIsNone(violation("data/cloud.npz", pointer))
        for name, data in [
            ("results/local/report.json", b"{}"),
            ("experiments/run/mesh.npz", b"anything"),
            ("tool", b"ELF\0"),
            ("tests/data/large.npz", b"x" * (LIMIT + 1)),
        ]:
            with self.subTest(name=name):
                self.assertIsNotNone(violation(name, data))

    def test_git_gate_checks_committed_changes_and_grandfathers_history(self):
        repository = self.root / "repository"
        repository.mkdir()

        def git(*args):
            return (
                subprocess.check_output(
                    [
                        "git",
                        "-C",
                        str(repository),
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        *args,
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )

        git("init", "-q")
        (repository / "old.npz").write_bytes(b"x" * (LIMIT + 1))
        git("add", "old.npz")
        git("commit", "-qm", "historical raw evidence")
        base = git("rev-parse", "HEAD")
        (repository / "report.json").write_text("{}")
        git("add", "report.json")
        git("commit", "-qm", "compact report")
        previous = Path.cwd()
        try:
            os.chdir(repository)
            self.assertEqual(check(base), [])
            (repository / "report.json").write_bytes(b"x" * (LIMIT + 1))
            self.assertEqual(check(base), [])  # Working data is not a committed blob.
            git("add", "report.json")
            git("commit", "-qm", "oversized report")
            self.assertEqual([name for name, _ in check(base)], ["report.json"])
        finally:
            os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
