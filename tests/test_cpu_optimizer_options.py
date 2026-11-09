import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.cpu_mesher import (
    add_optimizer_arguments,
    check_optimizer_report,
    optimizer_arguments,
)


class CPUOptimizerOptions(unittest.TestCase):
    def options(self, *argv):
        parser = argparse.ArgumentParser()
        add_optimizer_arguments(parser)
        return parser.parse_args(argv)

    def test_default_retains_historical_native_command_and_report(self):
        args = self.options()
        self.assertEqual(optimizer_arguments(args), [])
        check_optimizer_report(args, {})

    def test_order_and_enabled_report_are_checked(self):
        args = self.options("--perturb", "--exude", "--exude-bound", "15")
        self.assertEqual(optimizer_arguments(args), ["1", "1.0", "10.0", "1", "1.0", "15.0"])
        report = dict(triangulation_kind="weighted_regular")
        for name in ("perturb", "exude"):
            report.update(
                {
                    name + "_enabled": 1,
                    name + "_time_limit": 1.0,
                    name + "_sliver_bound": getattr(args, name + "_bound"),
                    name + "_return_code": 0,
                }
            )
        check_optimizer_report(args, report)
        for key in report:
            broken = report.copy()
            del broken[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_optimizer_report(args, broken)

    def test_bad_limits_are_rejected(self):
        for name in ("perturb", "exude"):
            for value in ("-1", "nan", "inf"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    optimizer_arguments(self.options("--" + name + "-seconds", value))
            for value in ("-1", "181", "nan", "inf"):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    optimizer_arguments(self.options("--" + name + "-bound", value))

    def test_timeout_preserves_partial_process_output_and_failed_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            surface = root / "tet.off"
            surface.write_text(
                "OFF\n4 4 0\n0 0 0\n1 0 0\n0 1 0\n0 0 1\n3 0 2 1\n3 0 1 3\n3 0 3 2\n3 1 2 3\n"
            )
            binary = root / "slow-native"
            binary.write_text(
                "#!"
                + sys.executable
                + "\nimport time\nprint('partial native result', flush=True)\ntime.sleep(10)\n"
            )
            binary.chmod(0o700)
            output = root / "result"
            command = [
                sys.executable,
                "-m",
                "benchmarks.cpu_mesher",
                "--binary",
                str(binary),
                "--surface",
                str(surface),
                "--size",
                "1",
                "--facet-distance",
                ".1",
                "--timeout",
                ".25",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            report = json.loads((output / "result.json").read_text())
            self.assertFalse(report["valid"])
            self.assertEqual(report["successful_samples"], 0)
            self.assertIsNone(report["median_complete_seconds"])
            self.assertTrue(report["runs"][0]["timed_out"])
            self.assertEqual(report["runs"][0]["exit_code"], 124)
            self.assertIn("partial native result", (output / "run-0.stdout").read_text())


if __name__ == "__main__":
    unittest.main()
