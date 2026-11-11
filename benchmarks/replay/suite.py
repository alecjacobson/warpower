"""Run a checksum-pinned representative suite, one GPU process at a time."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmarks.replay.fixture import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path, default=Path("experiments/geometry-updates/suite.json")
    )
    parser.add_argument("--root", type=Path, required=True, help="verified extracted corpus root")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument(
        "--backends", nargs="+", help="override only for cases with a matching contract"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--native-module-dir", type=Path)
    parser.add_argument("--cgal-binary", type=Path, required=True)
    parser.add_argument("--exact-checker", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())
    cases = [case for case in suite["cases"] if args.cases is None or case["name"] in args.cases]
    if not cases or (args.cases is not None and set(args.cases) != {c["name"] for c in cases}):
        parser.error("unknown or empty case selection")
    args.output.mkdir(parents=True, exist_ok=False)
    invocations = []
    for case in cases:
        fixture = Fixture(args.root / case["fixture"])
        if fixture.sha256 != case["manifest_sha256"]:
            raise ValueError(f"Corpus manifest differs: {case['name']}")
        bounded = case["contract"] == "bounded-voronoi-adjacency"
        names = case["backends"] if args.backends is None or bounded else args.backends
        command = [
            sys.executable,
            "-m",
            "benchmarks.incremental_replay",
            "run",
            str(fixture.root),
            "--mode",
            "pairs",
            "--transitions",
            *map(str, case["transitions"]),
            "--backends",
            *names,
            "--repeats",
            str(args.repeats),
            "--warmups",
            str(args.warmups),
            "--validate",
            "none" if bounded else "cgal",
            "--cgal-binary",
            str(args.cgal_binary),
            "--exact-checker",
            str(args.exact_checker),
            "--output",
            str(args.output / f"{case['name']}.json"),
        ]
        if args.native_module_dir is not None:
            command += ["--native-module-dir", str(args.native_module_dir)]
        with (args.output / f"{case['name']}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        invocations.append(dict(case=case["name"], command=command, exit_code=result.returncode))
        (args.output / "invocations.json").write_text(json.dumps(invocations, indent=2) + "\n")
        print(json.dumps(dict(case=case["name"], exit_code=result.returncode)), flush=True)
    raise SystemExit(int(any(row["exit_code"] for row in invocations)))


if __name__ == "__main__":
    main()
