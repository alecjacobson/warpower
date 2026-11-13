"""Optimizer-independent VTM/RadFoam geometry replay (issue #297).

Examples:
  python -m benchmarks.incremental_replay synthetic --output results/local/replay/control
  python -m benchmarks.incremental_replay inspect results/local/replay/control
  python -m benchmarks.incremental_replay run results/local/replay/control --backends scipy-full --output results/local/replay/scipy.json

GPU and native reference imports are deferred until their backend is requested.
"""

import argparse
import json
import sys
from pathlib import Path

from benchmarks.replay.backends import CONTRACTS
from benchmarks.replay.fixture import Fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser(
        "synthetic",
        help="controlled motion, reorder, restore, insertion/deletion, and adversarial cases",
    )
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--count", type=int, default=256)
    synthetic.add_argument("--seed", type=int, default=2026)
    vtm = commands.add_parser(
        "export-vtm", help="convert original fixed-slot full-budget VTM captures"
    )
    vtm.add_argument("--source", type=Path, required=True)
    vtm.add_argument("--record", type=Path)
    vtm.add_argument("--output", type=Path, required=True)
    radfoam = commands.add_parser(
        "export-radfoam",
        help="convert original-order legacy captures, with conservative identity resets",
    )
    radfoam.add_argument("--source", type=Path, required=True)
    radfoam.add_argument("--metadata", type=Path, required=True)
    radfoam.add_argument("--report", type=Path, required=True)
    radfoam.add_argument("--output", type=Path, required=True)
    inspect = commands.add_parser(
        "inspect", help="validate checksums, identity maps, and exact restoration without CUDA"
    )
    inspect.add_argument("fixture", type=Path)
    run = commands.add_parser(
        "run", help="repeat independent pairs or full persistent-state trajectories"
    )
    run.add_argument("fixture", type=Path)
    run.add_argument("--backends", nargs="+", choices=list(CONTRACTS), required=True)
    run.add_argument("--mode", choices=["pairs", "sequence"], default="sequence")
    run.add_argument("--transitions", nargs="+", type=int)
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--warmups", type=int, default=1)
    run.add_argument("--seed", type=int, default=2026)
    run.add_argument("--validate", choices=["scipy", "cgal", "none"], default="scipy")
    run.add_argument("--cgal-binary", type=Path)
    run.add_argument(
        "--exact-checker",
        type=Path,
        help="existing check-delaunay executable for CGAL-set mismatches",
    )
    run.add_argument("--native-module-dir", type=Path)
    run.add_argument("--native-module", default="radfoam_headless")
    run.add_argument("--tetrahedra-capacity", type=int)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--timeout-seconds",
        type=float,
        default=900,
        help="total worker wall-clock limit, including JIT; 0 disables supervision",
    )
    args = parser.parse_args()
    if args.command == "inspect":
        fixture = Fixture(args.fixture)
        print(
            json.dumps(
                fixture.validate()
                | dict(name=fixture.document["name"], limitations=fixture.document["limitations"]),
                indent=2,
            )
        )
    elif args.command == "run":
        from benchmarks.replay.runner import run, supervise

        if args.timeout_seconds > 0:
            command = [
                sys.executable,
                "-m",
                "benchmarks.incremental_replay",
                *sys.argv[1:],
                "--timeout-seconds",
                "0",
            ]
            raise SystemExit(
                supervise(command, timeout=args.timeout_seconds, output_path=args.output)
            )

        report = run(
            Fixture(args.fixture),
            names=args.backends,
            mode=args.mode,
            transitions=args.transitions,
            repeats=args.repeats,
            warmups=args.warmups,
            seed=args.seed,
            validate=args.validate,
            cgal_binary=args.cgal_binary,
            exact_checker=args.exact_checker,
            options=dict(
                native_module_dir=args.native_module_dir,
                native_module=args.native_module,
                tetrahedra_capacity=args.tetrahedra_capacity,
            ),
            output_path=args.output,
        )
        print(
            json.dumps(
                dict(
                    complete=report["complete"],
                    contract=report["contract"],
                    transitions=len(report["records"]),
                    output=str(args.output),
                )
            )
        )
    else:
        from benchmarks.replay import exporters

        if args.command == "synthetic":
            path = exporters.synthetic(args.output, count=args.count, seed=args.seed)
        elif args.command == "export-vtm":
            path = exporters.export_vtm(args.source, args.output, record=args.record)
        else:
            path = exporters.export_radfoam(
                args.source, args.metadata, args.output, report=args.report
            )
        print(json.dumps(dict(manifest=str(path), **Fixture(path.parent).validate())))


if __name__ == "__main__":
    main()
