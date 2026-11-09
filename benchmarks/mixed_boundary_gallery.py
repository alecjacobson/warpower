"""Complete gallery comparison for private mixed queries; no production API change."""

import hashlib
import json
import sys
from pathlib import Path


def main():
    mixed = "--mixed" in sys.argv
    if mixed:
        sys.argv.remove("--mixed")
        import warpower._vtm as core
        from benchmarks._mixed_boundary import MixedBoundaryWorkspace

        core.BoundaryWorkspace = MixedBoundaryWorkspace
    from examples import vtm_gallery

    try:
        vtm_gallery.main()
    finally:
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output") + 1]) / "result.json"
            if output.exists():
                report = json.loads(output.read_text())
                report["experimental_boundary_query"] = "mixed" if mixed else "reference"
                report["experimental_source_sha256"] = {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), Path("benchmarks/_mixed_boundary.py")]
                }
                output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
