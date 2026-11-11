"""Publish compact reproducible pairs, including exact reference initial meshes."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.replay.fixture import Fixture, Writer, digest
from benchmarks.replay.runner import reference


def package(source, transitions, output, cgal_binary):
    fixture = Fixture(source)
    fixture.validate()
    if not transitions or any(i < 1 or i >= len(fixture) for i in transitions):
        raise ValueError("select valid, noninitial transitions")
    writer = Writer(
        output,
        name=fixture.document["name"] + "-selected-pairs",
        provenance=dict(
            original_manifest_sha256=fixture.sha256,
            original_provenance=fixture.document["provenance"],
            original_transitions=transitions,
            cgal_binary_sha256=digest(cgal_binary),
            initial_meshes=[],
        ),
        limitations=[
            *fixture.document["limitations"],
            "Independent pairs; pair starts deliberately reset retained state and identity epoch.",
            "Initial meshes are recomputed by pinned CGAL on the exact producer coordinates; geometry timing excludes this preparation.",
        ],
    )
    for pair, index in enumerate(transitions):
        previous = fixture.frame(index - 1)
        points, ids = previous.canonical()
        stable_tets, error, info = reference(
            previous, "cgal", cgal_binary, preserve_orientation=True
        )
        if error:
            raise RuntimeError(f"Cannot construct initial mesh for {index}: {error}")
        local_tets = np.searchsorted(ids, stable_tets).astype(np.int32)
        name = f"initial-{pair:03d}.npz"
        np.savez_compressed(output / name, points=points, site_ids=ids, tetrahedra=local_tets)
        writer.document["provenance"]["initial_meshes"].append(
            dict(
                path=name,
                sha256=digest(output / name),
                original_frame=index - 1,
                pair_frame=2 * pair,
                reference=info,
            )
        )
        for offset, original in enumerate([index - 1, index]):
            frame = fixture.frame(original)
            record = frame.record
            writer.append(
                frame.points,
                frame.site_ids,
                active_count=frame.active_count,
                event="pair-start" if offset == 0 else record["event"],
                stage=record["stage"],
                reset=True if offset == 0 else record["reset"],
                identity_epoch=pair,
                source_step=record["source_step"],
                metadata=dict(
                    original_frame=original,
                    original_event=record["event"],
                    original_restored_from=record["restored_from"],
                    original_restore_target_sha256=(
                        fixture.records[record["restored_from"]]["sha256"]
                        if record["restored_from"] is not None
                        else None
                    ),
                    original_metadata=record["metadata"],
                    original_metadata_scope="Paths and event indices refer to the pinned original capture, not this pair package",
                ),
            )
    writer.finalize()
    return dict(
        fixture=output.name,
        transitions=list(range(1, 2 * len(transitions), 2)),
        validation=Fixture(output).validate(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--transitions", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cgal-binary", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(package(args.fixture, args.transitions, args.output, args.cgal_binary), indent=2)
    )


if __name__ == "__main__":
    main()
