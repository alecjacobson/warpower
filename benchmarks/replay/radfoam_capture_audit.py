"""CPU audit of producer identity operations and optional CGAL mesh comparisons."""

import argparse
import json
from pathlib import Path, PurePosixPath

import numpy as np

from benchmarks.replay.backends import canonical_rows
from benchmarks.replay.fixture import Fixture, correspondence, digest
from benchmarks.replay.runner import output_hash, reference, topology_churn


def payload(root, record):
    relative = PurePosixPath(record["path"])
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or path.is_symlink()
        or root not in path.resolve().parents
    ):
        raise ValueError("unsafe capture sidecar path")
    if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
        raise ValueError("capture sidecar checksum differs")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def audit(root, *, cgal_binary=None):
    root = Path(root).resolve()
    fixture = Fixture(root / "trajectory")
    fixture.validate()
    events = json.loads((root / "events.json").read_text())
    outputs = json.loads((root / "connectivity.json").read_text())
    if any(event["index"] != i for i, event in enumerate(events)):
        raise ValueError("producer events are misnumbered")
    initial = [event for event in events if event["kind"] == "initialize-site-ids"]
    if len(initial) != 1:
        raise ValueError("capture must initialize identity exactly once")
    ids = np.arange(initial[0]["count"], dtype=np.int64)
    next_id = len(ids)
    mutations = [event for event in events if event["kind"] in ["permutation", "insert", "prune"]]
    cursor = 0
    inserted, deleted = 0, 0

    def apply(event):
        nonlocal ids, next_id, inserted, deleted
        arrays = payload(root, event["payload"])
        if event["kind"] == "permutation":
            rows = arrays["permutation"]
            if not np.array_equal(np.sort(rows), np.arange(len(ids))):
                raise ValueError("recorded permutation is not a complete bijection")
            if not np.array_equal(ids, arrays["previous_ids"]):
                raise ValueError("permutation source identity differs")
            ids = ids[rows]
            if not np.array_equal(ids, arrays["current_ids"]):
                raise ValueError("permutation target identity differs")
        elif event["kind"] == "insert":
            added = arrays["new_ids"]
            if not np.array_equal(added, np.arange(next_id, next_id + len(added))):
                raise ValueError("inserted IDs are recycled or discontinuous")
            if len(added) != event["count"] or arrays["points"].shape != (len(added), 3):
                raise ValueError("insertion payload differs from recorded count")
            ids = np.concatenate([ids, added])
            next_id += len(added)
            inserted += len(added)
        else:
            mask = arrays["prune_mask"]
            if mask.dtype != np.bool_ or mask.shape != ids.shape:
                raise ValueError("prune mask does not match producer rows")
            if not np.array_equal(ids[mask], arrays["removed_ids"]) or mask.sum() != event["count"]:
                raise ValueError("recorded deleted IDs differ from the actual mask")
            deleted += int(mask.sum())
            ids = ids[~mask]

    if len({row["frame"] for row in outputs}) != len(outputs):
        raise ValueError("duplicate native output record")
    by_frame = {row["frame"]: row for row in outputs}
    previous_frame = previous_mesh = None
    rows = []
    for index in range(len(fixture)):
        frame = fixture.frame(index)
        for event_index in frame.record["metadata"]["producer_event_indices"]:
            if cursor >= len(mutations) or mutations[cursor]["index"] != event_index:
                raise ValueError("input does not consume producer mutations in order")
            apply(mutations[cursor])
            cursor += 1
        if not np.array_equal(frame.site_ids, ids):
            raise ValueError(f"frame {index} identity disagrees with actual producer operations")
        result = dict(
            frame=index,
            step=frame.record["source_step"],
            event=frame.record["event"],
            stage=frame.record["stage"],
            sites=len(ids),
            inserted=frame.record["inserted_count"],
            deleted=frame.record["deleted_count"],
            reset=frame.record["reset"],
            identity_chain_valid=True,
            input_sha256=frame.record["sha256"],
        )
        mesh = None
        native = by_frame.get(index)
        if native is None:
            result["native_result"] = "missing; interrupted before a native result was recorded"
        elif not native["success"]:
            result["native_result"] = native["error"]
        else:
            arrays = payload(root, native["payload"])
            output_ids, tets = arrays["site_ids"], arrays["tetrahedra"]
            if not np.array_equal(np.sort(output_ids), np.sort(ids)):
                raise ValueError("native connectivity mapping does not cover current input IDs")
            if tets.ndim != 2 or tets.shape[1] != 4 or np.any(tets < 0) or np.any(tets >= len(ids)):
                raise ValueError("native connectivity contains invalid indices")
            mesh = canonical_rows(output_ids[tets])
            if len(np.unique(mesh, axis=0)) != len(mesh) or np.any(np.diff(mesh, axis=1) == 0):
                raise ValueError("native connectivity contains duplicate/degenerate index tuples")
            result.update(
                tetrahedra=len(mesh),
                native_output_sha256=output_hash(mesh),
                native_result="success",
            )
            if cgal_binary is not None:
                expected, error, metadata = reference(frame, "cgal", cgal_binary)
                result.update(
                    cgal_matches=bool(np.array_equal(mesh, expected))
                    if expected is not None
                    else None,
                    cgal_error=error,
                    cgal_metadata=metadata,
                )
        if previous_frame is not None:
            a, old_ids = previous_frame.canonical()
            b, new_ids = frame.canonical()
            prior, born, died = correspondence(old_ids, new_ids)
            retained = prior >= 0
            displacement = b[retained].astype(np.float64) - a[prior[retained]].astype(np.float64)
            magnitude = np.linalg.norm(displacement, axis=1)
            result["displacement"] = dict(
                retained_sites=int(retained.sum()),
                inserted_sites=len(born),
                deleted_sites=len(died),
                moved_fraction=float(np.mean(magnitude > 0)),
                maximum=float(magnitude.max()),
                rms=float(np.sqrt(np.mean(magnitude**2))),
                scope="difference between original float32 positions for retained stable IDs",
            )
            result["native_topology_churn"] = (
                topology_churn(previous_mesh, mesh)
                if previous_mesh is not None and mesh is not None
                else None
            )
        rows.append(result)
        previous_frame, previous_mesh = frame, mesh
    # Initial or final ordering can occur after the last construction input.
    # Validate it, but never invent a subsequent geometry snapshot.
    trailing = len(mutations) - cursor
    for event in mutations[cursor:]:
        apply(event)
    return dict(
        fixture_manifest_sha256=fixture.sha256,
        frames=len(fixture),
        producer_identity_chain_valid=True,
        final_sites=len(ids),
        ids_ever_allocated=next_id,
        inserted_sites=inserted,
        deleted_sites=deleted,
        trailing_mutations=trailing,
        native_outputs=sum(r.get("native_result") == "success" for r in rows),
        cgal_comparisons=sum(r.get("cgal_matches") is not None for r in rows),
        cgal_matches=sum(r.get("cgal_matches") is True for r in rows),
        cgal_mismatches=sum(r.get("cgal_matches") is False for r in rows),
        cgal_binary_sha256=digest(cgal_binary) if cgal_binary is not None else None,
        scope="Checks recorded producer operation semantics and observed native connectivity; CGAL set equality is an optional independent comparison, not an added native runtime certificate.",
        records=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--cgal-binary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.capture, cgal_binary=args.cgal_binary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
