"""Export existing captures without loading either optimizer or any image data."""

import itertools
import json
from pathlib import Path

import numpy as np

from benchmarks.replay.fixture import SiteRegistry, Writer, digest


def export_vtm(source, output, *, record=None):
    source = Path(source)
    frames = np.load(source, mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.float32 or frames.ndim != 3 or frames.shape[2] != 3:
        raise ValueError("legacy VTM input must be float32 (frames, sites, 3)")
    provenance = dict(
        source=str(source),
        source_sha256=digest(source),
        producer="benchmarks.neighbor_clipping_trials.record_trials",
        identity_basis="The producer records the fixed full-budget site-array slots. It does not compact or reorder those slots during the captured reconstruction trials.",
    )
    if record is not None:
        provenance["producer_record"] = json.loads(Path(record).read_text())
        provenance["producer_record_sha256"] = digest(record)
    writer = Writer(
        output,
        name="legacy-vtm-armadillo",
        provenance=provenance,
        limitations=[
            "Legacy producer records full-budget reconstruction calls, including rejected trials, but does not label initialization/growth/refit/quality or acceptance decisions. Such phase labels are not reconstructed from motion.",
            "Exact repeated snapshots can be identified as coordinate restoration; this alone does not establish the optimizer's acceptance/rejection cause.",
        ],
    )
    ids = np.arange(frames.shape[1], dtype=np.int64)
    previous_hashes = {}
    for index, points in enumerate(frames):
        import hashlib

        point_hash = hashlib.sha256(points.tobytes()).hexdigest()
        restored = previous_hashes.get(point_hash)
        # Consecutive identical inputs are an unchanged update, not evidence of
        # restoration following a rejected proposal.
        restoration = restored if restored is not None and restored < index - 1 else None
        writer.append(
            points,
            ids,
            event="initial"
            if index == 0
            else "restoration-observed"
            if restoration is not None
            else "legacy-reconstruction",
            stage="unclassified-vtm-reconstruction",
            reset=index == 0,
            restored_from=restoration,
            source_step=index,
        )
        previous_hashes[point_hash] = index
    return writer.finalize()


def export_radfoam(source, metadata, output, *, report=None):
    source, metadata = Path(source), Path(metadata)
    events = json.loads(metadata.read_text())
    provenance = dict(
        source=str(source),
        source_sha256=digest(source),
        metadata_sha256=digest(metadata),
        producer="experiments/radfoam-integration/fit_scene.py",
        identity_basis="Original-order Warpower capture only. IDs are assigned within same-size incremental segments; full-refresh/count-change boundaries conservatively begin a new identity epoch.",
    )
    if report is not None:
        evidence = json.loads(Path(report).read_text())
        if evidence["backend"] not in ["warpower", "warpower-hints"]:
            raise ValueError(
                "legacy capture has no permutations; only original-order Warpower runs are supported"
            )
        provenance["run_backend"] = evidence["backend"]
        provenance["producer_revision"] = evidence.get("warpower_commit")
        provenance["producer_source_sha256"] = evidence.get("source_sha256")
        provenance["run_report_sha256"] = digest(report)
        topology_steps = [
            event["step"] for event in evidence["events"] if event["kind"] == "densify_prune"
        ]
    else:
        raise ValueError(
            "the original run report is required to establish backend/order provenance"
        )
    writer = Writer(
        output,
        name="legacy-radfoam-bonsai",
        provenance=provenance,
        limitations=[
            "Legacy snapshots omit persistent IDs and explicit prune/densify maps. Cross-boundary identity is unknown: replay resets instead of treating an index permutation as motion.",
            "Snapshots skip some geometry refreshes. Continuous replay follows the retained snapshots, not every original optimizer update.",
            "True insertion/deletion and rejected-proposal labels require new producer-side recording with Writer and SiteRegistry.",
        ],
    )
    next_id, epoch, previous_count, previous_step = 0, -1, None, -1
    with np.load(source, allow_pickle=False) as archive:
        for index, event in enumerate(events):
            points = archive[f"points_{index}"]
            step = event["step"]
            topology_between = any(
                previous_step < transition <= step for transition in topology_steps
            )
            reset = (
                index == 0
                or len(points) != previous_count
                or not event["incremental"]
                or topology_between
            )
            if reset:
                ids = np.arange(next_id, next_id + len(points), dtype=np.int64)
                next_id += len(points)
                epoch += 1
            writer.append(
                points,
                ids,
                event="identity-reset" if reset else "position-update",
                stage="radfoam-geometry-refresh",
                reset=reset,
                identity_epoch=epoch,
                source_step=step,
                metadata=dict(
                    incremental_requested=event["incremental"],
                    identity_reset_is_not_a_claim_of_all_site_births=reset,
                    topology_event_between_snapshots=topology_between,
                ),
            )
            previous_count, previous_step = len(points), step
    return writer.finalize()


def synthetic(output, *, count=256, seed=2026):
    if count < 32:
        raise ValueError("synthetic base corpus requires at least 32 sites")
    rng = np.random.default_rng(seed)
    points = rng.uniform(-1, 1, (count, 3)).astype(np.float32)
    registry = SiteRegistry(count)
    original_ids = registry.ids.copy()
    writer = Writer(
        output,
        name="controlled-motion-and-adversarial",
        provenance=dict(
            seed=seed, initial_sites=count, generator="benchmarks.replay.exporters.synthetic"
        ),
        limitations=[
            "Controlled cases extend the real-capture corpus; they do not replace real VTM/RadFoam trajectories.",
            "Capacity exhaustion is selected separately by the replay CLI's tetrahedra-capacity option.",
        ],
    )
    initial = writer.append(points, registry.ids, event="initial", reset=True, stage="uniform")
    writer.append(points, registry.ids, event="unchanged", stage="uniform")
    perm = rng.permutation(count)
    registry.select(perm)
    reordered = points[perm]
    writer.append(reordered, registry.ids, event="reorder", stage="uniform")
    moved = reordered.copy()
    moved += rng.normal(0, 0.005, moved.shape).astype(np.float32)
    writer.append(moved, registry.ids, event="global-motion", stage="uniform")
    local = moved.copy()
    nearest = np.argsort(np.linalg.norm(local, axis=1))[: max(1, count // 20)]
    local[nearest] += np.float32(0.03)
    writer.append(local, registry.ids, event="local-motion", stage="uniform")
    registry.restore(original_ids)
    writer.append(
        points,
        registry.ids,
        event="restore",
        stage="rejected-proposal-control",
        restored_from=initial,
    )
    extra = rng.uniform(-1, 1, (16, 3)).astype(np.float32)
    registry.insert(len(extra))
    grown = np.concatenate([points, extra])
    writer.append(grown, registry.ids, event="insert", stage="count-change")
    keep = np.arange(len(grown))[4:]
    registry.select(keep)
    pruned = grown[keep]
    pruned_index = writer.append(pruned, registry.ids, event="delete", stage="count-change")
    duplicates = pruned.copy()
    duplicates[1] = duplicates[0]
    writer.append(duplicates, registry.ids, event="duplicate-coordinates", stage="adversarial")
    writer.append(
        pruned, registry.ids, event="restore", stage="adversarial", restored_from=pruned_index
    )
    nonfinite = pruned.copy()
    nonfinite[0, 0] = np.nan
    writer.append(nonfinite, registry.ids, event="nonfinite", stage="adversarial")
    writer.append(
        pruned, registry.ids, event="restore", stage="adversarial", restored_from=pruned_index
    )
    # Every integer coordinate below has squared radius 50 exactly in float32.
    sphere = np.array(
        [
            np.array(p) * s
            for p in itertools.permutations([3, 4, 5])
            for s in itertools.product([-1, 1], repeat=3)
        ],
        dtype=np.float32,
    )
    new_ids = np.arange(registry.next_id, registry.next_id + len(sphere), dtype=np.int64)
    writer.append(sphere, new_ids, event="exact-cospherical", stage="adversarial", reset=True)
    near = sphere.copy()
    near[0, 0] = np.nextafter(near[0, 0], np.float32(np.inf))
    writer.append(near, new_ids, event="one-ulp-perturbation", stage="adversarial")
    thin = rng.uniform(-1, 1, (count, 3)).astype(np.float32)
    thin[:, 2] *= np.float32(1e-6)
    thin_ids = np.arange(new_ids[-1] + 1, new_ids[-1] + 1 + count, dtype=np.int64)
    thin_index = writer.append(thin, thin_ids, event="thin", stage="anisotropic", reset=True)
    # Deliberately corrupt retained indices. Fresh baselines may ignore this;
    # incremental adapters must consume safely or report unsupported injection.
    bad_tets = np.array([[0, 1, 2, count + 20], [0, 0, 1, 2], [-1, 1, 2, 3]], np.int64)
    writer.append(
        thin,
        thin_ids,
        event="corrupted-retained-state",
        stage="adversarial",
        previous_tets=bad_tets,
        previous_tet_site_ids=thin_ids,
    )
    planar = thin.copy()
    planar[:, 2] = 0
    writer.append(planar, thin_ids, event="coplanar", stage="adversarial")
    writer.append(thin, thin_ids, event="restore", stage="adversarial", restored_from=thin_index)
    return writer.finalize()
