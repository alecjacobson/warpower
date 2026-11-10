"""Versioned, checksummed float32 motion fixtures with explicit site identity.

This module intentionally imports neither Warp nor Torch. Producers may record
geometry from either optimizer; consumers only need NumPy to inspect a fixture.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

SCHEMA = "warpower.geometry-replay"
VERSION = 1


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _check_arrays(points, site_ids, active_count):
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be original float32 coordinates with shape (capacity, 3)")
    if site_ids.dtype != np.int64 or site_ids.shape != (len(points),):
        raise ValueError("site_ids must be int64 with one ID per stored point")
    if np.any(site_ids < 0) or len(np.unique(site_ids)) != len(site_ids):
        raise ValueError("site IDs must be unique and nonnegative")
    if type(active_count) is not int or not 0 <= active_count <= len(points):
        raise ValueError("active_count must select a valid prefix")


def correspondence(previous, current):
    """Current active rows -> previous active rows; -1 means a new site ID."""
    rows = np.full(len(current), -1, np.int64)
    _, old, new = np.intersect1d(previous, current, assume_unique=True, return_indices=True)
    rows[new] = old
    inserted = current[rows < 0]
    deleted = np.setdiff1d(previous, current, assume_unique=True)
    return rows, inserted, deleted


@dataclass
class Frame:
    points: np.ndarray
    site_ids: np.ndarray
    record: dict
    rows_from_previous: np.ndarray
    inserted_ids: np.ndarray
    deleted_ids: np.ndarray
    previous_tets: np.ndarray | None = None
    previous_tet_site_ids: np.ndarray | None = None

    @property
    def active_count(self):
        return self.record["active_count"]

    def canonical(self):
        """Normalize storage reordering without changing binary coordinates."""
        active = self.active_count
        order = np.argsort(self.site_ids[:active], kind="stable")
        return self.points[:active][order].copy(), self.site_ids[:active][order].copy()


class SiteRegistry:
    """Producer-side IDs that survive reorder/prune and are never recycled."""

    def __init__(self, count=0):
        self.ids = np.arange(count, dtype=np.int64)
        self.next_id = count

    def select(self, rows):
        selected = self.ids[np.asarray(rows)]
        if len(np.unique(selected)) != len(selected):
            raise ValueError("a selection may prune/reorder sites, but cannot clone an ID")
        self.ids = selected.copy()
        return self.ids.copy()

    def insert(self, count):
        if type(count) is not int or count < 0:
            raise ValueError("insertion count must be nonnegative")
        new = np.arange(self.next_id, self.next_id + count, dtype=np.int64)
        self.next_id += count
        self.ids = np.concatenate([self.ids, new])
        return new

    def restore(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= self.next_id) or len(np.unique(ids)) != len(ids):
            raise ValueError("restoration must use previously issued, distinct IDs")
        self.ids = ids.copy()


class Writer:
    """Append producer snapshots; finalize publishes the complete manifest last."""

    def __init__(self, root, *, name, provenance, limitations=()):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "frames").mkdir()
        self.document = dict(
            schema=SCHEMA,
            version=VERSION,
            name=name,
            coordinate_dtype="float32",
            identity="nonnegative int64 IDs, never recycled within an identity epoch",
            provenance=provenance,
            limitations=list(limitations),
            frames=[],
        )
        self.previous_ids = np.empty(0, np.int64)
        self.finished = False

    def append(
        self,
        points,
        site_ids,
        *,
        active_count=None,
        event="update",
        stage="unspecified",
        reset=False,
        restored_from=None,
        source_step=None,
        identity_epoch=0,
        previous_tets=None,
        previous_tet_site_ids=None,
        metadata=None,
    ):
        if self.finished:
            raise RuntimeError("fixture was already finalized")
        points, site_ids = np.asarray(points), np.asarray(site_ids)
        active_count = len(points) if active_count is None else active_count
        _check_arrays(points, site_ids, active_count)
        index = len(self.document["frames"])
        if restored_from is not None and not 0 <= restored_from < index:
            raise ValueError("restoration target must be an earlier frame")
        rows, inserted, deleted = correspondence(self.previous_ids, site_ids[:active_count])
        payload = dict(
            points=points,
            site_ids=site_ids,
            rows_from_previous=rows,
            inserted_ids=inserted,
            deleted_ids=deleted,
        )
        if (previous_tets is None) != (previous_tet_site_ids is None):
            raise ValueError("previous connectivity requires its explicit site-ID index mapping")
        if previous_tets is not None:
            previous_tets, previous_tet_site_ids = (
                np.asarray(previous_tets),
                np.asarray(previous_tet_site_ids),
            )
            if (
                previous_tets.dtype != np.int64
                or previous_tets.ndim != 2
                or previous_tets.shape[1] != 4
            ):
                raise ValueError("previous_tets must have int64 shape (cells, 4)")
            if previous_tet_site_ids.dtype != np.int64 or previous_tet_site_ids.ndim != 1:
                raise ValueError("previous_tet_site_ids must be a one-dimensional int64 mapping")
            # Index corruption is allowed deliberately: the consumer must reject
            # or recertify it, never index unchecked arrays with these values.
            payload.update(previous_tets=previous_tets, previous_tet_site_ids=previous_tet_site_ids)
        relative = f"frames/{index:06d}.npz"
        path = self.root / relative
        np.savez_compressed(path, **payload)
        record = dict(
            index=index,
            path=relative,
            bytes=path.stat().st_size,
            sha256=digest(path),
            capacity=len(points),
            active_count=active_count,
            event=event,
            stage=stage,
            reset=bool(reset),
            restored_from=restored_from,
            source_step=source_step,
            identity_epoch=int(identity_epoch),
            inserted_count=len(inserted),
            deleted_count=len(deleted),
            active_finite=bool(np.isfinite(points[:active_count]).all()),
            previous_connectivity=previous_tets is not None,
            metadata={} if metadata is None else metadata,
        )
        json.dumps(record, allow_nan=False)
        self.document["frames"].append(record)
        self.previous_ids = site_ids[:active_count].copy()
        return index

    def finalize(self):
        if not self.document["frames"]:
            raise ValueError("empty fixture")
        text = json.dumps(self.document, indent=2, allow_nan=False) + "\n"
        (self.root / "manifest.json").write_text(text)
        self.finished = True
        return self.root / "manifest.json"


class Fixture:
    def __init__(self, root, *, verify=True):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.document = json.loads(self.manifest_path.read_text())
        if self.document.get("schema") != SCHEMA or self.document.get("version") != VERSION:
            raise ValueError("unsupported geometry replay schema/version")
        self.records = self.document["frames"]
        if not self.records:
            raise ValueError("empty fixture")
        paths = set()
        for index, record in enumerate(self.records):
            relative = PurePosixPath(record["path"])
            if (
                record["index"] != index
                or relative.is_absolute()
                or ".." in relative.parts
                or relative.as_posix() != record["path"]
                or "\\" in record["path"]
                or record["path"] in paths
            ):
                raise ValueError("unsafe, duplicate, or misnumbered frame path")
            paths.add(record["path"])
            path = self.root / relative
            if path.is_symlink() or self.root not in path.resolve().parents:
                raise ValueError("frame must be a regular file inside the fixture")
            if verify and (
                path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]
            ):
                raise ValueError(f"frame checksum differs: {relative}")
        self.sha256 = digest(self.manifest_path)

    def __len__(self):
        return len(self.records)

    def frame(self, index):
        record = self.records[index]
        with np.load(self.root / record["path"], allow_pickle=False) as payload:
            required = ["points", "site_ids", "rows_from_previous", "inserted_ids", "deleted_ids"]
            arrays = {key: payload[key] for key in required}
            for key in ["previous_tets", "previous_tet_site_ids"]:
                arrays[key] = payload[key] if key in payload else None
        _check_arrays(arrays["points"], arrays["site_ids"], record["active_count"])
        if len(arrays["points"]) != record["capacity"]:
            raise ValueError("capacity metadata differs from payload")
        if (
            bool(np.isfinite(arrays["points"][: record["active_count"]]).all())
            != record["active_finite"]
        ):
            raise ValueError("finite-input metadata differs from payload")
        for key in ["rows_from_previous", "inserted_ids", "deleted_ids"]:
            if arrays[key].dtype != np.int64 or arrays[key].ndim != 1:
                raise ValueError(f"invalid identity mapping: {key}")
        return Frame(record=record, **arrays)

    def validate(self):
        previous = np.empty(0, np.int64)
        for index in range(len(self)):
            frame = self.frame(index)
            expected = correspondence(previous, frame.site_ids[: frame.active_count])
            actual = (frame.rows_from_previous, frame.inserted_ids, frame.deleted_ids)
            if any(not np.array_equal(a, b) for a, b in zip(expected, actual)):
                raise ValueError(f"site correspondence differs in frame {index}")
            if (
                len(frame.inserted_ids) != frame.record["inserted_count"]
                or len(frame.deleted_ids) != frame.record["deleted_count"]
            ):
                raise ValueError(f"insertion/deletion metadata differs in frame {index}")
            target = frame.record["restored_from"]
            if target is not None:
                if not 0 <= target < index:
                    raise ValueError("restoration target must be earlier")
                old_points, old_ids = self.frame(target).canonical()
                points, ids = frame.canonical()
                if not np.array_equal(ids, old_ids) or points.tobytes() != old_points.tobytes():
                    raise ValueError(
                        f"restoration does not reproduce original coordinates/IDs: {index}"
                    )
            previous = frame.site_ids[: frame.active_count]
        return dict(frames=len(self), manifest_sha256=self.sha256, correspondence_valid=True)


def motion_features(previous, current):
    """Offline explanatory features; never part of measured update latency."""
    from scipy.spatial import cKDTree

    a, ids_a = previous.canonical()
    b, ids_b = current.canonical()
    rows, inserted, deleted = correspondence(ids_a, ids_b)
    valid = rows >= 0
    result = dict(
        retained_sites=int(valid.sum()),
        inserted_sites=len(inserted),
        deleted_sites=len(deleted),
        identity_epoch_changed=previous.record["identity_epoch"]
        != current.record["identity_epoch"],
    )
    if not valid.any() or not np.isfinite(a).all() or not np.isfinite(b).all():
        return result | dict(finite_motion=False)
    retained = a[rows[valid]].astype(np.float64)
    displacement = np.linalg.norm(b[valid].astype(np.float64) - retained, axis=1)
    spacing = cKDTree(a.astype(np.float64)).query(retained, k=2)[0][:, 1]
    positive = np.isfinite(spacing) & (spacing > 0)
    ratio = displacement[positive] / spacing[positive]
    result.update(
        finite_motion=True,
        moved_fraction=float(np.mean(displacement > 0)),
        maximum_displacement=float(displacement.max()),
        median_displacement=float(np.median(displacement)),
        local_spacing_median=float(np.median(spacing[np.isfinite(spacing)]))
        if np.isfinite(spacing).any()
        else None,
        duplicate_spacing_sites=int((spacing == 0).sum()),
        missing_spacing_sites=int((~np.isfinite(spacing)).sum()),
        displacement_over_spacing_median=float(np.median(ratio)) if len(ratio) else None,
        displacement_over_spacing_p95=float(np.percentile(ratio, 95)) if len(ratio) else None,
    )
    return result
