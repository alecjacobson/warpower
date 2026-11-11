"""Producer-side RadFoam identity tracking; inspection/tests do not import CUDA."""

import io
import json
from pathlib import Path

import numpy as np

from benchmarks.replay.fixture import Fixture, SiteRegistry, Writer, digest


def numpy_copy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def same_bits(actual, expected):
    actual, expected = numpy_copy(actual), np.asarray(expected)
    return (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.tobytes() == expected.tobytes()
    )


class CaptureBudgetExceeded(RuntimeError):
    pass


class RadFoamRecorder:
    """Record actual constructor/rebuild inputs and explicit producer mutations.

    Connectivity lives in sidecars: the normal replay fixture must not silently
    inject native cells into every backend or require native state injection.
    """

    def __init__(self, root, *, provenance, max_bytes=150 * 1024**2):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.writer = Writer(
            self.root / "trajectory",
            name="radfoam-bonsai-live-producer",
            provenance=provenance,
            limitations=[
                "Actual inputs from the explicitly reduced fitting recipe; not the official full-budget training workload.",
                "Native output sidecars are observed connectivity, not an independent exact/symbolic certificate.",
                "Every native constructor/rebuild attempt is recorded, including retry coordinates if retries occur. Calls that only attach adjacency/AABBs are events, not invented geometry updates.",
            ],
        )
        self.max_bytes = max_bytes
        self.payload_bytes = 0
        self.registry = None
        self.events = []
        self.outputs = []
        self.context = dict(step=None, stage="initialization", call=0, attempt=0)
        self.optimizer_updates = 0
        self.previous_optimizer_updates = 0
        self.pending_events = []
        self.pending_topology = False
        self.last_output = None
        self.finished = False

    def _reserve(self, size):
        # Reserve manifest/journal space without dropping an already captured input.
        if self.payload_bytes + size + 1024**2 > self.max_bytes:
            raise CaptureBudgetExceeded("capture byte budget reached; existing frames are retained")

    def _array_file(self, relative, **arrays):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        data = buffer.getvalue()
        self._reserve(len(data))
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.payload_bytes += len(data)
        return dict(path=relative, bytes=len(data), sha256=digest(path))

    def event(self, kind, **values):
        record = dict(index=len(self.events), kind=kind, step=self.context["step"], **values)
        self.events.append(record)
        return record["index"]

    def initialize(self, count):
        if self.registry is not None:
            raise RuntimeError("site identity may only be initialized once")
        self.registry = SiteRegistry(count)
        self.event("initialize-site-ids", count=count, first_id=0, next_id=count)

    def permute(self, permutation, *, source="producer.permute_points"):
        permutation = np.asarray(permutation)
        if permutation.dtype.kind not in "iu" or not np.array_equal(
            np.sort(permutation), np.arange(len(self.registry.ids))
        ):
            raise ValueError("producer permutation must be a complete row bijection")
        before = self.registry.ids.copy()
        self.registry.select(permutation)
        payload = self._array_file(
            f"events/{len(self.events):06d}.npz",
            permutation=permutation.astype(np.int64),
            previous_ids=before,
            current_ids=self.registry.ids,
        )
        self.pending_events.append(self.event("permutation", source=source, payload=payload))

    def insert(self, new_points):
        points = numpy_copy(new_points)
        if points.dtype != np.float32 or points.shape != (len(points), 3):
            raise ValueError("inserted positions must be original float32 coordinates")
        new_ids = self.registry.insert(len(points))
        payload = self._array_file(
            f"events/{len(self.events):06d}.npz", points=points, new_ids=new_ids
        )
        self.pending_events.append(self.event("insert", count=len(new_ids), payload=payload))
        self.pending_topology = True

    def prune(self, prune_mask):
        mask = np.asarray(prune_mask)
        if mask.dtype != np.bool_ or mask.shape != self.registry.ids.shape:
            raise ValueError("pruning requires one boolean decision per current site")
        removed = self.registry.ids[mask].copy()
        payload = self._array_file(
            f"events/{len(self.events):06d}.npz", prune_mask=mask, removed_ids=removed
        )
        self.registry.select(np.flatnonzero(~mask))
        self.pending_events.append(self.event("prune", count=len(removed), payload=payload))
        self.pending_topology = True

    def optimizer_step(self):
        self.optimizer_updates += 1
        self.event("optimizer-step-completed", optimizer_updates=self.optimizer_updates)

    def begin_geometry(self, *, step, stage, rebuild, incremental):
        self.context = dict(step=step, stage=stage, call=self.context["call"] + 1, attempt=0)
        self.event(
            "geometry-refresh-call",
            stage=stage,
            rebuild=bool(rebuild),
            incremental=bool(incremental),
        )

    def input(self, points, *, incremental, constructor=False):
        points = numpy_copy(points)
        ids = self.registry.ids.copy()
        if points.shape != (len(ids), 3) or points.dtype != np.float32:
            raise ValueError("native input does not match the tracked producer identity")
        self._reserve(points.nbytes + 5 * ids.nbytes + 8192)
        attempt = self.context["attempt"]
        event = (
            "initial-native-construction"
            if constructor
            else "native-retry"
            if attempt
            else "prune-densify-refresh"
            if self.pending_topology
            else "optimizer-refresh"
            if incremental
            else "full-refresh"
        )
        index = self.writer.append(
            points,
            ids,
            event=event,
            stage=self.context["stage"],
            reset=constructor or not incremental,
            source_step=self.context["step"],
            metadata=dict(
                geometry_call=self.context["call"],
                native_attempt=attempt,
                incremental=bool(incremental),
                optimizer_updates=self.optimizer_updates,
                optimizer_updates_since_previous_input=self.optimizer_updates
                - self.previous_optimizer_updates,
                producer_event_indices=self.pending_events.copy(),
                previous_native_output=self.last_output,
                identity_basis="producer-tracked permutation, exact prune mask, and monotonic insertion IDs",
            ),
        )
        self.payload_bytes += self.writer.document["frames"][-1]["bytes"]
        self.context["attempt"] += 1
        self.previous_optimizer_updates = self.optimizer_updates
        self.pending_events.clear()
        self.pending_topology = False
        self._journal()
        return index, ids

    def success(self, index, input_ids, tetrahedra, permutation=None):
        tets = numpy_copy(tetrahedra).astype(np.int64)
        ids = np.asarray(input_ids, dtype=np.int64)
        if permutation is not None:
            permutation = numpy_copy(permutation).astype(np.int64)
            if not np.array_equal(np.sort(permutation), np.arange(len(ids))):
                raise ValueError("native output permutation is not a bijection")
            ids = ids[permutation]
        if tets.ndim != 2 or tets.shape[1] != 4 or np.any(tets < 0) or np.any(tets >= len(ids)):
            raise ValueError("native output has invalid tetrahedron indices")
        payload = self._array_file(f"connectivity/{index:06d}.npz", tetrahedra=tets, site_ids=ids)
        record = dict(frame=index, success=True, tetrahedra=len(tets), payload=payload)
        self.outputs.append(record)
        self.last_output = record
        self._journal()

    def failure(self, index, error):
        self.outputs.append(
            dict(frame=index, success=False, error=f"{type(error).__name__}: {error}")
        )
        # The failed native call may have changed internal state. Do not pretend
        # the previously observed successful mesh describes that retained state.
        self.last_output = None
        self._journal()

    def _journal(self):
        (self.root / "capture-progress.json").write_text(
            json.dumps(
                dict(
                    frames=len(self.writer.document["frames"]),
                    events=len(self.events),
                    outputs=self.outputs,
                    payload_bytes=self.payload_bytes,
                ),
                indent=2,
            )
            + "\n"
        )

    def finish(self):
        if self.finished:
            return
        self.writer.finalize()
        (self.root / "events.json").write_text(json.dumps(self.events, indent=2) + "\n")
        (self.root / "connectivity.json").write_text(json.dumps(self.outputs, indent=2) + "\n")
        Fixture(self.writer.root).validate()
        self.finished = True


def instrument_scene(base, recorder):
    """Mirror actual upstream identity operations and verify their row semantics."""

    class CapturedIdentityScene(base):
        def permute_points(self, permutation):
            rows = numpy_copy(permutation)
            before = numpy_copy(self.primal_points)
            result = super().permute_points(permutation)
            if not same_bits(self.primal_points, before[rows]):
                raise RuntimeError("producer permutation changed coordinate values")
            recorder.permute(rows)
            return result

        def densification_postfix(self, new_params):
            before = numpy_copy(self.primal_points)
            added = numpy_copy(new_params["primal_points"])
            result = super().densification_postfix(new_params)
            if not same_bits(self.primal_points, np.concatenate([before, added])):
                raise RuntimeError("producer insertion did not append the recorded coordinates")
            recorder.insert(added)
            return result

        def prune_points(self, prune_mask):
            mask = numpy_copy(prune_mask)
            before = numpy_copy(self.primal_points)
            result = super().prune_points(prune_mask)
            if not same_bits(self.primal_points, before[~mask]):
                raise RuntimeError("producer pruning did not preserve the recorded surviving rows")
            recorder.prune(mask)
            return result

    return CapturedIdentityScene
