"""Lazy optional geometry adapters. Importing this module does not initialize CUDA."""

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONTRACTS = {
    "scipy-full": "full-hull-tetrahedra",
    "vtm-fragment": "full-hull-tetrahedra",
    "vtm-hull-cache": "full-hull-tetrahedra",
    "vtm-split-search": "full-hull-tetrahedra",
    "vtm-combined": "full-hull-tetrahedra",
    "warp-full": "full-hull-tetrahedra",
    "radfoam-incremental": "full-hull-tetrahedra",
    "radfoam-full": "full-hull-tetrahedra",
    "warp-adjacency": "bounded-voronoi-adjacency",
    "warp-hints": "bounded-voronoi-adjacency",
}


def canonical_rows(rows):
    rows = np.sort(np.asarray(rows, dtype=np.int64), axis=1)
    return rows[np.lexsort(rows.T[::-1])] if len(rows) else rows


@dataclass
class Output:
    rows: np.ndarray
    error: str | None
    diagnostics: dict


class UnsupportedState(RuntimeError):
    pass


class Backend:
    def __init__(self, name, **options):
        self.name = name
        self.contract = CONTRACTS[name]
        self.ids = None
        self.error = None
        self.diagnostics = {}
        self.options = options

    def synchronize(self):
        pass

    def prepare(self, frame):
        points, ids = frame.canonical()
        return dict(frame=frame, points=points, ids=ids)

    def state_changed(self, prepared):
        return self.ids is None or not np.array_equal(self.ids, prepared["ids"])

    def close(self):
        pass


class ScipyBackend(Backend):
    def update(self, prepared):
        from scipy.spatial import Delaunay, QhullError

        self.ids = prepared["ids"]
        self.error = None
        try:
            self.tets = Delaunay(prepared["points"].astype(np.float64)).simplices
        except (QhullError, ValueError) as error:
            self.error = f"{type(error).__name__}: {str(error).splitlines()[0]}"
            self.tets = np.empty((0, 4), np.int64)
        self.diagnostics = dict(strategy="fresh", previous_connectivity="ignored by fresh baseline")

    def readback(self):
        return Output(canonical_rows(self.ids[self.tets]), self.error, self.diagnostics)


class WarpBackend(Backend):
    def __init__(self, name, **options):
        super().__init__(name, **options)
        import warp as wp

        self.wp = wp
        wp.init()
        wp.set_device("cuda:0")
        wp.set_mempool_release_threshold("cuda:0", options.get("retain_mb", 512) * 1024**2)
        self.workspace = None
        self.diagram = None

    def synchronize(self):
        self.wp.synchronize_device("cuda:0")

    def prepare(self, frame):
        prepared = super().prepare(frame)
        prepared["device_points"] = self.wp.array(
            prepared["points"], dtype=self.wp.vec3, device="cuda:0"
        )
        if frame.previous_tets is not None:
            indices = frame.previous_tets
            source_ids = frame.previous_tet_site_ids
            valid = (indices >= 0) & (indices < len(source_ids))
            stable = np.full(indices.shape, -1, np.int64)
            stable[valid] = source_ids[indices[valid]]
            row_indices = np.searchsorted(prepared["ids"], stable)
            in_bounds = row_indices < len(prepared["ids"])
            found = np.zeros_like(valid)
            found[in_bounds] = prepared["ids"][row_indices[in_bounds]] == stable[in_bounds]
            row_indices[~(valid & found)] = -1
            prepared["seed_rows"] = row_indices.astype(np.int32)
        return prepared

    def update(self, prepared):
        wp = self.wp
        points, ids, frame = prepared["device_points"], prepared["ids"], prepared["frame"]
        changed = self.state_changed(prepared)
        explicit_reset = frame.record["reset"]
        self.error = None
        self.diagnostics = dict(
            identity_set_changed=changed,
            explicit_reset=explicit_reset,
            restoration=frame.record["restored_from"],
        )
        if self.contract == "bounded-voronoi-adjacency":
            import warpower

            kwargs = {}
            hints = (
                self.name == "warp-hints"
                and self.diagram is not None
                and not changed
                and not explicit_reset
            )
            if hints:
                kwargs = dict(
                    initial_guesses=self.diagram.adjacency,
                    initial_guesses_offsets=self.diagram.offsets,
                )
            self.diagram = warpower.voronoi_diagram(points, **kwargs)
            self.diagnostics.update(strategy="previous-neighbor-hints" if hints else "fresh")
            self.ids = ids
            return
        if self.workspace is None or len(ids) != self.workspace.n:
            from warpower._delaunay_recovery import DelaunayWorkspace
            from warpower._wall_reuse import WallReuseDelaunayWorkspace

            cls = WallReuseDelaunayWorkspace if self.name.startswith("vtm-") else DelaunayWorkspace
            kwargs = {}
            if self.options.get("tetrahedra_capacity") is not None:
                kwargs["tetrahedra_capacity"] = self.options["tetrahedra_capacity"]
            if self.name == "warp-full":
                kwargs["cache_strategy"] = False
            # The workspace borrows its input array and later updates it in place.
            # Own that storage so a prepared previous input remains replayable.
            self.workspace = cls(wp.clone(points), **kwargs)
            if self.name in ("vtm-hull-cache", "vtm-combined"):
                from .hull_cache_probe import install

                install(self.workspace)
            if self.name in ("vtm-split-search", "vtm-combined"):
                from .split_search_probe import install

                install(self.workspace)
            self.diagnostics["workspace_reallocated"] = True
            changed = True
        else:
            wp.copy(self.workspace.points, points)
            self.diagnostics["workspace_reallocated"] = False
        if self.name.startswith("vtm-"):
            if changed or explicit_reset:
                self.workspace.clear_reuse()
            if "seed_rows" in prepared:
                rows = prepared["seed_rows"]
                if len(rows) > len(self.workspace._previous):
                    raise UnsupportedState(
                        "injected retained cells exceed the workspace's retained buffer"
                    )
                device_rows = wp.array(rows, dtype=wp.int32, device="cuda:0")
                wp.copy(self.workspace._previous, device_rows, count=rows.size)
                self.workspace._previous_count.fill_(len(rows))
                self.diagnostics["previous_connectivity"] = "explicit retained-state injection"
        elif frame.previous_tets is not None:
            self.diagnostics["previous_connectivity"] = "ignored by fresh baseline"
        self.workspace.run()
        self.ids = ids

    def readback(self):
        if self.contract == "bounded-voronoi-adjacency":
            offsets = self.diagram.offsets.numpy()
            adjacency = self.diagram.adjacency.numpy()
            status = self.diagram.status.numpy()
            self.error = (
                None
                if np.all(status == 0)
                else f"nonzero cell statuses: {np.unique(status).tolist()}"
            )
            rows = np.empty((0, 2), np.int64)
            if self.error is None:
                source = np.repeat(np.arange(len(self.ids)), np.diff(offsets))
                rows = np.column_stack([self.ids[source], self.ids[adjacency]])
                rows = rows[np.lexsort(rows.T[::-1])]
            self.diagnostics["status_counts"] = {
                str(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))
            }
        else:
            workspace = self.workspace
            error = int(workspace.error.numpy()[0])
            count = int(workspace.tetrahedra_count.numpy()[0])
            if error and count:
                raise RuntimeError("failed VTM/full reconstruction published nonempty tetrahedra")
            self.error = None if error == 0 else f"construction error bits: {error}"
            rows = canonical_rows(self.ids[workspace.tetrahedra.numpy()[:count]])
            for name in [
                "reuse_attempted",
                "reuse_accepted",
                "reuse_error",
                "reuse_seeds",
                "reuse_rounds",
                "wall_attempted",
                "wall_accepted",
                "wall_error",
                "wall_candidates",
            ]:
                if hasattr(workspace, name):
                    self.diagnostics[name] = int(getattr(workspace, name).numpy()[0])
            self.diagnostics["runtime_certificate"] = (
                "current-coordinate complete exact/symbolic certificate"
            )
            if hasattr(workspace, "_hull_cache_probe"):
                self.diagnostics["hull_cache_misses_hits"] = (
                    workspace._hull_cache_probe.statistics.numpy().tolist()
                )
            if hasattr(workspace, "_split_search_probe"):
                self.diagnostics["exact_search_lanes"] = int(
                    workspace._split_search_probe.statistics.numpy()[0]
                )
        return Output(rows, self.error, self.diagnostics.copy())

    def close(self):
        self.workspace = self.diagram = None


class NativeBackend(Backend):
    def __init__(self, name, **options):
        super().__init__(name, **options)
        import torch

        self.torch = torch
        torch.cuda.set_device(0)
        module_dir = options.get("native_module_dir")
        if module_dir is not None:
            sys.path.insert(0, str(Path(module_dir).resolve()))
        self.native = importlib.import_module(options.get("native_module", "radfoam_headless"))
        self.triangulation = None
        self.order = None

    def synchronize(self):
        self.torch.cuda.synchronize()

    def prepare(self, frame):
        prepared = super().prepare(frame)
        prepared["device_points"] = self.torch.tensor(prepared["points"], device="cuda")
        return prepared

    def update(self, prepared):
        frame = prepared["frame"]
        if frame.previous_tets is not None and self.name == "radfoam-incremental":
            raise UnsupportedState(
                "native RadFoam bindings cannot inject arbitrary retained connectivity"
            )
        if self.options.get("tetrahedra_capacity") is not None:
            raise UnsupportedState("native RadFoam does not expose the requested capacity control")
        changed = self.state_changed(prepared)
        incremental = (
            self.name == "radfoam-incremental"
            and self.triangulation is not None
            and not changed
            and not frame.record["reset"]
        )
        self.error = None
        points = prepared["device_points"]
        self.diagnostics = dict(
            strategy="incremental" if incremental else "full",
            identity_set_changed=changed,
            explicit_reset=frame.record["reset"],
            runtime_certificate="native reference; not VTM's complete certificate",
        )
        if frame.previous_tets is not None:
            self.diagnostics["previous_connectivity"] = "ignored by fresh baseline"
        if not np.isfinite(prepared["points"]).all():
            self.error = "nonfinite input rejected by the native-reference adapter"
            self.triangulation = self.order = None
            self.ids = prepared["ids"]
            return
        if len(prepared["points"]) and np.any(np.ptp(prepared["points"], axis=0) == 0):
            raise UnsupportedState(
                "axis-aligned planar input guarded before native construction after an observed stall"
            )
        try:
            if incremental:
                # Native state indexes its own sorted order, which is carried
                # across the sequence and never inferred from current storage.
                permuted = self.triangulation.rebuild(
                    points[self.order].contiguous(), incremental=True
                )
                if permuted:
                    raise RuntimeError(
                        "unexpected native reorder during same-ID incremental rebuild"
                    )
            else:
                self.triangulation = self.native.Triangulation(points)
                self.order = self.triangulation.permutation().long().clone()
        except (RuntimeError, self.native.TriangulationFailedError) as error:
            self.error = f"{type(error).__name__}: {error}"
            self.triangulation = None
            self.order = None
        self.ids = prepared["ids"]

    def readback(self):
        rows = np.empty((0, 4), np.int64)
        if self.error is None:
            local = self.order[self.triangulation.tets().long()].cpu().numpy()
            rows = canonical_rows(self.ids[local])
            self.diagnostics["directed_edges"] = self.triangulation.point_adjacency().numel()
        return Output(rows, self.error, self.diagnostics.copy())

    def close(self):
        self.triangulation = self.order = None


def create_backend(name, **options):
    if name == "scipy-full":
        return ScipyBackend(name, **options)
    if name.startswith("radfoam-"):
        return NativeBackend(name, **options)
    if name in CONTRACTS:
        return WarpBackend(name, **options)
    raise ValueError(f"unknown backend: {name}")
