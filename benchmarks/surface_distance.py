"""Optional native distance oracle for offline validation of dense input surfaces."""

import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from examples.vtm_inputs import write_off


class NativeSurfaceDistance:
    """Independent CGAL AABB distances; never invoked from a meshing graph."""

    description = "CGAL AABB triangle distances with float64 constructions"

    def __init__(self, binary, scratch):
        self.binary = Path(binary).resolve(strict=True)
        self.scratch = Path(scratch)
        self.scratch.mkdir(parents=True, exist_ok=True)
        version = subprocess.run(
            [str(self.binary), "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if not version.startswith("CGAL "):
            raise ValueError("Expected the compiled benchmarks/cpu/surface_distance.cpp oracle")
        self.metadata = dict(
            implementation=self.description,
            version=version,
            binary_sha256=hashlib.sha256(self.binary.read_bytes()).hexdigest(),
            source="benchmarks/cpu/surface_distance.cpp",
            source_sha256=hashlib.sha256(
                (Path(__file__).parent / "cpu/surface_distance.cpp").read_bytes()
            ).hexdigest(),
        )

    def __call__(self, points, triangles, queries):
        p, t, q = (
            np.asarray(points, np.float64),
            np.asarray(triangles),
            np.asarray(queries, np.float64),
        )
        if any(a.ndim != 2 or a.shape[1] != 3 for a in [p, t, q]):
            raise ValueError("Expected (N,3) points, (F,3) triangles and (Q,3) queries")
        if not np.issubdtype(t.dtype, np.integer) or not len(p) or not len(t):
            raise ValueError("Expected a nonempty indexed triangle surface")
        if t.min() < 0 or t.max() >= len(p) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError("Surface indices and finite coordinates are required")
        if not len(q):
            return np.empty(0, np.float64)
        # These are reproducible scratch inputs; unique results remain in the
        # caller's report. The caller controls the persistent scratch directory.
        with tempfile.TemporaryDirectory(prefix="surface-distance-", dir=self.scratch) as directory:
            target, samples = Path(directory) / "target.off", Path(directory) / "queries.txt"
            write_off(target, p, t)
            np.savetxt(samples, q, fmt="%.17g", header=str(len(q)), comments="")
            process = subprocess.run(
                [str(self.binary), str(target), str(samples)],
                capture_output=True,
                text=True,
                check=True,
            )
            squared = np.array([float(v) for v in process.stdout.split()], np.float64)
        if squared.shape != (len(q),) or not np.isfinite(squared).all() or np.any(squared < 0):
            raise ValueError("Native distance oracle returned invalid or incomplete distances")
        return np.sqrt(squared)
