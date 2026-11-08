"""Run unchanged complete benchmark/gallery with optional read-only device tracing."""
import hashlib
import json
import os
import runpy
import time
from pathlib import Path
import numpy as np
import warp as wp
import warpower
from warpower import _mesher
import step_observer_v1 as observer

mode = os.environ["WARPOWER_STEP_MODE"]
assert mode in ["reference", "observed"]
destination = Path(os.environ["WARPOWER_STEP_REPORT"])
assert not destination.exists()
sources = [*Path("warpower").glob("*.py"), Path(__file__), Path(observer.__file__),
           Path("benchmarks/mesher.py"), Path("examples/vtm_gallery.py")]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
if mode == "observed":
    observer.install()
base = _mesher.VariationalMeshingWorkspace
instances = []


class ObservedMesher(base):
    def __init__(self, domain, *args, **kwargs):
        start = time.perf_counter()
        super().__init__(domain, *args, **kwargs)
        self.observed_setup_seconds = time.perf_counter() - start
        instances.append(self)


warpower.VariationalMeshingWorkspace = ObservedMesher
_mesher.VariationalMeshingWorkspace = ObservedMesher
completed = False
try:
    module = "benchmarks.mesher" if os.environ["WARPOWER_STEP_KIND"] == "benchmark" else "examples.vtm_gallery"
    runpy.run_module(module, run_name="__main__")
    completed = True
finally:
    report = dict(scope=__doc__, mode=mode, completed=completed, source_sha256=hashes,
                  hooks=observer.HOOKS, timing_is_performance_evidence=False)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p,h in hashes.items())
    if instances:
        assert len(instances) == 1
        m = instances[0]
        report["core_options"] = dict(damping=m.core.damping, tolerance=m.core.tolerance,
            max_backtracks=m.core.max_backtracks, local_quality_retries=m.local_quality_retries)
        if mode == "observed":
            o = m.core.step_observer
            count = int(o.count.numpy()[0])
            overflow = int(o.overflow.numpy()[0])
            path = destination.with_suffix(".npz")
            assert not path.exists()
            np.savez_compressed(path, ints=o.ints.numpy()[:min(count,o.capacity)],
                                floats=o.floats.numpy()[:min(count,o.capacity)])
            report.update(count=count, overflow=overflow, capacity=o.capacity,
                          trace=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          int_columns=observer.INT_COLUMNS, float_columns=observer.FLOAT_COLUMNS,
                          events=observer.EVENTS, stages=observer.STAGES, phases=observer.PHASES)
        recovery = m.topology_recovery
        report["recovery"] = None if recovery is None else {
            name: getattr(recovery,name).numpy().tolist() for name in
            ["chosen", "invoked", "accepted", "attempts", "error", "status", "rejection_reason",
             "metrics", "previous_metrics", "topology", "previous_topology", "limit", "frame",
             "proposed_position", "references_rejected", "last_errors", "point_count", "full_count", "retained_count"]}
    destination.write_text(json.dumps(report, indent=2)+"\n")
    if completed and mode == "observed":
        assert report["count"] > 0 and report["overflow"] == 0
