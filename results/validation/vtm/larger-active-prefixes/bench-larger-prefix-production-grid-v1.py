"""Structured tie stress for the private retained-mesh route, not full VTM timing."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

from benchmarks.run import cpu_quota_delta, cpu_quota_snapshot

p = argparse.ArgumentParser()
p.add_argument("--side", type=int, default=17)
p.add_argument("--repeats", type=int, default=9)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
from warpower._wall_reuse import WallReuseDelaunayWorkspace  # noqa: E402

points = np.indices((a.side,) * 3).reshape(3, -1).T.astype(np.float32)
start = time.time()
mesh = WallReuseDelaunayWorkspace(points, facet_recovery=True)
with wp.ScopedCapture() as cap:
    mesh.run()
wp.synchronize()
setup = time.time() - start
q = cpu_quota_snapshot()
wp.capture_launch(cap.graph)
wp.synchronize()
initial_quota = cpu_quota_delta(q, cpu_quota_snapshot())
r = dict(
    scope=__doc__,
    n=len(points),
    side=a.side,
    source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    runtime_query_sha256={
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest() if Path(name).exists() else None
        for name in ["warpower/_active_delaunay.py", "warpower/_vtm.py", "warpower/_proved_completion.py", "warpower/_unproved_queries.py"]
    },
    points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
    setup_seconds=setup,
    initial_cpu_quota_delta=initial_quota,
    error=int(mesh.error.numpy()[0]),
    count=int(mesh.tetrahedra_count.numpy()[0]),
    interpretation="Initial full construction and setup are outside retained-mesh timing. Each timed replay rebuilds current owners/hierarchy, certifies the retained fragment and publishes a complete full hull on unchanged grid points. This is a stress/performance guard, not a fresh VTM or CPU speedup comparison.",
)
if r["error"] != 0:
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    raise SystemExit(1)
wp.capture_launch(cap.graph)
wp.synchronize()
assert int(mesh.reuse_accepted.numpy()[0]) == 1
begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
samples = []
q = cpu_quota_snapshot()
for _ in range(a.repeats):
    wp.record_event(begin)
    wp.capture_launch(cap.graph)
    wp.record_event(end)
    wp.synchronize_event(end)
    samples.append(wp.get_event_elapsed_time(begin, end))
r.update(
    cpu_quota_delta=cpu_quota_delta(q, cpu_quota_snapshot()),
    gpu_milliseconds=samples,
    median_ms=float(np.median(samples)),
    error=int(mesh.error.numpy()[0]),
    count=int(mesh.tetrahedra_count.numpy()[0]),
)
for name in [
    "reuse_attempted",
    "reuse_accepted",
    "reuse_error",
    "reuse_seeds",
    "reuse_rounds",
    "base_attempted",
]:
    r[name] = getattr(mesh, name).numpy().tolist()
np.savez_compressed(
    a.output.with_suffix(".npz"), points=points, tetrahedra=mesh.tetrahedra.numpy()[: r["count"]]
)
r["snapshot_sha256"] = hashlib.sha256(a.output.with_suffix(".npz").read_bytes()).hexdigest()
a.output.write_text(json.dumps(r, indent=2) + "\n")
print(r, flush=True)
assert r["error"] == 0 and r["reuse_accepted"] == [1]
