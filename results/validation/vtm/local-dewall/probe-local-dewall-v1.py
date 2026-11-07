import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from scipy.spatial import Delaunay
from warpower import _bvh
from warpower._delaunay_certificate import DelaunayCertificate
from benchmarks.run import cpu_quota_snapshot, cpu_quota_delta
from local_dewall_v1 import Candidate

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=128)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
wp.init()
wp.set_device('cuda:0')
points = np.random.default_rng(2026).uniform(-1, 1, (args.n, 3)).astype(np.float32)
reference = Delaunay(points).simplices
p = wp.array(points, dtype=wp.vec3)
h = _bvh.build(p, wp.zeros(args.n, dtype=float), leaf_size=4)
candidate = Candidate(p, h.tree)
certificate = DelaunayCertificate(p, candidate.tets, candidate.count, h.tree)
with wp.ScopedCapture() as capture:
    candidate.run()
with wp.ScopedCapture() as cert_capture:
    certificate.run()
wp.capture_launch(capture.graph)
wp.capture_launch(cert_capture.graph)
wp.synchronize()
begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
before = cpu_quota_snapshot()
timings = []
for _ in range(5):
    wp.record_event(begin)
    wp.capture_launch(capture.graph)
    wp.record_event(end)
    wp.synchronize_event(end)
    timings.append(wp.get_event_elapsed_time(begin, end))
count = int(candidate.count.numpy()[0])
tets = candidate.tets.numpy()[:min(count, len(candidate.tets))]
actual = {tuple(sorted(t)) for t in tets}
expected = {tuple(sorted(t)) for t in reference}
status, seed_status = candidate.status.numpy(), candidate.seed_status.numpy()
report = dict(n=args.n, candidate_ms=timings, median_ms=float(np.median(timings)), cpu=cpu_quota_delta(before, cpu_quota_snapshot()), count=count, unique=len(actual), reference_count=len(expected), missing=len(expected-actual), excess=len(actual-expected), certificate_error=int(certificate.error.numpy()[0]), seed_status={str(k): int(v) for k,v in zip(*np.unique(seed_status, return_counts=True))}, status={str(k): int(v) for k,v in zip(*np.unique(status, return_counts=True))}, seed_searches=int(candidate.seed_costs.numpy()[:,0].sum()), seed_nodes=int(candidate.seed_costs.numpy()[:,1].sum()), walk_searches=int(candidate.costs.numpy()[:,0].sum()), walk_nodes=int(candidate.costs.numpy()[:,1].sum()))
args.output.write_text(json.dumps(report, indent=2)+'\n')
np.savez_compressed(args.output.with_suffix('.npz'), points=points, tets=tets, reference=reference, seed_status=seed_status, status=status, seeds=candidate.seeds.numpy(), seed_costs=candidate.seed_costs.numpy(), costs=candidate.costs.numpy())
print(json.dumps(report, indent=2))
