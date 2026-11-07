import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from scipy.spatial import Delaunay
from benchmarks.run import cpu_quota_snapshot, cpu_quota_delta
from local_dewall_integration_v2 import install

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=1000)
parser.add_argument('--mode', choices=['control', 'candidate'], required=True)
parser.add_argument('--case', choices=['random', 'grid', 'thin', 'cluster'], default='random')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.mode == 'candidate':
    install()
from warpower import DelaunayWorkspace
wp.init()
wp.set_device('cuda:0')
rng = np.random.default_rng(2026)
points = rng.uniform(-1, 1, (args.n, 3)).astype(np.float32)
if args.case == 'grid':
    from itertools import product
    points = np.array(list(product(range(4), repeat=3)), np.float32)
elif args.case == 'thin':
    points[:, 2] *= 0.05
elif args.case == 'cluster':
    centers = rng.uniform(-0.8, 0.8, (5, 3))
    points = (centers[np.arange(args.n) % 5] + rng.normal(0, 0.05, (args.n, 3))).astype(np.float32)
workspace = DelaunayWorkspace(points, cache_strategy=False, fast_candidates=False)
workspace.run()
wp.synchronize()
begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
before = cpu_quota_snapshot()
timings = []
for _ in range(5):
    wp.record_event(begin)
    workspace.run()
    wp.record_event(end)
    wp.synchronize_event(end)
    timings.append(wp.get_event_elapsed_time(begin, end))
quota = cpu_quota_delta(before, cpu_quota_snapshot())
count = int(workspace.tetrahedra_count.numpy()[0])
tets = workspace.tetrahedra.numpy()[:count]
reference = Delaunay(points).simplices
actual = {tuple(sorted(t)) for t in tets}
expected = {tuple(sorted(t)) for t in reference}
report = dict(n=len(points), mode=args.mode, case=args.case, full_ms=timings, median_ms=float(np.median(timings)), cpu=quota, count=count, unique=len(actual), reference_count=len(expected), missing=len(expected-actual), excess=len(actual-expected), error=int(workspace.error.numpy()[0]), certificate_error=int(workspace.certificate_error.numpy()[0]))
if args.mode == 'candidate':
    base = workspace._base
    report.update(wall_accepted=int(base.wall_accepted.numpy()[0]), wall_fallback=int(base.wall_fallback.numpy()[0]), candidate_error=int(base._wall.error.numpy()[0]), seed_error=int(base._wall_filter.error.numpy()[0]), raw_count=int(base._wall.raw_count.numpy()[0]), seed_count=int(base._wall_filter.tetrahedra_count.numpy()[0]), completion_error=int(base._wall_completion.error.numpy()[0]), completion_certificate_error=int(base._wall_completion.certificate_error.numpy()[0]), completion_rounds=int(base._wall_completion.rounds.numpy()[0]))
args.output.write_text(json.dumps(report, indent=2)+'\n')
np.savez_compressed(args.output.with_suffix('.npz'), points=points, tets=tets, reference=reference)
print(json.dumps(report, indent=2))
assert report['error'] == 0
if args.case != 'grid':
    assert report['missing'] == report['excess'] == 0
