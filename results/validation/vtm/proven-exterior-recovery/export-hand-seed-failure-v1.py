"""Save internal final candidates without changing the mesher's failure publication."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

from examples.vtm_inputs import surface
from warpower import SurfaceDomain, VariationalMeshingWorkspace

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.with_suffix('.npz').exists() and not a.output.with_suffix('.json').exists()
input_points, input_faces = surface('thin-hand')
domain = SurfaceDomain(input_points, input_faces, surface_subdivisions=20,
                       crease_subdivisions=20, feature_angle=45, corner_angle=45)
m = VariationalMeshingWorkspace(domain, 1000, initial_count=256, seed=a.seed,
                                quality_rounds=4, local_quality_retries=0)
with wp.ScopedCapture() as capture:
    domain.refit()
    m.run()
wp.capture_launch(capture.graph)
wp.synchronize()
c, e = m.core, m.extraction
n, nf, nt, nb = [int(x.numpy()[0]) for x in
                [m.active_count, c.full_tetrahedra_count, e.tetrahedra_count, e.boundary_count]]
assert 0 <= n <= 1000 and 0 <= nf <= len(c.full_tetrahedra)
assert 0 <= nt <= len(e.tetrahedra) and 0 <= nb <= len(e.boundary_faces)
values = dict(points=m.points.numpy()[:n], full_tetrahedra=c.full_tetrahedra.numpy()[:nf],
              tetrahedra=e.tetrahedra.numpy()[:nt], boundary=e.boundary_faces.numpy()[:nb],
              source_indices=e.source_indices.numpy()[:nt], selected=e.selected.numpy()[:nf],
              centers=e.circumcenters.numpy()[:nf], radii=e.circumradii.numpy()[:nf],
              signed_distance=e.signed_distance.numpy()[:nf], radius_ratio=e.radius_ratio.numpy()[:nt],
              input_quality=e._input_quality.numpy()[:nf], roles=c.boundary.roles.numpy()[:n],
              targets=c.boundary.targets.numpy()[:n], corner_sites=c.boundary.corner_sites.numpy(),
              corner_points=domain.corner_points.numpy(), input_points=input_points, input_faces=input_faces)
np.savez_compressed(a.output.with_suffix('.npz'), **values)
record = dict(scope='Diagnostic internal final candidate, including invalid material selections. Public validity/counts remain authoritative. One ordinary complete captured solve; readback/export afterward. No timing claim.',
              runtime=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=a.seed,
              parameters=dict(case='thin-hand', n=1000, initial_count=256, surface_subdivisions=20,
                              crease_subdivisions=20, quality_rounds=4, local_quality_retries=0),
              internal_counts=[n, nf, nt, nb],
              status={name: getattr(m, name).numpy().tolist() for name in
                      ['valid', 'error', 'counts', 'topology', 'rounds', 'steps', 'moves', 'last_errors', 'last_bad_extraction']},
              extraction={name: getattr(e, name).numpy().tolist() for name in
                          ['valid', 'error', 'diagnostics', 'repaired_count', 'repair_rounds',
                           'removed_count', 'exterior_rounds', 'exterior_rejected', 'component_count', 'shell_count', 'cavity_count']},
              core_error=int(c.error.numpy()[0]), exterior_enabled=int(m._exterior_enabled.numpy()[0]),
              snapshot_sha256=hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest())
a.output.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
assert record['core_error'] == 0
expected = 64 if a.seed == 7 else 0
assert record['status']['error'] == [expected]
