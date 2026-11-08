"""Bounded private post-failure continuation, followed by captured proof removal."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from warpower import SurfaceDomain, ExtractionWorkspace, _vtm
from representable_boundary_v1 import install as install_midpoint
from proven_exterior_transaction_v1 import ProofRemoval

p = argparse.ArgumentParser()
p.add_argument('--steps', type=int, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.with_suffix('.json').exists() and not a.output.with_suffix('.npz').exists()
source = Path('.work/vtm/hand-seed7-final-v1.npz'); d = np.load(source)
domain = SurfaceDomain(d['input_points'], d['input_faces'], surface_subdivisions=20,
                       crease_subdivisions=20, feature_angle=45, corner_angle=45)
core = _vtm.VTMWorkspace(domain, d['points'], max_iterations=a.steps)
install_midpoint(core)
e = ExtractionWorkspace(domain, core.points, core.full_tetrahedra, core.full_tetrahedra_count,
                        source_error=core.error, containment_recovery=True,
                        exterior_recovery=True, feature_roles=core.boundary.roles)
proof = ProofRemoval(e)
with wp.ScopedCapture() as capture:
    domain.refit()
    core._tie_stats.zero_()
    core.run()
    e.run()
wp.capture_launch(capture.graph)
wp.synchronize()
nf, nt, nb = [int(x.numpy()[0]) for x in [core.full_tetrahedra_count, e.tetrahedra_count, e.boundary_count]]
values = dict(points=core.points.numpy(), full_tetrahedra=core.full_tetrahedra.numpy()[:nf],
              tetrahedra=e.tetrahedra.numpy()[:nt], boundary=e.boundary_faces.numpy()[:nb],
              selected=e.selected.numpy()[:nf], source_indices=e.source_indices.numpy()[:nt],
              centers=e.circumcenters.numpy()[:nf], radii=e.circumradii.numpy()[:nf],
              signed_distance=e.signed_distance.numpy()[:nf], input_quality=e._input_quality.numpy()[:nf],
              radius_ratio=e.radius_ratio.numpy()[:nt], roles=core.boundary.roles.numpy(),
              input_points=d['input_points'], input_faces=d['input_faces'])
np.savez_compressed(a.output.with_suffix('.npz'), **values)
names = [__file__, '.work/vtm/proven_exterior_transaction_v1.py', '.work/vtm/proven_exterior_v1.py', '.work/vtm/representable_boundary_v1.py']
r = dict(scope='Private bounded continuation of the saved ordinary failed seed7 candidate. Original energy/geometry guards, midpoint-only rounding, then one-pass exact exterior removal with reference/volume/topology rollback. Not yet a full fresh mesher solve or performance result.',
         input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), steps=a.steps,
         source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names},
         core_error=int(core.error.numpy()[0]), accepted_iterations=int(core.iterations.numpy()[0]),
         rejected_steps=int(core.rejected_steps.numpy()[0]), midpoint=core._tie_stats.numpy().tolist(),
         extraction={k: getattr(e, k).numpy().tolist() for k in ['valid', 'error', 'diagnostics', 'tetrahedra_count', 'boundary_count', 'component_count', 'shell_count', 'cavity_count', 'volume', 'minimum_radius_ratio']},
         proof_removed=int(proof.removed.numpy()[0]), proof_rejected=int(proof.rejected.numpy()[0]),
         snapshot_sha256=hashlib.sha256(a.output.with_suffix('.npz').read_bytes()).hexdigest())
a.output.with_suffix('.json').write_text(json.dumps(r, indent=2)+'\n')
print(json.dumps(r), flush=True)
