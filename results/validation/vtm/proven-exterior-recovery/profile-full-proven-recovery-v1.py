"""First complete private recovered solve, for native kernel cost attribution."""
import argparse
import hashlib
import json
from pathlib import Path

import warp as wp
from full_proven_recovery_v1 import install
from examples.vtm_inputs import surface
from warpower import SurfaceDomain, VariationalMeshingWorkspace

p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); p.add_argument('--dot', type=Path, required=True)
a = p.parse_args(); assert not a.output.exists()
install()
domain = SurfaceDomain(*surface('thin-hand'), surface_subdivisions=20, crease_subdivisions=20, feature_angle=45, corner_angle=45)
mesh = VariationalMeshingWorkspace(domain, 1000, initial_count=256, seed=7, quality_rounds=4)
with wp.ScopedCapture() as capture:
    domain.refit(); mesh.run()
wp.synchronize(); wp.capture_debug_dot_print(capture.graph, str(a.dot), verbose=True)
import torch
cuda = torch.cuda.cudart(); cuda.cudaProfilerStart()
wp.capture_launch(capture.graph); wp.synchronize(); cuda.cudaProfilerStop()
r = dict(scope='First native complete fresh private seed7 solve, including last-chance midpoint step and exact exterior-proof recovery in all quality stages. Setup/readback excluded; one final external synchronization. Profiling cost attribution, not clean timing.',
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         valid=int(mesh.valid.numpy()[0]), error=int(mesh.error.numpy()[0]), counts=mesh.counts.numpy().tolist(),
         steps=mesh.steps.numpy().tolist(), moves=mesh.moves.numpy().tolist(), rounds=mesh.rounds.numpy().tolist(),
         proof_enabled=int(mesh._proof_enabled.numpy()[0]), last_parent_proof_removed=int(mesh.extraction._proof.removed.numpy()[0]))
a.output.write_text(json.dumps(r, indent=2)+'\n'); print(json.dumps(r), flush=True)
assert r['valid'] == 1 and r['error'] == 0
