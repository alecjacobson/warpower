"""Profile one complete fresh integrated torus4096 solve, including refit."""
import argparse,json
from pathlib import Path
import warp as wp
import warpower
from benchmarks.mesher import surface
parser=argparse.ArgumentParser();parser.add_argument('--dot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
wp.init();domain=warpower.SurfaceDomain(*surface('torus'),surface_subdivisions=4)
mesh=warpower.VariationalMeshingWorkspace(domain,4096,initial_count=256,seed=2026)
with wp.ScopedCapture() as cap:
    domain.refit();mesh.run()
wp.synchronize();wp.capture_debug_dot_print(cap.graph,str(args.dot),verbose=True)
import torch
cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(cap.graph);wp.synchronize()
cuda.cudaProfilerStop()
full=mesh.core.mesh._full
r=dict(scope='One complete fresh captured torus4096 solve, including current domain refit, fresh sampling, all fitting/refinement/quality work, extraction and diagnostics. Setup and readback excluded from native profiling.',valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),steps=mesh.steps.numpy().tolist(),routing=mesh.core.mesh.route_calls.numpy().tolist(),last_full_reconstruction=dict(reuse_attempted=int(full.reuse_attempted.numpy()[0]),reuse_accepted=int(full.reuse_accepted.numpy()[0]),proofs_skipped_checked=full._reuse_completion._certificate.proof_statistics.numpy().tolist()))
args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert r['valid']==1 and r['error']==0
