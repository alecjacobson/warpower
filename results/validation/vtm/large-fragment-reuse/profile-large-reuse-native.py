"""One complete fresh solve for native activity attribution, not clean timing."""
import argparse,json
from pathlib import Path
import warp as wp
import warpower
from benchmarks.mesher import surface
parser=argparse.ArgumentParser();parser.add_argument('--reuse',action='store_true');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
if args.reuse:
    import warpower._active_delaunay as active
    from warpower._reuse_delaunay import ReuseDelaunayWorkspace
    active.DelaunayWorkspace=ReuseDelaunayWorkspace
    original=active.ActiveDelaunayWorkspace.clear_reuse
    def clear(self):
        original(self);self._full.clear_reuse()
    active.ActiveDelaunayWorkspace.clear_reuse=clear
wp.init()
domain=warpower.SurfaceDomain(*surface('torus'),surface_subdivisions=4)
mesh=warpower.VariationalMeshingWorkspace(domain,4096,initial_count=256,seed=2026)
with wp.ScopedCapture() as cap:
    domain.refit();mesh.run()
wp.synchronize()
import torch
cuda=torch.cuda.cudart()
cuda.cudaProfilerStart()
wp.capture_launch(cap.graph);wp.synchronize()
cuda.cudaProfilerStop()
r=dict(scope='One complete fresh captured torus4096 solve for native profiling; not a clean performance sample.',reuse=args.reuse,valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),steps=mesh.steps.numpy().tolist(),routing=mesh.core.mesh.route_calls.numpy().tolist())
args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert r['valid']==1 and r['error']==0
