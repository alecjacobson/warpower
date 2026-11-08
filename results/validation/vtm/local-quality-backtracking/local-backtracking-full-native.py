"""First native full solve with opt-in local rollback, followed by device consumption."""
import argparse,json
from pathlib import Path
import warp as wp
from examples.vtm_inputs import surface
from warpower import SurfaceDomain,VariationalMeshingWorkspace
from tests.test_vtm_core import consume
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dot',type=Path,required=True);a=p.parse_args()
domain=SurfaceDomain(*surface('counterbored-plate'),surface_subdivisions=16,crease_subdivisions=16,feature_angle=45,corner_angle=45)
mesh=VariationalMeshingWorkspace(domain,1000,initial_count=256,seed=2026,quality_rounds=4,local_quality_retries=12)
volume=wp.zeros(1,dtype=wp.float64);wp.load_module(module='tests.test_vtm_core',device=domain.device)
with wp.ScopedCapture() as capture:
 domain.refit();mesh.run();volume.zero_()
 wp.launch(consume,len(mesh.tetrahedra),inputs=[mesh.points,mesh.tetrahedra,mesh.tetrahedra_count,mesh.error,volume])
wp.synchronize();wp.capture_debug_dot_print(capture.graph,str(a.dot),verbose=True)
import torch
cuda=torch.cuda.cudart();cuda.cudaProfilerStart();wp.capture_launch(capture.graph);wp.synchronize();cuda.cudaProfilerStop()
r=dict(scope='First native full counterbored-plate 1000-site solve with local_quality_retries=12, domain refit, all fresh sampling/growth/optimization/quality/extraction and downstream device volume consumption. Setup/readback excluded; one final external synchronization.',valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),steps=mesh.steps.numpy().tolist(),moves=mesh.moves.numpy().tolist(),last_core_local_statistics=mesh.core.local_backtracking.statistics.numpy().tolist(),volume=float(mesh.volume.numpy()[0]),consumer_volume=float(volume.numpy()[0]))
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert r['valid']==1 and r['error']==0 and abs(r['volume']-r['consumer_volume'])<1e-9
