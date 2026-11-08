"""Native first-launch full solve or producer/mesher/consumer failure-recovery loop."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from examples.vtm_inputs import surface
from tests.test_proven_mesher import producer,consumer
from warpower import SurfaceDomain,VariationalMeshingWorkspace
p=argparse.ArgumentParser();p.add_argument('--loop',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--dot',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
domain=SurfaceDomain(*surface('thin-hand'),surface_subdivisions=20,crease_subdivisions=20)
seed=wp.full(1,7,dtype=int);target=wp.full(1,1000,dtype=int)
mesh=VariationalMeshingWorkspace(domain,1000,initial_count=256,seed=seed,target_count=target,local_quality_retries=4)
phase=wp.zeros(1,dtype=int);condition=wp.ones(1,dtype=int);rows=wp.empty((4,8),dtype=int)
wp.load_module(module='tests.test_proven_mesher',device=domain.device)
def step():
 wp.launch(producer,1,inputs=[phase,target,seed]);domain.refit();mesh.run()
 wp.launch(consumer,1,inputs=[phase,condition,mesh.error,mesh.valid,mesh.counts,mesh._proven_exterior_enabled,mesh._boundary_rounding.statistics,rows])
with wp.ScopedCapture() as capture:
 if a.loop:
  phase.zero_();condition.fill_(1);wp.capture_while(condition,step)
 else:domain.refit();mesh.run()
wp.synchronize();wp.capture_debug_dot_print(capture.graph,str(a.dot),verbose=True)
import torch
cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(capture.graph);wp.synchronize();cuda.cudaProfilerStop()
r=dict(scope='First native production graph launch; setup and final readback excluded, one external final synchronization; local_quality_retries=4. Profiling evidence, not clean timing.',loop=a.loop,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),rounds=mesh.rounds.numpy().tolist(),steps=mesh.steps.numpy().tolist(),moves=mesh.moves.numpy().tolist(),proof_enabled=int(mesh._proven_exterior_enabled.numpy()[0]),proof_removed=int(mesh.extraction.proven_removed_count.numpy()[0]),midpoint=mesh._boundary_rounding.statistics.numpy().tolist(),local_statistics=mesh.core.local_backtracking.statistics.numpy().tolist())
if a.loop:
 r.update(metadata=rows.numpy().tolist(),columns=['error','valid','proof_enabled','accepted_midpoint_axes','sites','full_tets','material_tets','boundary_faces'])
 np.testing.assert_array_equal(rows.numpy()[:,0],[0,1,0,0]);np.testing.assert_array_equal(rows.numpy()[:,1],[1,0,1,1]);np.testing.assert_array_equal(rows.numpy()[:,2],[1,0,0,1])
 np.testing.assert_array_equal(rows.numpy()[1,4:],0)
assert r['valid']==1 and r['error']==0 and r['proof_enabled']==1 and r['proof_removed']>0 and r['midpoint'][2]>0
assert mesh.core._boundary_rounding is None
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
