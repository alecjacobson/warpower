"""Profile the exact graded Armadillo gallery workload once, including refits."""
import argparse,json
from pathlib import Path
import warp as wp
from examples.vtm_inputs import surface
from warpower import SizingField,SurfaceDomain,VariationalMeshingWorkspace
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('.work/vtm/scanned-armadillo-input/armadillo.off'));p.add_argument('--output',type=Path,required=True);p.add_argument('--dot',type=Path,required=True);a=p.parse_args()
points,faces=surface('graded-surface',a.input)
domain=SurfaceDomain(points,faces,surface_subdivisions=1,crease_subdivisions=1,feature_angle=120,corner_angle=45)
lower,upper=points.min(axis=0),points.max(axis=0);anchor=(lower+upper)/2;anchor[0]=lower[0]
sizing=SizingField(wp.array(anchor[None],dtype=wp.vec3,device=domain.device),wp.array([0.35],dtype=float,device=domain.device),gradation=0.65/float(upper[0]-lower[0]),minimum=0.35,maximum=1)
mesh=VariationalMeshingWorkspace(domain,4096,initial_count=256,seed=2026,sizing=sizing,quality_rounds=4)
with wp.ScopedCapture() as capture:domain.refit();sizing.refit();mesh.run()
wp.synchronize();wp.capture_debug_dot_print(capture.graph,str(a.dot),verbose=True)
import torch
cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(capture.graph);wp.synchronize();cuda.cudaProfilerStop()
r=dict(scope='First native launch of a complete fresh graded Armadillo4096 solve with domain/sizing refits, sampling, all optimization/refinement/quality rounds, extraction and diagnostics. Exact gallery parameters (subdivisions1, feature_angle120, seed2026). Setup/readback excluded.',valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),steps=mesh.steps.numpy().tolist(),moves=mesh.moves.numpy().tolist())
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);assert r['valid']==1 and r['error']==0
