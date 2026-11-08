"""One warmed fresh complete torus16384-fixed32 solve, for native cost attribution."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
from prepared_new_cell_integration_v1 import install
install()
from benchmarks.mesher import surface
from warpower import SurfaceDomain,VariationalMeshingWorkspace

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dot',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
inputs=surface('torus')
domain=SurfaceDomain(*inputs,surface_subdivisions=4,crease_subdivisions=4,feature_angle=45,corner_angle=45)
mesh=VariationalMeshingWorkspace(domain,16384,initial_count=16384,growth_iterations=0,polish_iterations=32,seed=2026,quality_rounds=4)
with wp.ScopedCapture() as capture:
 domain.refit();mesh.run()
wp.capture_launch(capture.graph);wp.synchronize()
assert int(mesh.error.numpy()[0])==0 and int(mesh.valid.numpy()[0])==1
wp.capture_debug_dot_print(capture.graph,str(a.dot),verbose=True)
import torch
cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(capture.graph);wp.synchronize();cuda.cudaProfilerStop()
r=dict(scope='One warmed fresh complete resident torus16384-fixed32 solve with private shared/prepared new-cell queries, including refit, sampling, fixed-site fitting, four quality rounds and extraction. One warmup outside profiling; setup/readback excluded. One external final synchronization. Diagnostic cost attribution, not clean performance evidence.',runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),parameters=dict(case='torus',n=16384,initial=16384,growth_iterations=0,polish_iterations=32,seed=2026,quality_rounds=4,subdivisions=4,crease_subdivisions=4),device=domain.device.name,warp=wp.__version__)
for name in ['valid','error','counts','rounds','steps','moves','topology','quality_rejections','refinement_rejections']:r[name]=getattr(mesh,name).numpy().tolist()
assert r['error']==[0] and r['valid']==[1]
n,full,m,b=r['counts'];arrays=dict(points=mesh.points.numpy()[:n],tetrahedra=mesh.tetrahedra.numpy()[:m],full_tetrahedra=mesh.full_tetrahedra.numpy()[:full],boundary=mesh.boundary_faces.numpy()[:b],radius_ratio=mesh.radius_ratio.numpy()[:m],roles=mesh.core.boundary.roles.numpy()[:n],input_points=inputs[0],input_faces=inputs[1])
snapshot=a.output.with_suffix('.npz');assert not snapshot.exists();np.savez_compressed(snapshot,**arrays)
ref=Path('.work/vtm/prepared-new-cell-pilot-v1/torus16384-fixed32-after');baseline=json.loads(ref.with_suffix('.json').read_text())
r['reference_result_sha256']=hashlib.sha256(ref.with_suffix('.json').read_bytes()).hexdigest();r['snapshot_sha256']=hashlib.sha256(snapshot.read_bytes()).hexdigest()
r['work_equal']={k:r[k]==baseline[k] for k in ['counts','rounds','topology']}
r['work_equal'].update(steps=r['steps']==baseline['optimizer_steps'],moves=r['moves']==baseline['quality_moves'])
r['work_equal'].update({k:r[k]==[baseline[k]] for k in ['quality_rejections','refinement_rejections']})
def cells(x):
 y=np.sort(x,axis=1);return y[np.lexsort(y.T[::-1])]
with np.load(ref.with_suffix('.npz')) as old:
 r['array_bitwise_equal']={k:np.array_equal(old[k],v) for k,v in arrays.items() if k in old}
 r['canonical_cells_equal']={k:np.array_equal(cells(old[k]),cells(arrays[k])) for k in ['tetrahedra','full_tetrahedra','boundary']}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert all(r['work_equal'].values()) and all(r['canonical_cells_equal'].values())
assert r['array_bitwise_equal']['points'] and r['array_bitwise_equal']['roles']
