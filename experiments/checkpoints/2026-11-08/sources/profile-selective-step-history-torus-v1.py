"""Native16K full solve with exercised selective-history paths; profiling only."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
from benchmarks.mesher import surface
from warpower import SurfaceDomain,VariationalMeshingWorkspace
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--dot',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
inputs=surface('torus');domain=SurfaceDomain(*inputs,surface_subdivisions=4)
mesh=VariationalMeshingWorkspace(domain,16384,initial_count=16384,seed=2026,growth_iterations=0,polish_iterations=32,selective_iterations=10,quality_rounds=4)
with wp.ScopedCapture() as cap:domain.refit();mesh.run()
wp.capture_launch(cap.graph);wp.synchronize();assert mesh.valid.numpy()[0]==1 and mesh.error.numpy()[0]==0
wp.capture_debug_dot_print(cap.graph,str(a.dot),verbose=True)
import torch
cuda=torch.cuda.cudart();cuda.cudaProfilerStart();wp.capture_launch(cap.graph);wp.synchronize();cuda.cudaProfilerStop()
assert mesh.valid.numpy()[0]==1 and mesh.error.numpy()[0]==0
n,full,m,b=mesh.counts.numpy().tolist()
arrays=dict(points=mesh.points.numpy()[:n],tetrahedra=mesh.tetrahedra.numpy()[:m],full_tetrahedra=mesh.full_tetrahedra.numpy()[:full],boundary=mesh.boundary_faces.numpy()[:b],roles=mesh.core.boundary.roles.numpy()[:n],input_points=inputs[0],input_faces=inputs[1])
np.savez_compressed(a.output.with_suffix('.npz'),**arrays)
sources=[Path(__file__),*Path('warpower').glob('*.py')]
r=dict(scope=__doc__,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},work={k:getattr(mesh,k).numpy().tolist() for k in ['counts','rounds','steps','moves','topology','error','valid']},history_statistics=mesh.core.step_history.statistics.numpy().tolist())
a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['work'],r['history_statistics'],flush=True)
assert r['history_statistics'][0]>0 and r['history_statistics'][1]>0 and r['history_statistics'][3]>0
