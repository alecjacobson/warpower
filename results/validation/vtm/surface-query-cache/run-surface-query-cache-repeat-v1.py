"""Retain every fresh seed1 replay for coordinate/connectivity repeatability, not timing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import warp as wp
from examples.vtm_inputs import surface
from warpower import SurfaceDomain, VariationalMeshingWorkspace

p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
p.add_argument('--replays',type=int,default=8);a=p.parse_args();assert not a.output.exists()
a.output.mkdir();points,faces=surface('thin-hand')
domain=SurfaceDomain(points,faces,surface_subdivisions=20,crease_subdivisions=20,feature_angle=45,corner_angle=45)
mesh=VariationalMeshingWorkspace(domain,1000,initial_count=256,seed=1,quality_rounds=4)
with wp.ScopedCapture() as capture:
    domain.refit();mesh.run()
result=dict(scope=__doc__,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),replays=[])
for i in range(a.replays):
    wp.capture_launch(capture.graph)
    r={name:getattr(mesh,name).numpy().tolist() for name in ['valid','error','counts','rounds','steps','moves','topology','quality_rejections','refinement_rejections']}
    assert r['error']==[0] and r['valid']==[1]
    n,f,m,b=r['counts'];dest=a.output/f'replay-{i}.npz'
    np.savez_compressed(dest,points=mesh.points.numpy()[:n],tetrahedra=mesh.tetrahedra.numpy()[:m],
                        full_tetrahedra=mesh.full_tetrahedra.numpy()[:f],boundary=mesh.boundary_faces.numpy()[:b],
                        roles=mesh.core.boundary.roles.numpy()[:n],radius_ratio=mesh.radius_ratio.numpy()[:m],input_points=points,input_faces=faces)
    r.update(index=i,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    result['replays'].append(r);(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(i,r['counts'],r['sha256'],flush=True)
