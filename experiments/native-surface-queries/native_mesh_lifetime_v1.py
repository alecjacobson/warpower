"""Minimal native Mesh / conditional-query capture diagnostic; no Warpower imports."""
import argparse,json
import numpy as np
import warp as wp
from benchmarks.memcheck import configure

@wp.kernel
def query(mesh:wp.uint64,result:wp.array[float]):
    hit=wp.mesh_query_point(mesh,wp.vec3(0.0),100.0)
    result[0]=wp.where(hit.result,hit.sign,99.0)

p=argparse.ArgumentParser();p.add_argument('--depth',type=int,default=1);p.add_argument('--refit',action='store_true');p.add_argument('--output');a=p.parse_args()
configure(32768,True)
points=wp.array(np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],np.float32),dtype=wp.vec3,device='cuda:0')
indices=wp.array([0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,2,3,7,2,7,6,0,4,7,0,7,3,1,2,6,1,6,5],dtype=int,device='cuda:0')
mesh=wp.Mesh(points,indices,bvh_constructor='lbvh');condition=wp.ones(1,dtype=int,device='cuda:0');result=wp.zeros(1,dtype=float,device='cuda:0')
wp.load_module(module=__name__,device='cuda:0');wp.launch(query,1,inputs=[mesh.id,result]);wp.synchronize()
print('mesh descriptor',hex(mesh.id),flush=True)
def nested(depth):
    if depth:wp.capture_if(condition,lambda:nested(depth-1))
    else:wp.launch(query,1,inputs=[mesh.id,result])
with wp.ScopedCapture() as cap:
    if a.refit:mesh.refit()
    nested(a.depth)
wp.synchronize()
for _ in range(3):wp.capture_launch(cap.graph)
wp.synchronize();value=float(result.numpy()[0]);assert value<0,value
report=dict(depth=a.depth,refit=a.refit,mesh_id=hex(mesh.id),replays=3,result=value,all_owners_alive=True,setup_and_capture_synchronized=True)
print(json.dumps(report),flush=True)
if a.output:
 from pathlib import Path
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
