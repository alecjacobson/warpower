"""First native launch: eight device-controlled VTM producer/consumer phases."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from tests.test_selective import initial
from tests.test_tetrahedra import check_mesh,tet_set
from warpower._vtm import VTMWorkspace

@wp.kernel
def produce(base:wp.array[wp.vec3],points:wp.array[wp.vec3],surface:wp.array[wp.vec3],current:wp.array[wp.vec3],phase:wp.array[int],budget:wp.array[int],mode:wp.array[int]):
    i=wp.tid();scale=float(1.0)
    if phase[0]==1:scale=1.001
    if i<points.shape[0]:
        points[i]=base[i]*scale
        if phase[0]==6 and i==0:points[i]=wp.vec3(wp.nan)
    if i<current.shape[0]:current[i]=surface[i]*scale
    if i==0:
        budget[0]=8;mode[0]=1
        if phase[0]==2:budget[0]=2;mode[0]=0
        elif phase[0]==3:budget[0]=20
        elif phase[0]==4:mode[0]=-1

@wp.kernel
def count_hit(needed:wp.array[int],calls:wp.array[int]):
    calls[0]+=1
    if needed[0]==0:calls[1]+=1

@wp.kernel
def consume(phase:wp.array[int],points:wp.array[wp.vec3],cells:wp.array2d[int],count:wp.array[int],error:wp.array[int],iterations:wp.array[int],saved_points:wp.array2d[wp.vec3],saved_cells:wp.array3d[int],meta:wp.array2d[int]):
    i=wp.tid();p=phase[0]
    if i<points.shape[0]:saved_points[p,i]=points[i]
    if i<count[0]:
        for j in range(4):saved_cells[p,i,j]=cells[i,j]
    if i==0:meta[p,0]=error[0];meta[p,1]=iterations[0];meta[p,2]=count[0]

@wp.kernel
def advance(phase:wp.array[int],condition:wp.array[int]):
    phase[0]+=1;condition[0]=int(phase[0]<8)

parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true');parser.add_argument('--dot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
fitted=initial(256,20);domain=fitted.domain;source=wp.clone(fitted.points);surface=wp.clone(domain.points);device=domain.device
phase=wp.zeros(1,dtype=int,device=device);condition=wp.zeros(1,dtype=int,device=device)
budget=wp.zeros(1,dtype=int,device=device);mode=wp.zeros(1,dtype=int,device=device)
core=VTMWorkspace(domain,wp.clone(source),max_iterations=8,iteration_limit=budget,selective_enabled=mode,selective_threshold=0.3)
points=wp.empty((8,core.n),dtype=wp.vec3,device=device);cells=wp.empty((8,len(core.full_tetrahedra),4),dtype=int,device=device);meta=wp.empty((8,3),dtype=int,device=device);calls=wp.zeros(2,dtype=int,device=device)
wp.load_module(module=__name__,device=device)
original=core.selection._select_current

def counted_select():
    wp.launch(count_hit,1,inputs=[core.selection._current_needed,calls],device=device)
    original()
core.selection._select_current=counted_select

def step():
    wp.launch(produce,max(core.n,len(surface)),inputs=[source,core.points,surface,domain.points,phase,budget,mode],device=device)
    domain.refit();core.run()
    wp.launch(consume,len(core.full_tetrahedra),inputs=[phase,core.points,core.full_tetrahedra,core.full_tetrahedra_count,core.error,core.iterations,points,cells,meta],device=device)
    wp.launch(advance,1,inputs=[phase,condition],device=device)

def record():
    with wp.ScopedCapture() as cap:
        phase.zero_();condition.fill_(1);calls.zero_();wp.capture_while(condition,step)
    return cap.graph

graph=record();wp.synchronize();wp.capture_debug_dot_print(graph,str(args.dot),verbose=True)
if args.profile:
    import torch
    cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(graph);wp.synchronize()
if args.profile:cuda.cudaProfilerStop()
a=meta.numpy();p=points.numpy();t=cells.numpy();hit_calls=calls.numpy().tolist()
assert int(phase.numpy()[0])==8 and hit_calls[1]>0
for i in range(8):
    if i in [3,4,6]:assert a[i,0]!=0 and a[i,2]==0
    else:assert a[i,0]==0 and a[i,2]>0;check_mesh(p[i],t[i,:a[i,2]])
# A separately recorded fresh-evaluation graph is the per-phase oracle.
core.selection._select_current=core.selection.select
fresh_graph=record();wp.capture_launch(fresh_graph);wp.synchronize()
b=meta.numpy();q=points.numpy();u=cells.numpy();np.testing.assert_array_equal(a,b)
validations=[]
for i in range(8):
    if i in [3,4,6]:validations.append(dict(phase=i,invalid_empty=True));continue
    assert tet_set(t[i,:a[i,2]])==tet_set(u[i,:b[i,2]])
    np.testing.assert_allclose(p[i],q[i],atol=1e-6,rtol=1e-6)
    validations.append(dict(phase=i,fresh_connectivity_equal=True,maximum_coordinate_difference=float(np.max(np.abs(p[i]-q[i]))),mesh_checks_passed=True))
saved=dict(points=p,metadata=a,fresh_points=q,fresh_metadata=b)
for i in range(8):
    saved['tets_'+str(i)]=t[i,:a[i,2]]
    saved['fresh_tets_'+str(i)]=u[i,:b[i,2]]
np.savez_compressed(args.output.with_suffix('.npz'),**saved)
result=dict(scope='First native launch of one eight-phase captured producer/domain-refit/VTM/consumer loop; device inputs scale, global/selective modes, invalid budget/mode/point and recovery. A device counter observes internal selection hits; otherwise production code. Setup, final readback and separate forced-fresh graph validation excluded from native profiling.',selection_calls=hit_calls[0],selection_hits=hit_calls[1],metadata=a.tolist(),validation=validations,mesh_sha256=hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest())
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
