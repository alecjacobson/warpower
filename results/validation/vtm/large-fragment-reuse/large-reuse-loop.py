"""Native audit of a device-controlled producer/routed-constructor/consumer loop."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import warp as wp
from tests.test_active_delaunay import produce,consume
from tests.test_large_reuse import consume_reuse
from tests.test_tetrahedra import check_mesh,reference,tet_set
from warpower._active_delaunay import ActiveDelaunayWorkspace
from warpower._active_sites import prepare_mesh

@wp.kernel
def advance(phase:wp.array[int],condition:wp.array[int],total:int):
    phase[0]+=1
    condition[0]=int(phase[0]<total)

parser=argparse.ArgumentParser();parser.add_argument('--profile',action='store_true');parser.add_argument('--dot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
wp.init();device='cuda:0';n=4096
rng=np.random.default_rng(161);initial=rng.normal(size=(n,3)).astype(np.float32)
nearby=initial.copy();nearby[5]+=np.float32(1e-5)
unrelated=rng.normal(size=(n,3)).astype(np.float32)
invalid=initial.copy();invalid[4,0]=np.nan
counts=np.array([n,n,512,n,n+1,n,n,n],np.int32)
frames=np.stack([initial,nearby,initial,unrelated,initial,initial,invalid,initial]);frames[2,512:]=np.nan
source=wp.array(frames,dtype=wp.vec3,device=device);sizes=wp.array(counts,dtype=int,device=device)
points=wp.zeros(n,dtype=wp.vec3,device=device);prepared=wp.zeros_like(points);active=wp.zeros(1,dtype=int,device=device)
phase=wp.zeros(1,dtype=int,device=device);condition=wp.zeros(1,dtype=int,device=device)
w=ActiveDelaunayWorkspace(prepared,active_count=active);full=w._full
history=wp.empty((len(counts),len(w.tetrahedra),4),dtype=int,device=device);meta=wp.empty((len(counts),3),dtype=int,device=device);reuse=wp.empty((len(counts),4),dtype=int,device=device)
for module in [__name__,'tests.test_active_delaunay','tests.test_large_reuse','warpower._active_sites']:wp.load_module(module=module,device=device)
def step():
    wp.launch(produce,n,inputs=[source,sizes,phase,points,active],device=device)
    wp.launch(prepare_mesh,n,inputs=[points,active,prepared],device=device)
    w.run()
    wp.launch(consume,len(w.tetrahedra),inputs=[phase,w.tetrahedra,w.tetrahedra_count,w.error,w.selected_capacity,history,meta],device=device)
    wp.launch(consume_reuse,1,inputs=[phase,full.reuse_attempted,full.reuse_accepted,full._reuse_completion._certificate.proof_statistics,reuse],device=device)
    wp.launch(advance,1,inputs=[phase,condition,len(counts)],device=device)
with wp.ScopedCapture() as cap:
    phase.zero_();condition.fill_(1);w.clear_reuse()
    wp.capture_while(condition,step)
wp.synchronize()
args.dot.parent.mkdir(parents=True,exist_ok=True)
wp.capture_debug_dot_print(cap.graph,str(args.dot),verbose=True)
if args.profile:
    import torch
    cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(cap.graph);wp.synchronize()
if args.profile:cuda.cudaProfilerStop()
rows=meta.numpy();cells=history.numpy();validations=[];saved=dict(frames=frames,counts=counts,meta=rows,reuse= reuse.numpy())
for i,count in enumerate(counts):
    if i in [4,6]:
        assert rows[i,0]==0 and rows[i,1]!=0
        validations.append(dict(phase=i,invalid_empty=True))
    else:
        assert rows[i,0]>0 and rows[i,1]==0
        cloud=frames[i,:count];tets=cells[i,:rows[i,0]]
        assert tet_set(tets)==reference(cloud)
        check_mesh(cloud,tets)
        saved['tets_'+str(i)]=tets
        validations.append(dict(phase=i,independent_scipy_match=True,mesh_checks_passed=True,tetrahedra=len(tets)))
assert int(phase.numpy()[0])==len(counts)
assert saved['reuse'][1,0:2].tolist()==[1,1] and saved['reuse'][1,2]>0 and saved['reuse'][7,0]==0
np.savez_compressed(args.output.with_suffix('.npz'),**saved)
report=dict(scope='One graph launch with eight device-controlled producer/routed-Delaunay/consumer phases; includes full 4096-site reuse, smaller prefix, unrelated inputs, invalid count, nonfinite input and recovery. Setup, upload, final readback and independent checks are outside profiling.',warp=wp.__version__,device=wp.get_device(device).name,counts=counts.tolist(),meta=rows.tolist(),reuse=saved['reuse'].tolist(),route_calls=w.route_calls.numpy().tolist(),phase=int(phase.numpy()[0]),validation=validations,mesh_archive_sha256=hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest())
args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
