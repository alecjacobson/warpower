"""Same changing-input routed construction on either recorded runtime."""
import argparse,faulthandler,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
from benchmarks.memcheck import configure
from tests.test_active_delaunay import advance,consume,produce
from tests.test_tetrahedra import check_mesh,reference,tet_set
from warpower._active_delaunay import ActiveDelaunayWorkspace
from warpower._active_sites import prepare_mesh
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
configure(stack_bytes=131072,disable_mempool=True)
n=4096;device='cuda:0';rng=np.random.default_rng(161);initial=rng.normal(size=(n,3)).astype(np.float32)
nearby=initial.copy();nearby[5]+=np.float32(1e-5)
unrelated=rng.normal(size=(n,3)).astype(np.float32);invalid=initial.copy();invalid[4,0]=np.nan
counts=[n,n,512,n,n+1,n,n,n]
frames=np.stack([initial,nearby,initial,unrelated,initial,initial,invalid,initial]);frames[2,512:]=np.nan
source=wp.array(frames,dtype=wp.vec3,device=device);sizes=wp.array(counts,dtype=int,device=device)
phase=wp.zeros(1,dtype=int,device=device);points=wp.zeros(n,dtype=wp.vec3,device=device);prepared=wp.zeros_like(points);active=wp.zeros(1,dtype=int,device=device)
w=ActiveDelaunayWorkspace(prepared,active_count=active)
history=wp.empty((len(counts),len(w.tetrahedra),4),dtype=int,device=device);meta=wp.empty((len(counts),3),dtype=int,device=device)
for module in ['tests.test_active_delaunay','warpower._active_sites']:wp.load_module(module=module,device=device)
stream=wp.Stream(device)
with wp.ScopedStream(stream),wp.ScopedCapture(stream=stream) as cap:
 wp.launch(produce,n,inputs=[source,sizes,phase,points,active])
 wp.launch(prepare_mesh,n,inputs=[points,active,prepared])
 w.run(stream)
 wp.launch(consume,len(w.tetrahedra),inputs=[phase,w.tetrahedra,w.tetrahedra_count,w.error,w.selected_capacity,history,meta])
 wp.launch(advance,1,inputs=[phase])
for i in range(len(counts)):
 print('launch',i,flush=True);wp.capture_launch(cap.graph,stream=stream)
wp.synchronize_stream(stream)
rows,cells=meta.numpy(),history.numpy()
for i,count in enumerate(counts):
 if i in [4,6]:assert rows[i,0]==0 and rows[i,1]!=0
 else:
  assert rows[i,0]>0 and rows[i,1]==0
  p=frames[i,:count];t=cells[i,:rows[i,0]];assert tet_set(t)==reference(p);check_mesh(p,t)
result=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),full_constructor=type(w._full).__name__,scope='Eight changing-input routed-constructor replays with device producers/consumers, same seed161 frames as failed test; omits new reuse-only metadata/assertions so preceding runtime can run the identical control.',counts=counts,meta=rows.tolist(),independent_all_valid_phases=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True);faulthandler.cancel_dump_traceback_later()
