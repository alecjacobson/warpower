"""One changing-prefix producer/local optimizer/consumer loop, including failure and recovery."""
import argparse,json
from pathlib import Path
import numpy as np
import warp as wp
from warpower import SurfaceDomain,SizingField,_vtm
from tests.test_domain import cube
from tests.test_vtm_core import consume

@wp.kernel
def produce(source:wp.array[wp.vec3],points:wp.array[wp.vec3],phase:wp.array[int],active:wp.array[int]):
 i=wp.tid();points[i]=source[i]
 if i==0:
  active[0]=64
  if phase[0]==1:active[0]=0

@wp.kernel
def finish(phase:wp.array[int],condition:wp.array[int],error:wp.array[int],count:wp.array[int],
           iterations:wp.array[int],rejected:wp.array[int],statistics:wp.array[int],volume:wp.array[wp.float64],
           reference:wp.array[wp.float64],meta:wp.array2d[int],metrics:wp.array2d[wp.float64]):
 row=phase[0];meta[row,0]=error[0];meta[row,1]=count[0];meta[row,2]=iterations[0];meta[row,3]=rejected[0]
 meta[row,4]=statistics[0];meta[row,5]=statistics[1];metrics[row,0]=volume[0];metrics[row,1]=reference[0]
 phase[0]+=1;condition[0]=int(phase[0]<3)

p=argparse.ArgumentParser();p.add_argument('--profile',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--dot',type=Path,required=True);a=p.parse_args()
domain=SurfaceDomain(*cube(),surface_subdivisions=6)
source=np.full((96,3),np.nan,np.float32);source[:64]=np.random.default_rng(174).uniform(-.9,.9,(64,3))
source=wp.array(source,dtype=wp.vec3);points=wp.clone(source);active=wp.array([64],dtype=int)
field=SizingField(np.array([[0,0,0]],np.float32),wp.array([.3],dtype=float),gradation=.4)
c=_vtm.VTMWorkspace(domain,points,active_count=active,sizing=field,max_iterations=3,selective_threshold=.3,local_quality_retries=4)
phase=wp.zeros(1,dtype=int);condition=wp.zeros(1,dtype=int);volume=wp.zeros(1,dtype=wp.float64)
meta=wp.zeros((3,6),dtype=int);metrics=wp.zeros((3,2),dtype=wp.float64)
wp.load_module(module=__name__,device=domain.device);wp.load_module(module='tests.test_vtm_core',device=domain.device)
def step():
 wp.launch(produce,len(points),inputs=[source,points,phase,active]);domain.refit();field.refit();c.run();volume.zero_()
 wp.launch(consume,len(c.full_tetrahedra),inputs=[points,c.full_tetrahedra,c.full_tetrahedra_count,c.error,volume])
 wp.launch(finish,1,inputs=[phase,condition,c.error,c.full_tetrahedra_count,c.iterations,c.rejected_steps,c.local_backtracking.statistics,volume,c.volume,meta,metrics])
with wp.ScopedCapture() as capture:
 phase.zero_();condition.fill_(1);wp.capture_while(condition,step)
wp.synchronize();wp.capture_debug_dot_print(capture.graph,str(a.dot),verbose=True)
if a.profile:
 import torch
 cuda=torch.cuda.cudart();cuda.cudaProfilerStart()
wp.capture_launch(capture.graph);wp.synchronize()
if a.profile:cuda.cudaProfilerStop()
r=dict(scope='First native launch of one captured three-phase device producer/domain+sizing refit/local-selective optimizer/volume-consumer loop. Active prefix64, invalid prefix0, prefix64 recovery; reserved inactive NaN tail. Setup and final readback excluded.',metadata=meta.numpy().tolist(),columns=['error','count','accepted_iterations','rejected_attempts','local_retries','blocked_vertex_events'],metrics=metrics.numpy().tolist(),metric_columns=['consumed_volume','core_volume'],phase=int(phase.numpy()[0]))
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
assert r['phase']==3 and [x[0] for x in r['metadata']]==[0,8,0]
assert r['metadata'][1][1]==0 and r['metadata'][1][4:]==[0,0]
np.testing.assert_allclose(metrics.numpy()[:,0],metrics.numpy()[:,1],rtol=1e-12)
np.testing.assert_array_equal(points.numpy()[64:].view(np.uint32),source.numpy()[64:].view(np.uint32))
