"""UNRUN private #130 instrumentation; never treats old cells as certified.

Records input movement around unchanged production reconstruction. Additional
kernels/copies make these runs diagnostic, not performance measurements.
"""
import json
import os
from pathlib import Path

import warp as wp
from warpower._reuse_delaunay import ReuseDelaunayWorkspace
from warpower._mesher import VariationalMeshingWorkspace

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def begin_history(count:wp.array[int], index:wp.array[int], history:wp.array2d[int]):
    row=index[0]
    if row<history.shape[0]:
        history[row,0]=count[0]
        history[row,1]=0
        history[row,2]=0


@wp.kernel
def changed_points(points:wp.array[wp.vec3], previous:wp.array[wp.vec3], count:wp.array[int], index:wp.array[int], history:wp.array2d[int]):
    i=wp.tid()
    row=index[0]
    if count[0]<1 or row>=history.shape[0]:
        return
    p=wp.vec3(points[i])
    q=wp.vec3(previous[i])
    if p[0]!=q[0] or p[1]!=q[1] or p[2]!=q[2]:
        wp.atomic_add(history,row,1,1)


@wp.kernel
def unchanged_corners(points:wp.array[wp.vec3], previous:wp.array[wp.vec3], tets:wp.array2d[int], count:wp.array[int], index:wp.array[int], history:wp.array2d[int]):
    i=wp.tid()
    row=index[0]
    if i>=count[0] or row>=history.shape[0]:
        return
    same=True
    for k in range(4):
        j=tets[i,k]
        p=wp.vec3(points[j])
        q=wp.vec3(previous[j])
        same=same and p[0]==q[0] and p[1]==q[1] and p[2]==q[2]
    if same:
        wp.atomic_add(history,row,2,1)


@wp.kernel
def finish_history(seeds:wp.array[int], accepted:wp.array[int], error:wp.array[int], index:wp.array[int], history:wp.array2d[int]):
    row=index[0]
    if row<history.shape[0]:
        history[row,3]=seeds[0]
        history[row,4]=accepted[0]
        history[row,5]=error[0]
    index[0]=row+1


instances=[]
original_init=ReuseDelaunayWorkspace.__init__
original_record=ReuseDelaunayWorkspace._record_reuse
original_solve=VariationalMeshingWorkspace._solve


def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    with wp.ScopedDevice(self.device):
        self._probe_previous=wp.empty_like(self.points)
        self._probe_index=wp.zeros(1,dtype=int)
        self._probe_history=wp.zeros((4096,6),dtype=int)
        wp.load_module(module=__name__,device=self.device)
        with wp.ScopedCapture() as cap:
            self._record_reuse()
        self._reuse_graph=cap.graph
    wp.synchronize_device(self.device)
    instances.append(self)


def record(self):
    if not hasattr(self,'_probe_history'):
        original_record(self)
        return
    common=[self._previous_count,self._probe_index,self._probe_history]
    wp.launch(begin_history,1,inputs=common,device=self.device)
    wp.launch(changed_points,self.n,inputs=[self.points,self._probe_previous]+common,device=self.device)
    wp.launch(unchanged_corners,len(self._previous),inputs=[self.points,self._probe_previous,self._previous]+common,device=self.device)
    original_record(self)
    wp.copy(self._probe_previous,self.points)
    wp.launch(finish_history,1,inputs=[self.reuse_seeds,self.reuse_accepted,self.error,self._probe_index,self._probe_history],device=self.device)


def solve(self):
    for instance in instances:
        instance._probe_index.zero_()
        instance._probe_history.zero_()
    original_solve(self)


ReuseDelaunayWorkspace.__init__=initialize
ReuseDelaunayWorkspace._record_reuse=record
VariationalMeshingWorkspace._solve=solve
if os.environ.get('WARPOWER_PROBE_GALLERY'):
    from examples.vtm_gallery import main
else:
    from benchmarks.mesher import main
main()
result=[]
for instance in instances:
    count=int(instance._probe_index.numpy()[0]);assert count<=4096
    rows=instance._probe_history.numpy()[:count]
    valid=rows[rows[:,0]>0]
    result.append(dict(capacity=instance.n,calls=count,cached_calls=len(valid),
                       unchanged_input_calls=int((valid[:,1]==0).sum()),
                       single_moved_site_calls=int((valid[:,1]==1).sum()),
                       previously_certified_cells=int(valid[:,0].sum()),
                       cells_with_unchanged_corners=int(valid[:,2].sum()),
                       columns=['previous_cells','changed_points','unchanged_corner_cells','retained_seeds','reuse_accepted','final_error'],history=rows.tolist()))
path=Path(os.environ.get('WARPOWER_PROBE_OUTPUT','.work/vtm/reuse-inputs-profile.json'))
path.write_text(json.dumps(dict(scope='Diagnostic instrumentation; unchanged-corner counts alone do not establish reusable exact certificates or canonical-owner validity.',instances=result),indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k not in ('history','columns')} for row in result]),flush=True)
