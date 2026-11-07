"""#190 observer: recompute every evaluation, measuring exact repeated inputs/outputs."""
import json,os
from pathlib import Path
import warp as wp
from warpower import _selective

wp.set_module_options({'enable_backward':False})
Base=_selective.SelectiveUpdate
instances=[]

@wp.kernel
def start(valid:wp.array[int],flags:wp.array[int]):
    flags[0]=valid[0]
    flags[1]=1

@wp.kernel
def compare_ints(current:wp.array[int],saved:wp.array[int],count:wp.array[int],use_count:bool,stride:int,flags:wp.array[int],column:int):
    i=wp.tid()
    if not use_count or i<stride*count[0]:
        if current[i]!=saved[i]:wp.atomic_and(flags,column,0)

@wp.kernel
def compare_longs(current:wp.array[wp.int64],saved:wp.array[wp.int64],count:wp.array[int],use_count:bool,flags:wp.array[int]):
    i=wp.tid()
    if not use_count or i<count[0]:
        if current[i]!=saved[i]:wp.atomic_and(flags,1,0)

@wp.kernel
def record(kind:int,previous:wp.array[int],flags:wp.array[int],count:wp.array[int],error:wp.array[int],valid:wp.array[int],calls:wp.array[int],records:wp.array2d[int],overflow:wp.array[int]):
    row=calls[0];calls[0]+=1
    if row<records.shape[0]:
        records[row,0]=kind;records[row,1]=previous[0]
        records[row,2]=flags[0];records[row,3]=flags[1]
        records[row,4]=count[0];records[row,5]=error[0]
    else:overflow[0]=1
    previous[0]=kind;valid[0]=int(error[0]==0)

class ObservedSelection(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        c=self.core;self._observer_kind=0
        with wp.ScopedDevice(self.device):
            self._obs_valid=wp.zeros(1,dtype=int);self._obs_previous=wp.zeros(1,dtype=int)
            self._obs_flags=wp.zeros(2,dtype=int);self._obs_calls=wp.zeros(1,dtype=int)
            self._obs_records=wp.zeros((4096,6),dtype=int);self._obs_overflow=wp.zeros(1,dtype=int)
            arrays=[(c.points.view(wp.int32).flatten(),False,1),(c.full_tetrahedra.flatten(),True,4),
                    (c.full_tetrahedra_count,False,1),(c.domain.points.view(wp.int32).flatten(),False,1),
                    (c.domain._indices,False,1),(c.domain.error,False,1),(c.mesh.error,False,1),
                    (c.boundary.roles,False,1)]
            for value in [c.active_count,c.material_recovery,getattr(c,'exterior_enabled',None),self.enabled]:
                if value is not None:arrays.append((value,False,1))
            self._obs_inputs=[(x,wp.empty_like(x),bounded,stride) for x,bounded,stride in arrays]
            self._obs_outputs=[(x,wp.empty_like(x),bounded) for x,bounded in [(self.keep,True),(self.quality.view(wp.int64),True),(self.error,False),(self.retained_count,False),(self.minimum.view(wp.int64),False)]]
            wp.load_module(module=__name__,device=self.device)
        instances.append(self)

    def reset(self):
        super().reset();self._obs_valid.zero_()

    def _select_all(self):
        self._obs_valid.zero_();super()._select_all()

    def _select(self):
        self._observer_kind=1
        try:super()._select()
        finally:self._observer_kind=0

    def _check(self):
        self._observer_kind=2
        try:super()._check()
        finally:self._observer_kind=0

    def evaluate(self):
        count=self.core.full_tetrahedra_count
        wp.launch(start,1,inputs=[self._obs_valid,self._obs_flags],device=self.device)
        for current,saved,bounded,stride in self._obs_inputs:
            wp.launch(compare_ints,len(current),inputs=[current,saved,count,bounded,stride,self._obs_flags,0],device=self.device)
        super().evaluate()  # Every ordinary evaluation still executes.
        for current,saved,bounded in self._obs_outputs:
            if current.dtype==wp.int64:
                wp.launch(compare_longs,len(current),inputs=[current,saved,count,bounded,self._obs_flags],device=self.device)
            else:
                wp.launch(compare_ints,len(current),inputs=[current,saved,count,bounded,1,self._obs_flags,1],device=self.device)
        wp.launch(record,1,inputs=[self._observer_kind,self._obs_previous,self._obs_flags,count,self.error,self._obs_valid,self._obs_calls,self._obs_records,self._obs_overflow],device=self.device)
        for current,saved,*_ in self._obs_inputs+self._obs_outputs:wp.copy(saved,current)

def install():_selective.SelectiveUpdate=ObservedSelection

if __name__=='__main__':
    install()
    from benchmarks.mesher import main
    main()
    rows=[]
    for v in instances:
        n=int(v._obs_calls.numpy()[0]);assert n<=len(v._obs_records) and v._obs_overflow.numpy()[0]==0
        a=v._obs_records.numpy()[:n];duplicates=a[:,2]==1
        assert (a[duplicates,3]==1).all()
        rows.append(dict(site_capacity=v.core.n,calls=n,select_calls=int((a[:,0]==1).sum()),check_calls=int((a[:,0]==2).sum()),identical_input_calls=int(duplicates.sum()),check_to_select_duplicates=int(((a[:,0]==1)&(a[:,1]==2)&duplicates).sum()),all_repeated_outputs_bitwise_equal=True,records=a.tolist()))
    Path(os.environ['WARPOWER_SELECTION_OBSERVER']).write_text(json.dumps(dict(scope='Observation only: every full material/quality evaluation still executes. Repeated inputs compare bitwise positions, live connectivity/count, domain points/topology/error, source error, roles and device mode/recovery/prefix controls. Current evaluation outputs are independently compared with the preceding saved evaluation. Counters span all benchmark replays; timing is instrumented and is not a speed measurement.',columns=['kind_select1_check2','previous_kind','inputs_bitwise_equal','outputs_bitwise_equal','tet_count','error'],instances=rows),indent=2)+'\n')
