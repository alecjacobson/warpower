"""Private #182 observer; always retain the current fragment construction path."""
import json,os
from pathlib import Path
import warp as wp
from warpower._reuse_delaunay import ReuseDelaunayWorkspace, completed

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def record(input_count:wp.array[int],output_count:wp.array[int],error:wp.array[int],
           local:wp.array[int],global_check:wp.array[int],statistics:wp.array[int],
           boundary:wp.array[int],records:wp.array2d[int],calls:wp.array[int],overflow:wp.array[int]):
    slot=calls[0]
    calls[0]+=1
    if slot>=records.shape[0]:
        overflow[0]=1
        return
    records[slot,0]=input_count[0]
    records[slot,1]=output_count[0]
    records[slot,2]=error[0]
    records[slot,3]=local[0]
    records[slot,4]=global_check[0]
    records[slot,5]=statistics[0]
    records[slot,6]=statistics[1]
    records[slot,7]=boundary[0]

instances=[]
def install():
    original_init=ReuseDelaunayWorkspace.__init__
    original_attempt=ReuseDelaunayWorkspace._attempt_reuse
    def init(self,*args,**kwargs):
        self._whole_observer_ready=False
        original_init(self,*args,**kwargs)
        assert self._certificate._local is not None  # This observer targets >=128 sites.
        with wp.ScopedDevice(self.device):
            self._whole_records=wp.zeros((2048,8),dtype=int)
            self._whole_calls=wp.zeros(1,dtype=int)
            self._whole_overflow=wp.zeros(1,dtype=int)
            wp.load_module(module=__name__,device=self.device)
            self._whole_observer_ready=True
            with wp.ScopedCapture() as capture:self._record_reuse()
            self._reuse_graph=capture.graph
        wp.synchronize_device(self.device)
        instances.append(self)
    def attempt(self):
        if not self._whole_observer_ready:
            return original_attempt(self)
        self._refresh_original()
        wp.copy(self._candidate,self._previous)
        wp.copy(self._candidate_count,self._previous_count)
        c=self._certificate
        c._local.zero_();c._global.zero_();c._boundary_count.zero_()
        c.run(self._original.tree)
        self._launch(record,1,[self._previous_count,self.tetrahedra_count,c.error,
            c._local,c._global,c.statistics,c._boundary_count,self._whole_records,
            self._whole_calls,self._whole_overflow])
        # This observer never publishes its certificate result. Complete the
        # unchanged fragment path, which replaces any candidate metadata on success.
        self._reuse_completion.run()
        self._launch(completed,1,[self._reuse_filter.error,self._reuse_completion.error,
            self._reuse_completion.tetrahedra_count,len(self.tetrahedra),self.reuse_error,self.reuse_accepted])
        wp.capture_if(self.reuse_accepted,self._publish_reuse)
    ReuseDelaunayWorkspace.__init__=init
    ReuseDelaunayWorkspace._attempt_reuse=attempt

if __name__=='__main__':
    install()
    from benchmarks.mesher import main
    main()
    rows=[]
    for instance in instances:
        count=int(instance._whole_calls.numpy()[0]);assert instance._whole_overflow.numpy()[0]==0 and count<=len(instance._whole_records)
        history=instance._whole_records.numpy()[:count]
        rows.append(dict(site_capacity=instance.n,calls=count,accepted=int((history[:,2]==0).sum()),records=history.tolist()))
    out=dict(scope='Observer only; every attempted fresh whole-mesh certificate is followed by the unchanged fragment path. Includes all graph executions; no timing claim.',columns=['previous_count','selected_count','certificate_error','local','global','sphere_queries','node_visits','boundary_count'],instances=rows)
    Path(os.environ['WARPOWER_WHOLE_OBSERVER']).write_text(json.dumps(out,indent=2)+'\n')
