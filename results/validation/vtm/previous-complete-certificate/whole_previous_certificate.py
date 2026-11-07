"""Private #182: freshly certify a complete prior mesh before fragment filtering."""
import warp as wp
from warpower._reuse_delaunay import ReuseDelaunayWorkspace, completed

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def classify_whole(certificate_error:wp.array[int],count:wp.array[int],capacity:int,
                   whole:wp.array[int],accepted:wp.array[int],reuse_error:wp.array[int],
                   error:wp.array[int],recovered:wp.array[int],fragment_ready:wp.array[int]):
    code=certificate_error[0]
    if count[0]<1 or count[0]>capacity:
        code |= 16
    whole[0]=1
    whole[1]=int(code==0)
    whole[2]=code
    accepted[0]=int(code==0)
    fragment_ready[0]=int(code!=0)
    if code==0:
        reuse_error[0]=0
        error[0]=0
        recovered[0]=1

instances=[]
def install():
    original_init=ReuseDelaunayWorkspace.__init__
    original_record=ReuseDelaunayWorkspace._record_reuse
    original_attempt=ReuseDelaunayWorkspace._attempt_reuse
    def init(self,*args,**kwargs):
        self._whole_enabled=False
        original_init(self,*args,**kwargs)
        with wp.ScopedDevice(self.device):
            self._whole_status=wp.zeros(3,dtype=int)
            self._whole_accepted=self._whole_status[1:2]
            wp.load_module(module=__name__,device=self.device)
            self._whole_enabled=True
            with wp.ScopedCapture() as capture:self._record_reuse()
            self._reuse_graph=capture.graph
        wp.synchronize_device(self.device)
        instances.append(self)
    def record(self):
        if self._whole_enabled:self._whole_status.zero_()
        original_record(self)
    def publish(self):
        # The ordinary certificate already owns the published cells and their
        # current face index. Its input candidate must match that compacted output.
        wp.copy(self._candidate,self.tetrahedra)
        wp.copy(self._candidate_count,self.tetrahedra_count)
        # No fragment completion ran on this branch. Do not expose stale proof
        # consumption counters from an older call as current observations.
        self._reuse_completion._certificate.proof_statistics.zero_()
    def fragment(self):
        self._reuse_completion.run()
        self._launch(completed,1,[self._reuse_filter.error,self._reuse_completion.error,
            self._reuse_completion.tetrahedra_count,len(self.tetrahedra),self.reuse_error,self.reuse_accepted])
        wp.capture_if(self.reuse_accepted,self._publish_reuse)
    def attempt(self):
        if not self._whole_enabled:return original_attempt(self)
        self._refresh_original()
        wp.copy(self._candidate,self._previous)
        wp.copy(self._candidate_count,self._previous_count)
        self._certificate.run(self._original.tree)
        self._launch(classify_whole,1,[self.certificate_error,self.tetrahedra_count,
            len(self.tetrahedra),self._whole_status,self.reuse_accepted,self.reuse_error,
            self.error,self.recovered,self._reuse_ready])
        wp.capture_if(self._whole_accepted,lambda:publish(self))
        wp.capture_if(self._reuse_ready,lambda:fragment(self))
    ReuseDelaunayWorkspace.__init__=init
    ReuseDelaunayWorkspace._record_reuse=record
    ReuseDelaunayWorkspace._attempt_reuse=attempt
