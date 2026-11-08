"""Private bounded cell-local quality retries, with original global fallback."""
import types
import warp as wp
from warpower import _vtm
from representable_boundary_v1 import install as install_midpoint,rounded_boundary
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.func
def proposal_at(i:int,original:wp.array[wp.vec3],points:wp.array[wp.vec3],delta:wp.array[wp.vec3d],
                edge:wp.array[wp.float64],alpha:wp.array[wp.float64],maximum:wp.array[wp.float64],
                targets:wp.array[wp.vec3],limit:wp.float64,flags:wp.array[int],stats:wp.array[int],
                factor:wp.array[wp.float64],midpoint:int):
    requested=wp.vec3d(original[i])+(alpha[0]*factor[i])*delta[i]
    point=wp.vec3(requested);changed=int(0)
    if midpoint!=0:
        result=rounded_boundary(original[i],requested,targets[i],limit*edge[i])
        point=result.point;changed=result.changed
    points[i]=point;flags[i]=changed
    if changed!=0:
        wp.atomic_add(stats,0,changed);wp.atomic_add(stats,1,1)
    movement=wp.length(wp.vec3d(point)-wp.vec3d(original[i]))/edge[i]
    if not wp.isfinite(movement):movement=wp.float64(1e300)
    wp.atomic_max(maximum,0,movement)

@wp.kernel
def propose(original:wp.array[wp.vec3],points:wp.array[wp.vec3],delta:wp.array[wp.vec3d],
            edge:wp.array[wp.float64],alpha:wp.array[wp.float64],maximum:wp.array[wp.float64],
            targets:wp.array[wp.vec3],limit:wp.float64,flags:wp.array[int],stats:wp.array[int],
            factor:wp.array[wp.float64],midpoint:int):
    proposal_at(wp.tid(),original,points,delta,edge,alpha,maximum,targets,limit,flags,stats,factor,midpoint)

@wp.kernel
def propose_active(original:wp.array[wp.vec3],points:wp.array[wp.vec3],delta:wp.array[wp.vec3d],
                   edge:wp.array[wp.float64],alpha:wp.array[wp.float64],maximum:wp.array[wp.float64],
                   active:wp.array[int],targets:wp.array[wp.vec3],limit:wp.float64,flags:wp.array[int],
                   stats:wp.array[int],factor:wp.array[wp.float64],midpoint:int):
    i=wp.tid();flags[i]=0
    if i<active[0]:proposal_at(i,original,points,delta,edge,alpha,maximum,targets,limit,flags,stats,factor,midpoint)

@wp.kernel
def eligible(accepted:wp.array[int],rejections:wp.array[int],before:wp.array[int],reason:wp.array[int],
             used:wp.array[int],budget:int,ready:wp.array[int]):
    ready[0]=int(accepted[0]==0 and rejections[0]>before[0] and reason[0]==4 and used[0]<budget)

@wp.kernel
def mark(tets:wp.array2d[int],count:wp.array[int],keep:wp.array[int],quality:wp.array[wp.float64],
         previous:wp.array[wp.float64],original:wp.array[wp.vec3],points:wp.array[wp.vec3],
         mask:wp.array[int],marks:wp.array[int],any_mark:wp.array[int]):
    i=wp.tid()
    if i<count[0] and keep[i]!=0 and quality[i]<previous[0]*(wp.float64(1)-wp.float64(1e-10)):
        for k in range(4):
            v=tets[i,k]
            if mask[v]!=0 and wp.length_sq(wp.vec3d(points[v])-wp.vec3d(original[v]))>wp.float64(0):
                wp.atomic_or(marks,v,1);wp.atomic_or(any_mark,0,1)

@wp.kernel
def shrink(marks:wp.array[int],factor:wp.array[wp.float64],stats:wp.array[int]):
    i=wp.tid()
    if marks[i]!=0:
        factor[i]*=wp.float64(.5);wp.atomic_add(stats,1,1)

@wp.kernel
def local_finished(used:wp.array[int],rejected:wp.array[int],stats:wp.array[int]):
    used[0]+=1;rejected[0]+=1;stats[0]+=1

@wp.kernel
def ordinary_stats(accepted:wp.array[int],used:wp.array[int],stats:wp.array[int]):
    if accepted[0]==0:stats[2]+=1
    elif used[0]>0:stats[3]+=1


def install(core,*,midpoint=True,budget=12):
    assert core.selection is not None and budget>=0
    install_midpoint(core)
    core._local_factor=wp.ones(core.n,dtype=wp.float64,device=core.device)
    core._local_marks=wp.zeros(core.n,dtype=int,device=core.device)
    core._local_ready=wp.zeros(1,dtype=int,device=core.device)
    core._local_any=wp.zeros(1,dtype=int,device=core.device)
    core._local_used=wp.zeros(1,dtype=int,device=core.device)
    core._local_before=wp.zeros(1,dtype=int,device=core.device)
    core._local_stats=wp.zeros(4,dtype=int,device=core.device)
    core._propose=propose if core.active_count is None else propose_active
    core._propose_args += [core._local_factor,int(midpoint)]
    launch=core._launch
    def marked():
        c=core;s=c.selection;c._local_marks.zero_();c._local_any.zero_()
        launch(mark,len(c.full_tetrahedra),[c.full_tetrahedra,c.full_tetrahedra_count,s.keep,s.quality,
               s.previous_minimum,c._original,c.points,s.mask,c._local_marks,c._local_any])
        wp.copy(c._local_ready,c._local_any)
    def local_finish():
        launch(shrink,core.n,[core._local_marks,core._local_factor,core._local_stats])
        launch(local_finished,1,[core._local_used,core.rejected_steps,core._local_stats])
    def augmented(self,kernel,dim,args,*,block_dim=256):
        if kernel is self._propose:wp.copy(self._local_before,self.selection.rejections)
        if kernel is _vtm.finish_attempt:
            s=self.selection
            launch(eligible,1,[self._accepted,s.rejections,self._local_before,s.last_error,
                   self._local_used,budget,self._local_ready])
            wp.capture_if(self._local_ready,marked)
            def ordinary():
                # A geometry/objective failure, no moved limiting vertices, or
                # exhausted local budget resumes original global backtracking.
                self._local_factor.fill_(1)
                launch(ordinary_stats,1,[self._accepted,self._local_used,self._local_stats])
                launch(kernel,dim,args,block_dim=block_dim)
            wp.capture_if(self._local_ready,local_finish,ordinary)
        else:
            launch(kernel,dim,args,block_dim=block_dim)
            if kernel is self._directions:
                self._local_factor.fill_(1);self._local_used.zero_()
    core._launch=types.MethodType(augmented,core)
    wp.load_module(module=__name__,device=core.device)
    with wp.ScopedCapture(device=core.device) as capture:
        core._local_stats.zero_();core._record()
    core.graph=capture.graph
    return core
