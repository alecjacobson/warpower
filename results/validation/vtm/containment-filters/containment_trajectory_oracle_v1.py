"""Compare legacy and filtered whole-cell decisions on every eligible live recovery cell."""
import warp as wp
from warpower import _containment as candidate
import containment_exact_baseline_v1 as original
from warpower._surface_bvh import SurfaceTree
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def audited_add(tree: SurfaceTree,points: wp.array[wp.vec3],tets: wp.array2d[int],count: wp.array[int],keep: wp.array[int],indices: wp.array[int],table: wp.array[int],occurrences: wp.array[int],balance: wp.array[int],additions: wp.array[int],total: wp.array[int],stats: wp.array[int]):
 t=wp.tid()
 if t>=count[0] or keep[t]!=0:return
 ids=wp.vec4i(tets[t,0],tets[t,1],tets[t,2],tets[t,3]);touches=bool(False)
 for i in range(4):
  for j in range(i+1,4):
   if original.bad_edge(ids[i],ids[j],indices,table,occurrences,balance):touches=True
 if touches:
  before=original.contained(tree,points,ids);after=candidate.contained(tree,points,ids)
  wp.atomic_add(stats,0,1);wp.atomic_add(stats,1,int(before!=after));wp.atomic_add(stats,2,before);wp.atomic_add(stats,3,after)
  if after!=0:
   keep[t]=1;wp.atomic_add(additions,0,1);wp.atomic_add(total,0,1)


def install():
 stats=wp.zeros(4,dtype=int);wp.load_module(module=__name__)
 launch=wp.launch
 def checked(kernel,dim,*args,**kwargs):
  if kernel is candidate.add_contained:
   assert not args and 'inputs' in kwargs
   return launch(audited_add,dim,**(kwargs|{'inputs':[*kwargs['inputs'],stats]}))
  return launch(kernel,dim,*args,**kwargs)
 wp.launch=checked
 return stats
