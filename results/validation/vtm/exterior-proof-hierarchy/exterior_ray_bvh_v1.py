"""Private exact exterior proof with conservative rational-centroid ray bounds.

A pruned box cannot intersect the forward ray from the exact centroid. The leaf
predicates remain exact; geometrically disjoint coplanar triangles need not cause
the exhaustive ray algorithm's conservative ambiguity. Qualify this difference.
"""
import warp as wp
from warpower import _filtered_predicates as fp, _proven_exterior as production
from warpower._exterior_proof import centroid_row, triangle, BallRow
from warpower._surface_bvh import SurfaceTree

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.func
def overlaps(a:wp.vec3,b:wp.vec3,lo:wp.vec3,hi:wp.vec3):
    return not (b[0]<lo[0] or b[1]<lo[1] or b[2]<lo[2]
                or a[0]>hi[0] or a[1]>hi[1] or a[2]>hi[2])

@wp.func
def axis(row:BallRow,k:int):
    if k==0:return row.x
    if k==1:return row.y
    return row.z

@wp.func
def negate(value:fp.Ball):
    result=fp.Ball();result.value=-value.value;result.error=value.error
    return result

@wp.func
def centroid_box(lo:wp.vec3,hi:wp.vec3,p0:wp.vec3,p1:wp.vec3,p2:wp.vec3,p3:wp.vec3,direction:wp.vec3):
    # Bounds for 4*box-sum(p_i), preserving the rational centroid. All direction
    # components must be nonzero (the three fixed directions below satisfy this).
    low=centroid_row(lo,p0,p1,p2,p3);high=centroid_row(hi,p0,p1,p2,p3)
    for j in range(3):
        leaving=axis(high,j)
        if direction[j]<0.0:leaving=negate(axis(low,j))
        if fp.certain(leaving) and leaving.value<wp.float64(0.0):return False
        for i in range(3):
            entering=axis(low,i)
            if direction[i]<0.0:entering=negate(axis(high,i))
            # Positive denominators: entry_i > exit_j proves disjoint slabs.
            gap=fp.minus(fp.times(entering,fp.scalar(wp.abs(wp.float64(direction[j])))),
                         fp.times(leaving,fp.scalar(wp.abs(wp.float64(direction[i])))))
            if fp.certain(gap) and gap.value>wp.float64(0.0):return False
    return True

@wp.func(inline=False)
def centroid_ray(tree:SurfaceTree,p0:wp.vec3,p1:wp.vec3,p2:wp.vec3,p3:wp.vec3,direction:wp.vec3):
    count=int(0);node=int(1)
    while node>0:
        if not centroid_box(tree.lower[node],tree.upper[node],p0,p1,p2,p3,direction):
            node=tree.escape[node]
        elif node<tree.leaf_base:
            node=tree.right[node]
        else:
            first=4*(node-tree.leaf_base)
            for j in range(first,wp.min(first+4,tree.n)):
                f=tree.ids[j]
                q0=wp.vec3(tree.points[tree.indices[3*f]])
                q1=wp.vec3(tree.points[tree.indices[3*f+1]])
                q2=wp.vec3(tree.points[tree.indices[3*f+2]])
                result=triangle(p0,p1,p2,p3,q0,q1,q2,direction)
                if result[1]!=0:return result
                count+=result[0]
            node=tree.escape[node]
    return wp.vec2i(count,0)

@wp.func(inline=False)
def exterior(tree:SurfaceTree,points:wp.array[wp.vec3],ids:wp.vec4i):
    p0=wp.vec3(points[ids[0]]);p1=wp.vec3(points[ids[1]])
    p2=wp.vec3(points[ids[2]]);p3=wp.vec3(points[ids[3]])
    orientation=fp.orient3d(p0,p1,p2,p3)
    if orientation.valid==0 or orientation.sign<=0:return 0
    lo=wp.min(wp.min(p0,p1),wp.min(p2,p3));hi=wp.max(wp.max(p0,p1),wp.max(p2,p3))
    node=int(1)
    while node>0:
        if not overlaps(tree.lower[node],tree.upper[node],lo,hi):
            node=tree.escape[node]
        elif node<tree.leaf_base:
            node=tree.right[node]
        else:
            first=4*(node-tree.leaf_base)
            for j in range(first,wp.min(first+4,tree.n)):
                f=tree.ids[j]
                a=wp.vec3(tree.points[tree.indices[3*f]])
                b=wp.vec3(tree.points[tree.indices[3*f+1]])
                c=wp.vec3(tree.points[tree.indices[3*f+2]])
                qlo=wp.min(a,wp.min(b,c));qhi=wp.max(a,wp.max(b,c))
                if not overlaps(qlo,qhi,lo,hi):continue
                positive=bool(False);negative=bool(False)
                for k in range(4):
                    side=fp.orient3d(a,b,c,wp.vec3(points[ids[k]]))
                    if side.valid==0:return 0
                    positive=positive or side.sign>0
                    negative=negative or side.sign<0
                if positive and negative:return 0
            node=tree.escape[node]
    r=centroid_ray(tree,p0,p1,p2,p3,wp.vec3(11.0,23.0,37.0))
    if r[1]==1:r=centroid_ray(tree,p0,p1,p2,p3,wp.vec3(-17.0,29.0,43.0))
    if r[1]==1:r=centroid_ray(tree,p0,p1,p2,p3,wp.vec3(31.0,-47.0,19.0))
    return int(r[1]==0 and r[0]==0)

@wp.kernel
def remove_proved(tree:SurfaceTree,points:wp.array[wp.vec3],tets:wp.array2d[int],count:wp.array[int],
                  selected:wp.array[int],distance:wp.array[wp.float64],removed:wp.array[int]):
    i=wp.tid()
    if i<count[0] and selected[i]!=0 and distance[i]>wp.float64(0.0):
        ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
        if exterior(tree,points,ids)!=0:
            selected[i]=0;wp.atomic_add(removed,0,1)

def install(*,block_dim=256):
    old=production.ProvenExteriorRecovery.launch
    def launch(self,kernel,dim,args):
        if kernel is production.remove_proved:
            wp.launch(remove_proved,dim,inputs=args,device=self.device,block_dim=block_dim)
        else:old(self,kernel,dim,args)
    production.ProvenExteriorRecovery.launch=launch
