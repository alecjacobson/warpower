"""Private, not-yet-evaluated bounded facet search. No production import."""
import warp as wp
from warpower import _filtered_predicates as fp, _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, halfspace, inside_halfspace, outside, sphere, symbolic_sphere
from warpower._mesh_topology import face_vertices

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def search_tree(
    points:wp.array[wp.vec3],
    tree:Tree,
    tets:wp.array2d[int],
    faces:wp.array[int],
    count:wp.array[int],
    candidates:wp.array[int],
    workers:int,
    error:wp.array[int],
    supported:wp.array[int],
    statistics:wp.array[int],
):
    tid=wp.tid()
    i,lane=tid//workers,tid%workers
    candidates[tid]=-1
    if i>=count[0]:
        return
    if lane==0:
        wp.atomic_add(statistics,wp.where(supported[i]!=0,0,1),1)
    if supported[i]!=0:
        return
    if workers!=32:
        wp.atomic_or(error,0,2)
        return
    face=face_vertices(tets,faces[i])
    a,b,c=points[face[0]],points[face[1]],points[face[2]]
    plane=halfspace(a,b,c)
    root=int(1)
    for level in range(5):
        if tree.count[root]>0:
            if (lane>>level)!=0:
                return
            break
        if ((lane>>level)&1)==0:
            root=tree.left[root]
        else:
            root=tree.right[root]
    stack=wp.zeros(shape=128,dtype=int)
    stack[0]=root
    size=int(1)
    best=int(-1)
    shape=Sphere()
    box=sb.Box()
    while size>0:
        size-=1
        node=stack[size]
        lo,hi=tree.lower[node],tree.upper[node]
        if inside_halfspace(plane,lo,hi):
            continue
        if best>=0:
            if sb.outside(box,lo,hi) or outside(shape,lo,hi):
                continue
        if tree.count[node]>0:
            for k in range(tree.start[node],tree.start[node]+tree.count[node]):
                q=tree.ids[k]
                if tree.owners[q]!=q or q==face[0] or q==face[1] or q==face[2]:
                    continue
                p=points[q]
                if inside_halfspace(plane,p,p):
                    continue
                if best>=0:
                    if sb.outside(box,p,p) or outside(shape,p,p):
                        continue
                orientation=fp.orient3d(a,b,c,p)
                if orientation.valid==0:
                    wp.atomic_or(error,0,2)
                    return
                if orientation.sign<=0:
                    continue
                replace=best<0
                if best>=0:
                    result=symbolic_sphere(points,wp.vec4i(face[0],face[1],face[2],best),q)
                    if result.valid==0 or result.sign==0:
                        wp.atomic_or(error,0,2)
                        return
                    replace=result.sign>0
                if replace:
                    best=q
                    shape=sphere(a,b,c,p)
                    box=sb.bounds(shape.origin,shape.nx,shape.ny,shape.nz,shape.determinant)
        else:
            if size>125:
                wp.atomic_or(error,0,2)
                return
            stack[size]=tree.left[node]
            stack[size+1]=tree.right[node]
            size+=2
    candidates[tid]=best
