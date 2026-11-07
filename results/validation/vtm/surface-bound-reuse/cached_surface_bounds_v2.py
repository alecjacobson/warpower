"""Private exact cached bounds, with the near child kept in registers."""
import warp as wp
from warpower._surface_bvh import SurfaceTree,Closest,box_distance,closest_triangle,closest_point
wp.set_module_options({'enable_backward':False,'fast_math':False})


@wp.func(inline=True)
def cached_closest(tree:SurfaceTree,p:wp.vec3d,limit:wp.float64):
    result=Closest()
    result.point=wp.vec3d(0.0)
    result.face=-1
    result.distance2=limit*limit
    stack=wp.zeros(shape=128,dtype=int)
    bounds=wp.zeros(shape=128,dtype=wp.float64)
    size=int(0)
    node=int(1)
    bound=box_distance(tree,1,p)
    while node>0:
        # Negated >= preserves the original behavior for NaN bounds/limits.
        if not (bound>=result.distance2):
            if node>=tree.leaf_base:
                first=4*(node-tree.leaf_base)
                for j in range(first,wp.min(first+4,tree.n)):
                    f=tree.ids[j]
                    a=wp.vec3d(tree.points[tree.indices[3*f]])
                    b=wp.vec3d(tree.points[tree.indices[3*f+1]])
                    c=wp.vec3d(tree.points[tree.indices[3*f+2]])
                    q=closest_triangle(p,a,b,c)
                    distance=wp.length_sq(p-q)
                    if distance<result.distance2:
                        result.point=q
                        result.face=f
                        result.distance2=distance
            else:
                left,right=int(tree.left[node]),int(tree.right[node])
                left_distance=box_distance(tree,left,p)
                right_distance=box_distance(tree,right,p)
                if left_distance<right_distance:
                    left,right=right,left
                    left_distance,right_distance=right_distance,left_distance
                stack[size]=left
                bounds[size]=left_distance
                size+=1
                node=right
                bound=right_distance
                continue
        if size==0:
            break
        size-=1
        node=stack[size]
        bound=bounds[size]
    return result


@wp.kernel
def reference(tree:SurfaceTree,points:wp.array[wp.vec3d],limits:wp.array[wp.float64],hits:wp.array[Closest]):
    i=wp.tid();hits[i]=closest_point(tree,points[i],limits[i])


@wp.kernel
def candidate(tree:SurfaceTree,points:wp.array[wp.vec3d],limits:wp.array[wp.float64],hits:wp.array[Closest]):
    i=wp.tid();hits[i]=cached_closest(tree,points[i],limits[i])
