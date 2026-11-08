from pathlib import Path
w=Path('.work/vtm');s=(w/'exterior_plane_bvh_v1.py').read_text()
s=s.replace('Private exact exterior proof: prune plane candidates with current exact BVH bounds.\n\nKeep the original complete rational-centroid ray scan and predicates unchanged.','Private exact exterior proof with conservative rational-centroid ray bounds.\n\nA pruned box cannot intersect the forward ray from the exact centroid. The leaf\npredicates remain exact; geometrically disjoint coplanar triangles need not cause\nthe exhaustive ray algorithm\'s conservative ambiguity. Qualify this difference.')
s=s.replace('from warpower._exterior_proof import centroid_ray','from warpower._exterior_proof import centroid_row, triangle, BallRow')
new='''@wp.func
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

'''
s=s.replace('@wp.func(inline=False)\ndef exterior',new+'@wp.func(inline=False)\ndef exterior')
p=w/'exterior_ray_bvh_v1.py';assert not p.exists();p.write_text(s)
s=(w/'run-tangential-plane-bvh-gallery-v1.py').read_text().replace('exterior_plane_bvh_v1','exterior_ray_bvh_v1')
p=w/'run-tangential-ray-bvh-gallery-v1.py';assert not p.exists();p.write_text(s)
s=(w/'run-tangential-plane-bvh-pilot-v1.py').read_text().replace('tangential-plane-bvh','tangential-ray-bvh')
p=w/'run-tangential-ray-bvh-pilot-v1.py';assert not p.exists();p.write_text(s)
s=(w/'compare-exterior-plane-bvh-v1.py').read_text().replace('tangential-plane-bvh','tangential-ray-bvh').replace('exterior-plane-bvh','exterior-ray-bvh')
p=w/'compare-exterior-ray-bvh-v1.py';assert not p.exists();p.write_text(s)
s=(w/'test_exterior_plane_bvh_v1.py').read_text().replace('exterior_plane_bvh_v1','exterior_ray_bvh_v1')
p=w/'test_exterior_ray_bvh_v1.py';assert not p.exists();p.write_text(s)
