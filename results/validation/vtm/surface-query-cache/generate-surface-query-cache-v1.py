"""Build private exact-query memoization; original query paths remain the miss fallback."""
from pathlib import Path
w=Path('.work/vtm');p=w/'surface_query_cache_kernels_v1.py';assert not p.exists()
s=(w/'surface_query_reuse_kernels_v2.py').read_text()
s=s.replace('SurfaceTree, closest_point, closest_triangle, material_side','SurfaceTree, Closest, closest_point, closest_triangle, material_side')
end=s.index('@wp.kernel\ndef observe_full')
head=s[:end]
helpers='''@wp.func
def hash_center(center:wp.vec3d,mask:int):
    h=wp.cast(center[0],wp.uint64)*wp.uint64(0x9E3779B97F4A7C15)
    h=h^(wp.cast(center[1],wp.uint64)*wp.uint64(0xBF58476D1CE4E5B9))
    h=h^(wp.cast(center[2],wp.uint64)*wp.uint64(0x94D049BB133111EB))
    h=(h^(h>>wp.uint64(30)))*wp.uint64(0xBF58476D1CE4E5B9)
    h=(h^(h>>wp.uint64(27)))*wp.uint64(0x94D049BB133111EB)
    h=h^(h>>wp.uint64(31))
    return int(h&wp.uint64(mask))

@wp.func
def same_center(a:wp.vec3d,b:wp.vec3d):
    return wp.cast(a[0],wp.uint64)==wp.cast(b[0],wp.uint64) and wp.cast(a[1],wp.uint64)==wp.cast(b[1],wp.uint64) and wp.cast(a[2],wp.uint64)==wp.cast(b[2],wp.uint64)

@wp.func
def lookup(center:wp.vec3d,previous:QueryState,table:wp.array[int]):
    slot=hash_center(center,table.shape[0]-1)
    for _ in range(128):
        j=table[slot]
        if j<0:return -1
        if same_center(center,previous.centers[j]):return j
        slot=(slot+1)&(table.shape[0]-1)
    return -1

@wp.func(inline=True)
def cached_side(tree:SurfaceTree,center:wp.vec3d,previous:QueryState,table:wp.array[int]):
    j=lookup(center,previous,table)
    if j>=0 and previous.rays[j]!=0:return previous.sides[j]
    return material_side(tree,center)

@wp.func(inline=True)
def cached_nearest(tree:SurfaceTree,center:wp.vec3d,limit:wp.float64,previous:QueryState,table:wp.array[int]):
    j=lookup(center,previous,table)
    if j>=0 and previous.nearest[j]!=0 and wp.cast(limit,wp.uint64)==wp.cast(previous.limits[j],wp.uint64):
        result=Closest()
        result.point=previous.hit_points[j]
        result.face=previous.faces[j]
        result.distance2=previous.distances[j]
        return result
    return closest_point(tree,center,limit)

@wp.kernel
def build_cache(previous:QueryState,table:wp.array[int],error:wp.array[int]):
    i=wp.tid()
    if error[0]!=0 or previous.valid[i]==0:return
    center=previous.centers[i];mask=table.shape[0]-1;slot=hash_center(center,mask)
    for _ in range(128):
        old=wp.atomic_cas(table,slot,-1,i)
        if old<0:return
        if same_center(center,previous.centers[old]):
            wp.atomic_min(table,slot,i)
            return
        slot=(slot+1)&mask
    # Uninserted keys remain misses, so bounded probing only reduces reuse.

'''
body=s[end:]
body=body.replace('observe_full','classify_cached').replace('observe_retained','retained_cached').replace('observe_selection_center','selection_center_cached').replace('observe_selection','selection_cached')
body=body.replace('    probe: QueryState,\n','    probe: QueryState,\n    previous: QueryState,\n    table: wp.array[int],\n')
body=body.replace('probe: QueryState, slot: int\n):','probe: QueryState, slot: int, previous: QueryState, table: wp.array[int]\n):')
body=body.replace('selection_center_cached(tree, center, radius, tolerance, probe, i)','selection_center_cached(tree, center, radius, tolerance, probe, i, previous, table)')
body=body.replace('material_side(tree, center)','cached_side(tree, center, previous, table)')
body=body.replace('closest_point(tree, center, wp.float64(1.0e150))','cached_nearest(tree, center, wp.float64(1.0e150), previous, table)')
body=body.replace('closest_point(tree, center, limit)','cached_nearest(tree, center, limit, previous, table)')
p.write_text(head+helpers+body)
