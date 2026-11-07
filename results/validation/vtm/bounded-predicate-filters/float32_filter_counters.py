import warp as wp
import bounded_float32_filters as af
from warpower import _analytic_predicates as old
from warpower import _filtered_predicates as fp
from warpower._mesh_topology import face_vertices

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})

@wp.kernel
def counted_search_float32(
    points: wp.array[wp.vec3],
    owners: wp.array[int],
    tets: wp.array2d[int],
    faces: wp.array[int],
    count: wp.array[int],
    candidates: wp.array[int],
    workers: int,
    error: wp.array[int],
    supported: wp.array[int],
    statistics: wp.array[int],
    counters: wp.array[wp.int64],
):
    tid = wp.tid()
    i, lane = tid // workers, tid % workers
    candidates[tid] = -1
    if i >= count[0]:
        return
    if lane == 0:
        wp.atomic_add(statistics, wp.where(supported[i] != 0, 0, 1), 1)
    if supported[i] != 0:
        return
    face = face_vertices(tets, faces[i])
    a, b, c = points[face[0]], points[face[1]], points[face[2]]
    oa, of, od, sa, sf, sd = int(0), int(0), int(0), int(0), int(0), int(0)
    best = int(-1)
    for q in range(lane, points.shape[0], workers):
        if owners[q] != q or q == face[0] or q == face[1] or q == face[2]:
            continue
        oa += 1
        value = af.orientation_filter(a, b, c, points[q])
        orientation = fp.predicate(value)
        if not old.strict(value):
            of += 1
            value = old.orientation_filter(a, b, c, points[q])
            orientation = fp.predicate(value)
            if not old.strict(value):
                od += 1
                orientation = fp.orient3d(a, b, c, points[q])
        if orientation.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_add(counters, 6, wp.int64(1))
            return
        if orientation.sign <= 0:
            continue
        if best < 0:
            best = q
        else:
            sa += 1
            ids = wp.vec4i(face[0], face[1], face[2], best)
            value = af.sphere_filter(a, b, c, points[best], points[q])
            result = fp.predicate(value)
            if not old.strict(value):
                sf += 1
                value = old.sphere_filter(a, b, c, points[best], points[q])
                result = fp.predicate(value)
                if not old.strict(value):
                    sd += 1
                    result = old.legacy_symbolic_sphere(points, ids, q)
            if result.valid == 0 or result.sign == 0:
                wp.atomic_or(error, 0, 2)
                wp.atomic_add(counters, 6, wp.int64(1))
                return
            if result.sign > 0:
                best = q
    candidates[tid] = best


    wp.atomic_add(counters, 0, wp.int64(oa))
    wp.atomic_add(counters, 1, wp.int64(of))
    wp.atomic_add(counters, 2, wp.int64(od))
    wp.atomic_add(counters, 3, wp.int64(sa))
    wp.atomic_add(counters, 4, wp.int64(sf))
    wp.atomic_add(counters, 5, wp.int64(sd))

from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._analytic_predicates import strict
from warpower._prepared_seeds import point_value, prepare
from warpower._grouped_seeds import GroupedSeedQueries, group_bounds
import bounded_float32_filters as low
@wp.kernel
def counted_float32_groups(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    tree: Tree,
    live: wp.array[int],
    error: wp.array[int],
    shapes: wp.array[Sphere],
    boxes: wp.array[sb.Box],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    counters: wp.array[wp.int64],
):
    tid = wp.tid()
    i, lane = tid // 32, tid % 32
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    a, b, c, d = points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]]
    shape = shapes[i]
    box = boxes[i]
    attempted, fallback, exact = int(0), int(0), int(0)
    for group in range(lower.shape[0]):
        # All 32 lanes see the same group box. An invalid sphere enclosure
        # cannot skip any group; all surviving queries retain the exact policy.
        if sb.outside(box, lower[group], upper[group]):
            continue
        rank = 32 * group + lane
        if rank >= points.shape[0]:
            continue
        q = tree.ids[rank]
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        p = points[q]
        if sb.outside(box, p, p):
            continue
        attempted += 1
        fast = low.sphere_filter(a, b, c, d, p)
        if strict(fast):
            if fast.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        fallback += 1
        value = point_value(shape, p)
        if (
            wp.isfinite(value.value)
            and wp.isfinite(value.error)
            and wp.abs(value.value) > value.error
        ):
            if value.value > wp.float64(0.0):
                wp.atomic_and(live, i, 0)
            continue
        exact += 1
        result = symbolic_sphere(points, ids, q)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)


    wp.atomic_add(counters, 0, wp.int64(attempted))
    wp.atomic_add(counters, 1, wp.int64(fallback))
    wp.atomic_add(counters, 2, wp.int64(exact))

face_instances, seed_instances = [], []
class CountedQueries(GroupedSeedQueries):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counters = wp.zeros(3, dtype=wp.int64, device=self.device)
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)
        seed_instances.append(self)
    def run(self):
        wp.launch(group_bounds, len(self.lower), inputs=self._group_inputs, block_dim=self.block_dim, device=self.device)
        wp.launch(prepare, self.capacity, inputs=self._prepare_inputs, block_dim=self.block_dim, device=self.device)
        wp.launch(counted_float32_groups, 32*self.capacity, inputs=[*self._query_inputs,self.counters], block_dim=self.block_dim, device=self.device)

def install():
    from warpower import _grouped_seeds
    from warpower._proved_completion import SeededFacetCompletion
    _grouped_seeds.GroupedSeedQueries = CountedQueries
    original = SeededFacetCompletion.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self._float32_counters=wp.zeros(7,dtype=wp.int64,device=self.device)
        face_instances.append(self)
    def search(self,tree):
        c=self._certificate
        self._launch(counted_search_float32,4*self.capacity*self._workers,[self.points,tree.owners,self.tetrahedra,c._boundary,c._boundary_count,self._candidates,self._workers,self.error,c._face_supported,c.face_statistics,self._float32_counters],block_dim=self._query_block_dim)
    SeededFacetCompletion.__init__=init
    SeededFacetCompletion._search_faces=search
    for block in [32,256]:
        wp.load_module(module=__name__,device='cuda:0',block_dim=block)

if __name__=='__main__':
    import json,os
    from pathlib import Path
    install()
    from benchmarks.mesher import main
    main()
    face_names=['orientations','orientation_f32_uncertain','orientation_f64_uncertain','spheres','sphere_f32_uncertain','sphere_f64_uncertain','invalid_incomplete_threads']
    seed_names=['spheres','sphere_f32_uncertain','prepared_ball64_uncertain']
    rows=dict(scope='Instrumented complete graph executions; timings excluded.',face=[dict(site_capacity=len(x.points),counts=dict(zip(face_names,map(int,x._float32_counters.numpy())))) for x in face_instances],seed=[dict(site_capacity=len(x._check_inputs[0]),counts=dict(zip(seed_names,map(int,x.counters.numpy())))) for x in seed_instances])
    assert all(x['counts']['invalid_incomplete_threads']==0 for x in rows['face'])
    Path(os.environ['WARPOWER_FLOAT32_DIAGNOSTIC']).write_text(json.dumps(rows,indent=2)+'\n')
