"""Current-input Morton groups for private VTM seed certification.

Every invocation refreshes exact float32 point-group boxes. A strict sphere-box
separation can skip a complete group; surviving points retain the original
strict Ball64 test and complete symbolic fallback. No temporal proof is stored.
"""

import warp as wp

from warpower import _sphere_bounds as sb
from warpower._bvh import Tree
from warpower._delaunay_certificate import Sphere, symbolic_sphere
from warpower._prepared_seeds import PreparedSeedQueries, point_value, prepare

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.kernel
def group_bounds(
    points: wp.array[wp.vec3], tree: Tree, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3]
):
    # The current hierarchy supplies a permutation of every input slot. Including
    # noncanonical duplicates in a group only makes its box more conservative.
    group = wp.tid()
    first = group * 32
    lo = wp.vec3(points[tree.ids[first]])
    hi = wp.vec3(lo)
    for j in range(first + 1, wp.min(first + 32, points.shape[0])):
        p = points[tree.ids[j]]
        lo = wp.min(lo, p)
        hi = wp.max(hi, p)
    lower[group] = lo
    upper[group] = hi


@wp.kernel
def counted_check_groups(
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
    if tid == 0:
        wp.atomic_add(counters, 0, wp.int64(1))
    if error[0] != 0 or i >= count[0]:
        return
    ids = wp.vec4i(tets[i, 0], tets[i, 1], tets[i, 2], tets[i, 3])
    shape = shapes[i]
    box = boxes[i]
    groups = int(0)
    skipped = int(0)
    queries = int(0)
    filtered = int(0)
    exact = int(0)
    for group in range(lower.shape[0]):
        groups += 1
        # All 32 lanes see the same group box. An invalid sphere enclosure
        # cannot skip any group; all surviving queries retain the exact policy.
        if sb.outside(box, lower[group], upper[group]):
            skipped += 1
            continue
        rank = 32 * group + lane
        if rank >= points.shape[0]:
            continue
        q = tree.ids[rank]
        if tree.owners[q] != q or q == ids[0] or q == ids[1] or q == ids[2] or q == ids[3]:
            continue
        queries += 1
        p = points[q]
        if sb.outside(box, p, p):
            continue
        filtered += 1
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

    if lane == 0:
        wp.atomic_add(counters, 1, wp.int64(1))
        wp.atomic_add(counters, 2, wp.int64(groups))
        wp.atomic_add(counters, 3, wp.int64(skipped))
    if queries > 0:
        wp.atomic_add(counters, 4, wp.int64(queries))
    if filtered > 0:
        wp.atomic_add(counters, 5, wp.int64(filtered))
    if exact > 0:
        wp.atomic_add(counters, 6, wp.int64(exact))

from warpower._grouped_seeds import GroupedSeedQueries
instances=[]
class CountedGroupedQueries(GroupedSeedQueries):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.counters=wp.zeros(7,dtype=wp.int64,device=self.device)
        wp.load_module(module=__name__,device=self.device,block_dim=self.block_dim)
        instances.append(self)
    def run(self):
        wp.launch(group_bounds,len(self.lower),inputs=self._group_inputs,block_dim=self.block_dim,device=self.device)
        wp.launch(prepare,self.capacity,inputs=self._prepare_inputs,block_dim=self.block_dim,device=self.device)
        wp.launch(counted_check_groups,32*self.capacity,inputs=[*self._query_inputs,self.counters],block_dim=self.block_dim,device=self.device)

if __name__=='__main__':
    import json,os
    from pathlib import Path
    import warpower._grouped_seeds as module
    module.GroupedSeedQueries=CountedGroupedQueries
    from benchmarks.mesher import main
    main()
    names=['calls','cells','group_decisions','groups_skipped','eligible_points_in_surviving_groups','strict_point_evaluations','exact_fallback_calls']
    rows=[dict(site_capacity=len(x._check_inputs[0]),cell_capacity=x.capacity,counts=dict(zip(names,map(int,x.counters.numpy())))) for x in instances]
    Path(os.environ['WARPOWER_GROUP_DIAGNOSTIC']).write_text(json.dumps(dict(scope='Counter-instrumented complete meshing; includes all graph executions. Counter overhead excludes these timings from performance evidence.',instances=rows),indent=2)+'\n')
