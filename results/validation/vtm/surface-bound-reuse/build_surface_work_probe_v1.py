"""Generate observers from the current scalar surface-query arithmetic."""
from pathlib import Path

source=Path('warpower/_surface_bvh.py').read_text()
def body(name,next_name):
    a=source.index('@wp.func(inline=True)\ndef '+name+'(')
    b=source.index('\n\n@wp.func(inline=True)\ndef '+next_name+'(',a)
    return source[a:b]

nearest=body('closest_point','ray_box').replace('def closest_point(', 'def counted_nearest(')
nearest=nearest.replace('    result = Closest()','    work = Counts()\n    work[0] = 1\n    work[7] = 1\n    result = Closest()')
nearest=nearest.replace('        node = stack[size]','        node = stack[size]\n        work[1] += 1\n        work[8] += 1')
nearest=nearest.replace('if box_distance(tree, node, p) >= result.distance2:\n            continue','if box_distance(tree, node, p) >= result.distance2:\n            work[2] += 1\n            continue')
nearest=nearest.replace('        if node >= tree.leaf_base:\n','        if node >= tree.leaf_base:\n            work[4] += 1\n')
nearest=nearest.replace('                f = tree.ids[j]','                work[5] += 1\n                f = tree.ids[j]')
nearest=nearest.replace('                if distance < result.distance2:\n','                if distance < result.distance2:\n                    work[6] += 1\n')
nearest=nearest.replace('            left, right = int(tree.left[node]), int(tree.right[node])','            work[3] += 1\n            work[8] += 2\n            left, right = int(tree.left[node]), int(tree.right[node])')
nearest=nearest.replace('            size += 2','            size += 2\n            work[7] = wp.max(work[7], size)')
nearest=nearest.replace('    return result','    observed = NearestWork()\n    observed.hit = result\n    observed.work = work\n    return observed')
ray=body('ray_count','material_side').replace('def ray_count(', 'def counted_ray(')
ray=ray.replace('    crossings, ambiguous = int(0), int(0)','    work = Counts()\n    work[0] = 1\n    crossings, ambiguous = int(0), int(0)')
ray=ray.replace('    while node > 0:\n','    while node > 0:\n        work[1] += 1\n        work[8] += 1\n')
ray=ray.replace('        if not ray_box(tree, node, origin, inverse):\n','        if not ray_box(tree, node, origin, inverse):\n            work[2] += 1\n')
ray=ray.replace('        if node >= tree.leaf_base:\n','        if node >= tree.leaf_base:\n            work[4] += 1\n')
ray=ray.replace('                a = wp.vec3d(tree.points[tree.indices[3 * f]])','                work[5] += 1\n                a = wp.vec3d(tree.points[tree.indices[3 * f]])')
ray=ray.replace('                        ambiguous = 1','                        ambiguous = 1\n                        work[7] += 1')
ray=ray.replace('                    crossings += wp.where','                    work[6] += 1\n                    crossings += wp.where')
ray=ray.replace('        else:\n            node = tree.right[node]','        else:\n            work[3] += 1\n            node = tree.right[node]')
ray=ray.replace('    return wp.vec2i(crossings, ambiguous)','    observed = RayWork()\n    observed.result = wp.vec2i(crossings, ambiguous)\n    observed.work = work\n    return observed')
text='''"""Private query-work counters, not performance timing or production code."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, Closest, box_distance, closest_triangle, ray_box, closest_point, material_side
wp.set_module_options({'enable_backward': False, 'fast_math': False})
Counts = wp.types.vector(length=9, dtype=wp.int32)

@wp.struct
class NearestWork:
    hit: Closest
    work: Counts

@wp.struct
class RayWork:
    result: wp.vec2i
    work: Counts

'''+nearest+'\n\n'+ray+'''

@wp.func(inline=True)
def counted_side(tree: SurfaceTree, p: wp.vec3d):
    first = counted_ray(tree, p, wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319))
    work = first.work
    result = first.result
    if result[1] != 0:
        second = counted_ray(tree, p, wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319))
        work += second.work
        result = second.result
    if result[1] != 0:
        third = counted_ray(tree, p, wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121))
        work += third.work
        result = third.result
    output = RayWork()
    output.result = result
    output.work = work
    return output


@wp.func(inline=True)
def same(a: Closest, b: Closest):
    return int(a.face == b.face and a.distance2 == b.distance2
               and a.point[0] == b.point[0] and a.point[1] == b.point[1] and a.point[2] == b.point[2])


@wp.kernel
def observe(
    tree: SurfaceTree, centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
    count: wp.array[int], error: wp.array[int], tolerance: wp.float64,
    full_work: wp.array[Counts], retained_work: wp.array[Counts], ray_work: wp.array[Counts],
    status: wp.array2d[int],
):
    i = wp.tid()
    if error[0] != 0 or i >= count[0]:
        return
    p = centers[i]
    r = radii[i]
    full = counted_nearest(tree, p, wp.float64(1.0e150))
    reference = closest_point(tree, p, wp.float64(1.0e150))
    full_work[i] = full.work
    status[i, 0] = 1 - same(reference, full.hit)
    status[i, 1] = int(reference.face >= 0 and wp.isfinite(reference.distance2))
    rays = counted_side(tree, p)
    side = rays.result[0]
    if rays.result[1] != 0:
        side = 128
    ray_work[i] = rays.work
    status[i, 2] = int(side != material_side(tree, p))
    status[i, 3] = side
    nearest_work = Counts()
    if side != 1 or r >= wp.float64(1.0e140):
        limit = wp.float64(1.0e150)
        bounded = side == 0 and tolerance > wp.float64(0.0) and r < wp.float64(1.0e140)
        if bounded:
            limit = wp.min(limit, wp.max(wp.float64(1.0e-150), tolerance*r*(wp.float64(1.0)+wp.float64(1.0e-12))))
        status[i, 4] = 1 + int(bounded)
        limited = counted_nearest(tree, p, limit)
        expected = closest_point(tree, p, limit)
        nearest_work = limited.work
        status[i, 5] = 1 - same(expected, limited.hit)
        status[i, 6] = int(expected.face >= 0)
    retained_work[i] = nearest_work
'''
dest=Path('.work/vtm/surface_work_probe_v1.py');assert not dest.exists();dest.write_text(text)
print(dest)
