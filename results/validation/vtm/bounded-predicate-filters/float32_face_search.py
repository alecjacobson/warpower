import warp as wp
import bounded_float32_filters as af
from warpower._mesh_topology import face_vertices

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})

@wp.kernel
def search_float32(
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
    best = int(-1)
    for q in range(lane, points.shape[0], workers):
        if owners[q] != q or q == face[0] or q == face[1] or q == face[2]:
            continue
        orientation = af.orient3d(a, b, c, points[q])
        if orientation.valid == 0:
            wp.atomic_or(error, 0, 2)
            return
        if orientation.sign <= 0:
            continue
        if best < 0:
            best = q
        else:
            result = af.symbolic_sphere(points, wp.vec4i(face[0], face[1], face[2], best), q)
            if result.valid == 0 or result.sign == 0:
                wp.atomic_or(error, 0, 2)
                return
            if result.sign > 0:
                best = q
    candidates[tid] = best


