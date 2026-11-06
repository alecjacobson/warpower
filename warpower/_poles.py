"""Finite Voronoi poles and a double-precision pole-distance hierarchy."""

import warp as wp

from ._mesh_topology import face_vertices, sorted_face
from ._tetrahedra import sorted4

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})


@wp.func
def supporting_face(tets: wp.array2d[int], face: int):
    # Canonical cyclic order preserves the outward orientation while avoiding
    # differences in arithmetic caused by the incident tetrahedron's ordering.
    ids = face_vertices(tets, face)
    if ids[1] < ids[0] and ids[1] < ids[2]:
        return wp.vec3i(ids[1], ids[2], ids[0])
    if ids[2] < ids[0] and ids[2] < ids[1]:
        return wp.vec3i(ids[2], ids[0], ids[1])
    return ids


@wp.kernel
def centers(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    output: wp.array[wp.vec3d],
    error: wp.array[int],
):
    t = wp.tid()
    if t >= count[0] or error[0] != 0:
        return
    p = wp.vec3d(points[tets[t, 0]])
    a = wp.vec3d(points[tets[t, 1]]) - p
    b = wp.vec3d(points[tets[t, 2]]) - p
    c = wp.vec3d(points[tets[t, 3]]) - p
    det = wp.dot(a, wp.cross(b, c))
    if det <= wp.float64(0.0) or not wp.isfinite(det):
        wp.atomic_or(error, 0, 4)
        return
    center = p + (
        wp.dot(a, a) * wp.cross(b, c)
        + wp.dot(b, b) * wp.cross(c, a)
        + wp.dot(c, c) * wp.cross(a, b)
    ) / (wp.float64(2.0) * det)
    if (
        not wp.isfinite(center[0])
        or not wp.isfinite(center[1])
        or not wp.isfinite(center[2])
        or wp.length_sq(center) >= wp.float64(1e298)
    ):
        wp.atomic_or(error, 0, 4)
    else:
        output[t] = center


@wp.kernel
def hull_counts(
    tets: wp.array2d[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    counts: wp.array[int],
    error: wp.array[int],
):
    slot = wp.tid()
    if error[0] != 0 or occurrences[slot] != 1:
        return
    ids = face_vertices(tets, table[slot])
    for k in range(3):
        wp.atomic_add(counts, ids[k] + 1, 1)


@wp.kernel
def hull_scatter(
    tets: wp.array2d[int],
    table: wp.array[int],
    occurrences: wp.array[int],
    cursor: wp.array[int],
    faces: wp.array[int],
    error: wp.array[int],
):
    slot = wp.tid()
    if error[0] != 0 or occurrences[slot] != 1:
        return
    face = table[slot]
    ids = face_vertices(tets, face)
    for k in range(3):
        at = wp.atomic_add(cursor, ids[k], 1)
        if at < faces.shape[0]:
            faces[at] = face
        else:
            wp.atomic_or(error, 0, 4)


@wp.func
def face_less(tets: wp.array2d[int], a: int, b: int):
    x, y = sorted_face(face_vertices(tets, a)), sorted_face(face_vertices(tets, b))
    for k in range(3):
        if x[k] != y[k]:
            return x[k] < y[k]
    return False


@wp.kernel
def hull_directions(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    offsets: wp.array[int],
    faces: wp.array[int],
    hull_face: wp.array[int],
    normals: wp.array[wp.vec3d],
    error: wp.array[int],
):
    i = wp.tid()
    if error[0] != 0:
        return
    first, end = offsets[i], offsets[i + 1]
    if first < 0 or end > faces.shape[0] or end < first:
        wp.atomic_or(error, 0, 4)
        return
    # Each vertex owns a disjoint star. Sorting this short list by original
    # vertex IDs makes floating accumulation independent of atomic scatter and
    # tetrahedron row order. No global floating atomic reduction is involved.
    for k in range(first + 1, end):
        face = int(faces[k])
        at = int(k)
        while at > first:
            previous = int(faces[at - 1])
            if not face_less(tets, face, previous):
                break
            faces[at] = previous
            at -= 1
        faces[at] = face
    total = wp.vec3d(0.0)
    for k in range(first, end):
        ids = supporting_face(tets, faces[k])
        a = wp.vec3d(points[ids[1]]) - wp.vec3d(points[ids[0]])
        b = wp.vec3d(points[ids[2]]) - wp.vec3d(points[ids[0]])
        normal = wp.cross(a, b)
        length = wp.length(normal)
        if length <= wp.float64(0.0) or not wp.isfinite(length):
            wp.atomic_or(error, 0, 4)
            return
        total += normal / length
    if end > first:
        if wp.length_sq(total) <= wp.float64(0.0):
            wp.atomic_or(error, 0, 4)
            return
        hull_face[i] = faces[first]
        normals[i] = total


@wp.kernel
def farthest_distance(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    centers: wp.array[wp.vec3d],
    hull_face: wp.array[int],
    normals: wp.array[wp.vec3d],
    maximum: wp.array[wp.float64],
    opposite: bool,
    error: wp.array[int],
):
    t = wp.tid()
    if error[0] != 0 or t >= count[0]:
        return
    for k in range(4):
        i = tets[t, k]
        delta = centers[t] - wp.vec3d(points[i])
        eligible = hull_face[i] == 2147483647
        if opposite:
            eligible = wp.dot(delta, normals[i]) < wp.float64(0.0)
        if eligible:
            wp.atomic_max(maximum, i, wp.length_sq(delta))


@wp.kernel
def farthest_select(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    count: wp.array[int],
    centers: wp.array[wp.vec3d],
    hull_face: wp.array[int],
    normals: wp.array[wp.vec3d],
    maximum: wp.array[wp.float64],
    keys: wp.array2d[int],
    selected: wp.array[int],
    prefix: int,
    opposite: bool,
    error: wp.array[int],
):
    t = wp.tid()
    if error[0] != 0 or t >= count[0]:
        return
    key = sorted4(wp.vec4i(tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]))
    for k in range(4):
        i = tets[t, k]
        delta = centers[t] - wp.vec3d(points[i])
        eligible = hull_face[i] == 2147483647
        if opposite:
            eligible = wp.dot(delta, normals[i]) < wp.float64(0.0)
        eligible = eligible and wp.length_sq(delta) == maximum[i]
        for j in range(prefix):
            eligible = eligible and key[j] == keys[i, j]
        if eligible:
            if prefix < 4:
                wp.atomic_min(keys, i, prefix, key[prefix])
            else:
                wp.atomic_min(selected, i, t)


@wp.kernel
def pole_directions(
    points: wp.array[wp.vec3],
    tets: wp.array2d[int],
    centers: wp.array[wp.vec3d],
    hull_face: wp.array[int],
    first: wp.array[int],
    normals: wp.array[wp.vec3d],
    error: wp.array[int],
):
    i = wp.tid()
    normal = wp.vec3d(0.0)
    if error[0] == 0:
        if hull_face[i] != 2147483647:
            # The positive sum of incident outward unit normals supplies the
            # first pole's direction at infinity. Its length is irrelevant.
            normal = normals[i]
        elif first[i] != 2147483647:
            normal = centers[first[i]] - wp.vec3d(points[i])
    normals[i] = normal


@wp.kernel
def publish_poles(
    centers: wp.array[wp.vec3d],
    first: wp.array[int],
    second: wp.array[int],
    poles: wp.array[wp.vec3d],
    valid: wp.array[int],
    source: wp.array[int],
    sortable: wp.array[wp.vec3],
    count: wp.array[int],
    error: wp.array[int],
):
    j = wp.tid()
    i, side = j // 2, j % 2
    t = int(first[i])
    if side == 1:
        t = second[i]
    p = wp.vec3d(0.0)
    valid[j] = 0
    source[j] = -1
    if error[0] == 0 and t != 2147483647:
        p = centers[t]
        valid[j] = 1
        source[j] = t
        wp.atomic_add(count, 0, 1)
    poles[j] = p
    # Only sort order is rounded. Exact distance queries use the double arrays.
    sortable[j] = wp.vec3(wp.min(wp.max(p, wp.vec3d(-1e29)), wp.vec3d(1e29)))


@wp.struct
class PoleTree:
    poles: wp.array[wp.vec3d]
    valid: wp.array[int]
    ids: wp.array[int]
    lower: wp.array[wp.vec3d]
    upper: wp.array[wp.vec3d]
    base: int
    n: int


@wp.kernel
def pole_leaves(tree: PoleTree):
    i = wp.tid()
    lo, hi = wp.vec3d(1e300), wp.vec3d(-1e300)
    for k in range(4 * i, wp.min(4 * i + 4, tree.n)):
        index = tree.ids[k]
        if tree.valid[index] != 0:
            p = tree.poles[index]
            lo, hi = wp.min(lo, p), wp.max(hi, p)
    tree.lower[tree.base + i] = lo
    tree.upper[tree.base + i] = hi


@wp.kernel
def pole_merge(tree: PoleTree, first: int):
    i = first + wp.tid()
    tree.lower[i] = wp.min(tree.lower[2 * i], tree.lower[2 * i + 1])
    tree.upper[i] = wp.max(tree.upper[2 * i], tree.upper[2 * i + 1])


@wp.func(inline=True)
def pole_box_distance(tree: PoleTree, node: int, p: wp.vec3d):
    delta = wp.max(wp.max(tree.lower[node] - p, p - tree.upper[node]), wp.vec3d(0.0))
    return wp.dot(delta, delta)


@wp.func(inline=True)
def nearest_pole(tree: PoleTree, p: wp.vec3d):
    best = wp.float64(1e300)
    stack = wp.zeros(shape=64, dtype=int)
    stack[0] = 1
    size = int(1)
    while size > 0:
        size -= 1
        node = stack[size]
        # No pole ID is requested; pruning equal distances is important for
        # duplicate circumcenters on cospherical surfaces.
        if pole_box_distance(tree, node, p) >= best:
            continue
        if node >= tree.base:
            begin = 4 * (node - tree.base)
            for k in range(begin, wp.min(begin + 4, tree.n)):
                i = tree.ids[k]
                if tree.valid[i] != 0:
                    delta = tree.poles[i] - p
                    best = wp.min(best, wp.dot(delta, delta))
        else:
            left, right = 2 * node, 2 * node + 1
            if pole_box_distance(tree, left, p) < pole_box_distance(tree, right, p):
                left, right = right, left
            stack[size] = left
            stack[size + 1] = right
            size += 2
    return wp.sqrt(best)
