"""Generate a private, reproducible #197 probe from current ray arithmetic."""
from pathlib import Path
import hashlib

root = Path.cwd()
source = (root / 'warpower/_surface_bvh.py').read_text()
start = source.index('@wp.func(inline=True)\ndef ray_count(')
end = source.index('\n\n@wp.func(inline=True)\ndef material_side', start)
original = source[start:end]
partial = original.replace('def ray_count(', 'def partial_ray_count(').replace(
    'excluded_face: int = -1):', 'excluded_face: int, lane: int, width: int, partition: int):'
).replace('if excluded_face >= 0:', 'if excluded_face >= 0 and lane == 0:')
partial = partial.replace('    node = int(1)\n    while node > 0:', '''    node = int(1)
    end_node = int(0)
    offset = lane
    stride = width
    if partition == 1:
        # Lanes own disjoint subtrees. Test every ancestor exactly as the scalar
        # traversal does. If a leaf is reached early, only its canonical lane
        # owns it. The excluded-face correction is still applied exactly once.
        bit = width // 2
        while bit > 0:
            if not ray_box(tree, node, origin, inverse):
                return wp.vec2i(crossings, ambiguous)
            if node >= tree.leaf_base:
                if (lane & (2 * bit - 1)) != 0:
                    return wp.vec2i(crossings, ambiguous)
                break
            if (lane & bit) == 0:
                node = tree.right[node]
            else:
                node = tree.left[node]
            bit //= 2
        end_node = tree.escape[node]
        offset = 0
        stride = 1
    while node != end_node:''')
partial = partial.replace('range(first, wp.min(first + 4, tree.n))', 'range(first + offset, wp.min(first + 4, tree.n), stride)')
header = '''"""Private #197 partial-ray experiment; no production replacement.

The leaf floating expressions are copied verbatim from the current scalar ray.
Integer sums implement both crossing reduction and ambiguity OR (>0). Every
thread reaches each tile collective, including inactive final-block lanes.
"""
import warp as wp
from warpower._surface_bvh import SurfaceTree, ray_box, ray_count, material_side, closest_point
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.kernel
def reference(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    i = wp.tid()
    r = ray_count(tree, points[i], wp.vec3d(1.0, 1.0, 1.0), excluded[i])
    out[i, 0] = r[0]
    out[i, 1] = r[1]
    out[i, 2] = material_side(tree, points[i], excluded[i])

@wp.kernel
def reference_side(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    i = wp.tid()
    out[i, 2] = material_side(tree, points[i], excluded[i])

@wp.kernel
def reference_nearest(tree: SurfaceTree, points: wp.array[wp.vec3d], distances: wp.array[wp.float64]):
    i = wp.tid()
    hit = closest_point(tree, points[i], wp.float64(1.0e150))
    distances[i] = hit.distance2

'''
pieces = [header, partial, '\n\n']
for mode, groups in [('leaf', [1, 2, 4]), ('subtree', [2, 4, 8, 16, 32])]:
    for group in groups:
        for diagnostic in [True, False]:
            name = f'{mode}_{group}' + ('' if diagnostic else '_side')
            text = f'''@wp.kernel
def {name}(tree: SurfaceTree, points: wp.array[wp.vec3d], excluded: wp.array[int], out: wp.array2d[int]):
    block, lane = wp.tid()
    row = lane // {group}
    part = lane % {group}
    i = block * {32 // group} + row
    p = wp.vec3d(0.0)
    ex = int(-1)
    if i < points.shape[0]:
        p = points[i]
        ex = excluded[i]
'''
            passes = []
            if diagnostic:
                passes.append(('raw', 'wp.vec3d(1.0, 1.0, 1.0)', 'i < points.shape[0]'))
            passes += [
                ('first', 'wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319)', 'i < points.shape[0]'),
                ('second', 'wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319)', 'i < points.shape[0] and first[1] != 0'),
                ('third', 'wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121)', 'i < points.shape[0] and first[1] != 0 and second[1] != 0'),
            ]
            for label, direction, condition in passes:
                text += f'''    {label}_part = wp.vec2i(0, 0)
    if {condition}:
        {label}_part = partial_ray_count(tree, p, {direction}, ex, part, {group}, {int(mode == 'subtree')})
'''
                if group == 1:
                    text += f'    {label} = {label}_part\n'
                else:
                    text += f'''    {label}_tile = wp.tile_reshape(wp.tile({label}_part), shape=({32 // group}, {group}))
    {label}_sums = wp.tile_sum({label}_tile, axis=1)
    {label} = {label}_sums[row]
'''
            text += '''    side = first[0]
    if first[1] != 0:
        side = second[0]
        if second[1] != 0:
            side = third[0]
            if third[1] != 0:
                side = 128
    if part == 0 and i < points.shape[0]:
        out[i, 2] = side
'''
            if diagnostic:
                text += '        out[i, 0] = raw[0]\n        out[i, 1] = int(raw[1] != 0)\n'
            pieces.append(text + '\n\n')
dest = root / '.work/vtm/cooperative_rays_v1.py'
assert not dest.exists()
dest.write_text(''.join(pieces))
print(dest, hashlib.sha256(dest.read_bytes()).hexdigest())
