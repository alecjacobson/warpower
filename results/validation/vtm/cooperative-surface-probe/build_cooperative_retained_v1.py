"""Build a private fused retained-classifier experiment; keep original source."""
from pathlib import Path

w = Path('.work/vtm')
source = Path('warpower/_selective.py').read_text()
start = source.index('    ids = wp.vec4i(')
stop = source.index('    side = material_side(tree, center)', start)
geometry = source[start:stop]
geometry = geometry.replace('wp.atomic_or(error, 0, 4)\n            return', 'result.error = 4\n            return result')
geometry = geometry.replace('wp.atomic_or(error, 0, 8)\n            return', 'result.error = 8\n            return result')
geometry = geometry.replace('wp.atomic_or(error, 0, 8)\n        return', 'result.error = 8\n        return result')
assert 'atomic' not in geometry
tail_start = stop + len('    side = material_side(tree, center)\n')
tail_end = source.index('\n\n\n@wp.kernel\ndef summarize', tail_start)
tail = source[tail_start:tail_end]
tail = tail.replace('quality[i] = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))', 'quality[i] = geometry.quality')
parts = ['''"""Private #197 fused retained queries, not a production implementation."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, closest_point
from warpower._wide_center import exact_geometry, uncertain
from cooperative_rays_v2 import partial_ray_count
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.struct
class Geometry:
    center: wp.vec3d
    radius: wp.float64
    quality: wp.float64
    error: int

@wp.func(inline=True)
def retained_geometry(points: wp.array[wp.vec3], tets: wp.array2d[int], i: int):
    result = Geometry()
''', geometry, '''    result.center = center
    result.radius = radius
    result.quality = wp.min(wp.float64(1.0), wp.float64(3.0) * determinant / (area2 * radius))
    return result

''']
for group in [1, 2, 4]:
    text = f'''@wp.kernel
def retained_{group}(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
):
    block, lane = wp.tid()
    row = lane // {group}
    part = lane % {group}
    i = block * {32 // group} + row
    active = i < keep.shape[0] and i < count[0] and error[0] == 0
    if part == 0 and i < keep.shape[0]:
        keep[i] = 0
    geometry = Geometry()
    if active:
        geometry = retained_geometry(points, tets, i)
        if geometry.error != 0:
            wp.atomic_or(error, 0, geometry.error)
    center = geometry.center
    radius = geometry.radius
'''
    for label, direction, condition in [
        ('first', 'wp.vec3d(0.8170638283521119, 0.33141623424177, 0.4713172342319)', 'active and geometry.error == 0'),
        ('second', 'wp.vec3d(-0.213173497731, 0.93127172131, 0.44721714319)', 'active and geometry.error == 0 and first[1] != 0'),
        ('third', 'wp.vec3d(0.63719234173, -0.72819341297, 0.27191345121)', 'active and geometry.error == 0 and first[1] != 0 and second[1] != 0'),
    ]:
        text += f'''    {label}_part = wp.vec2i(0, 0)
    if {condition}:
        {label}_part = partial_ray_count(tree, center, {direction}, -1, part, {group}, 0)
'''
        if group == 1:
            text += f'    {label} = {label}_part\n'
        else:
            text += f'''    {label}_tile = wp.tile_reshape(wp.tile({label}_part, preserve_type=True), shape=({32 // group}, {group}))
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
    if part != 0 or not active or geometry.error != 0:
        return
'''
    parts.append(text + tail + '\n\n')
dest = w/'cooperative_retained_v1.py'
assert not dest.exists()
dest.write_text(''.join(parts))
print(dest)
