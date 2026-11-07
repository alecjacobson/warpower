"""Private #159/#197 staged retained queries with explicit geometry scratch."""
from pathlib import Path

w=Path('.work/vtm')
source=(w/'cooperative_retained_v1.py').read_text()
text='''"""Private staged retained classifier. Extra scratch: 32 bytes/cell capacity."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, closest_point
from cooperative_rays_v2 import partial_ray_count
from cooperative_retained_v1 import Geometry, retained_geometry
wp.set_module_options({'enable_backward': False, 'fast_math': False})

@wp.kernel
def prepare_retained(
    tree: SurfaceTree, points: wp.array[wp.vec3], tets: wp.array2d[int],
    count: wp.array[int], tolerance: wp.float64, error: wp.array[int],
    keep: wp.array[int], quality: wp.array[wp.float64],
    centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],
):
    i = wp.tid()
    keep[i] = 0
    if i >= count[0] or error[0] != 0:
        return
    geometry = retained_geometry(points, tets, i)
    if geometry.error != 0:
        wp.atomic_or(error, 0, geometry.error)
        return
    centers[i] = geometry.center
    radii[i] = geometry.radius
    quality[i] = geometry.quality

'''
start=source.index('@wp.kernel\ndef retained_1(')
queries=source[start:]
for group in [1,2,4]:queries=queries.replace(f'def retained_{group}(',f'def query_{group}(')
queries=queries.replace('    keep: wp.array[int], quality: wp.array[wp.float64],\n):','    keep: wp.array[int], quality: wp.array[wp.float64],\n    centers: wp.array[wp.vec3d], radii: wp.array[wp.float64],\n):')
queries=queries.replace('''    if active:
        geometry = retained_geometry(points, tets, i)
        if geometry.error != 0:
            wp.atomic_or(error, 0, geometry.error)
''','''    if active:
        geometry.center = centers[i]
        geometry.radius = radii[i]
        geometry.quality = quality[i]
''')
dest=w/'cooperative_retained_split_v1.py'
assert not dest.exists();dest.write_text(text+queries)
print(dest)
