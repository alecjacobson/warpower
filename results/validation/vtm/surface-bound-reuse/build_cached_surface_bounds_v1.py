"""Private per-query exact box-distance reuse; no geometry cache across calls."""
from pathlib import Path
s=Path('warpower/_surface_bvh.py').read_text()
a=s.index('@wp.func(inline=True)\ndef closest_point(');b=s.index('\n\n@wp.func(inline=True)\ndef ray_box(',a)
f=s[a:b].replace('def closest_point(', 'def cached_closest(')
f=f.replace('    stack[0] = 1','    bounds = wp.zeros(shape=128, dtype=wp.float64)\n    stack[0] = 1\n    bounds[0] = box_distance(tree, 1, p)')
f=f.replace('if box_distance(tree, node, p) >= result.distance2:', 'if bounds[size] >= result.distance2:')
f=f.replace('''            if box_distance(tree, left, p) < box_distance(tree, right, p):
                left, right = right, left
''','''            left_distance = box_distance(tree, left, p)
            right_distance = box_distance(tree, right, p)
            if left_distance < right_distance:
                left, right = right, left
                left_distance, right_distance = right_distance, left_distance
''')
f=f.replace('            stack[size + 1] = right','            stack[size + 1] = right\n            bounds[size] = left_distance\n            bounds[size + 1] = right_distance')
text='''"""Private per-query box-distance reuse, all original binary64 arithmetic."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, Closest, box_distance, closest_triangle, closest_point
wp.set_module_options({'enable_backward':False,'fast_math':False})

'''+f+'''

@wp.kernel
def reference(tree:SurfaceTree,points:wp.array[wp.vec3d],limits:wp.array[wp.float64],hits:wp.array[Closest]):
    i=wp.tid();hits[i]=closest_point(tree,points[i],limits[i])


@wp.kernel
def candidate(tree:SurfaceTree,points:wp.array[wp.vec3d],limits:wp.array[wp.float64],hits:wp.array[Closest]):
    i=wp.tid();hits[i]=cached_closest(tree,points[i],limits[i])
'''
dest=Path('.work/vtm/cached_surface_bounds_v1.py');assert not dest.exists();dest.write_text(text)
print(dest)
