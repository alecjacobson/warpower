"""Generate an exact private child-push filter; production source stays unchanged."""
import ast
import hashlib
import json
from pathlib import Path

w = Path(__file__).parent
path = Path('warpower/_surface_bvh.py')
source = path.read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'closest_point')
original = '\n'.join(source.splitlines()[node.decorator_list[0].lineno-1:node.end_lineno])
old = '''            stack[size] = left
            stack[size + 1] = right
            bounds[size] = left_distance
            bounds[size + 1] = right_distance
            size += 2'''
new = '''            # Keep the original rejection comparison, including unordered NaNs.
            # The best distance only decreases; a rejected child cannot revive.
            if not (left_distance >= result.distance2):
                stack[size] = left
                bounds[size] = left_distance
                size += 1
            if not (right_distance >= result.distance2):
                stack[size] = right
                bounds[size] = right_distance
                size += 1'''
assert original.count(old) == 1
candidate = original.replace(old,new).replace('def closest_point(', 'def pruned_closest_point(',1)
restored = candidate.replace(new,old).replace('def pruned_closest_point(', 'def closest_point(',1)
assert ast.dump(ast.parse(restored)) == ast.dump(ast.parse(original))
header = '''"""Private exact nearest traversal: omit children already rejected by the pop test."""
import warp as wp
from warpower._surface_bvh import SurfaceTree, Closest, box_distance, closest_triangle, closest_point
wp.set_module_options({"enable_backward": False, "fast_math": False})

'''
kernels = '''

@wp.kernel
def reference(tree: SurfaceTree, points: wp.array[wp.vec3d], limits: wp.array[wp.float64], hits: wp.array[Closest]):
    i = wp.tid()
    hits[i] = closest_point(tree, points[i], limits[i])


@wp.kernel
def candidate(tree: SurfaceTree, points: wp.array[wp.vec3d], limits: wp.array[wp.float64], hits: wp.array[Closest]):
    i = wp.tid()
    hits[i] = pruned_closest_point(tree, points[i], limits[i])
'''
out = w/'surface_push_pruning_v1.py'
with out.open('x') as stream: stream.write(header+candidate+kernels)
tests = Path('tests/test_surface_bound_reuse.py').read_text()
old_import = 'from warpower._surface_bvh import Closest, SurfaceTree, closest_point'
assert tests.count(old_import) == 1
tests = tests.replace(old_import,'from warpower._surface_bvh import Closest, SurfaceTree\nfrom surface_push_pruning_v1 import pruned_closest_point as closest_point')
tests += '''

@pytest.fixture(autouse=True)
def restore_options():
    options = dict(wp.get_module_options(module=__name__))
    try:
        yield
    finally:
        wp.set_module_options(options, module=__name__)
'''
test_out = w/'test_surface_push_pruning_v1.py'
with test_out.open('x') as stream: stream.write(tests)
report = dict(scope=__doc__, original_arithmetic_and_pop_order_preserved=True, production_unchanged=True,
              sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),path,out,test_out,Path('tests/test_surface_bound_reuse.py')]})
with (w/'surface-push-pruning-source-v1.json').open('x') as stream: json.dump(report,stream,indent=2); stream.write('\n')
print(report)
