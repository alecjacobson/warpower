"""Generate isolated binary32 first stages from the existing grouped formulas."""
import ast
from pathlib import Path

root = Path.cwd()
source = (root / 'warpower/_analytic_predicates.py').read_text()
start = source.index('@wp.func\ndef orientation_filter(')
end = source.index('@wp.func\ndef strict(')
filters = source[start:end]

# Preserve every determinant/permanent operation and its grouping. Inputs are
# already binary32. Promote only the finished value and its error enclosure.
tree = ast.parse(filters)
class LowerInputs(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'wp' and node.func.attr == 'float64':
            assert len(node.args) == 1 and isinstance(node.args[0], ast.Subscript)
            return node.args[0]
        return node

tree = LowerInputs().visit(tree)
for function in tree.body:
    assert isinstance(function, ast.FunctionDef)
    names = [a.arg for a in function.args.args]
    gate = ast.parse('if not (' + ' and '.join(f'bounded({n})' for n in names) + '):\n    return unavailable()').body[0]
    function.body.insert(0, gate)
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Attribute) and node.targets[0].attr in {'value', 'error'}:
            value = node.value
            if node.targets[0].attr == 'error':
                value = ast.BinOp(left=value, op=ast.Add(), right=ast.Name(id='ABS_FLOOR', ctx=ast.Load()))
            node.value = ast.Call(func=ast.Attribute(value=ast.Name(id='wp', ctx=ast.Load()), attr='float64', ctx=ast.Load()), args=[value], keywords=[])
filters = ast.unparse(ast.fix_missing_locations(tree))

header = '''"""Private bounded-range binary32 strict filters, #180.

Formula grouping follows Shewchuk's public-domain May 18, 1996 predicates.c,
https://www.cs.cmu.edu/~quake/robust.html, via Warpower's existing first stages.
Only finite original float32 coordinates with magnitude <=16 enter the new
arithmetic. Generic rounding-factor and absolute-underflow bounds are checked
by prove_float32_filter_bounds.py. Every uncertain/unsupported result retains
the old float64/Ball64/integer/symbolic path. No tolerance defines a tie.
"""
import warp as wp
from warpower import _analytic_predicates as old
from warpower import _filtered_predicates as fp

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
ORIENT_BOUND = wp.constant(wp.float32(2.0**-19))
SPHERE_BOUND = wp.constant(wp.float32(2.0**-17))
ABS_FLOOR = wp.constant(wp.float32(2.0**-64))


@wp.func(inline=True)
def bounded(p: wp.vec3):
    return (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])
            and wp.abs(p[0]) <= 16.0 and wp.abs(p[1]) <= 16.0 and wp.abs(p[2]) <= 16.0)


@wp.func(inline=True)
def unavailable():
    result = fp.Ball()
    result.value = wp.float64(0.0)
    result.error = wp.float64(wp.inf)
    return result


'''
footer = '''

@wp.func(inline=True)
def orient3d(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3):
    value = orientation_filter(a, b, c, d)
    if old.strict(value):
        return fp.predicate(value)
    return old.orient3d(a, b, c, d)


@wp.func(inline=True)
def symbolic_sphere(points: wp.array[wp.vec3], ids: wp.vec4i, query: int):
    value = sphere_filter(points[ids[0]], points[ids[1]], points[ids[2]], points[ids[3]], points[query])
    if old.strict(value):
        return fp.predicate(value)
    return old.symbolic_sphere(points, ids, query)
'''
out = root / '.work/vtm/bounded_float32_filters.py'
assert not out.exists()
out.write_text(header + filters + '\n' + footer)
print(out)
