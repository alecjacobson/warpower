"""Test-only copy that explicitly flushes both operands/results at every op."""
import ast
from pathlib import Path

w = Path('.work/vtm')
module = ast.parse((w / 'bounded_float32_filters.py').read_text())
functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in {'orientation_filter', 'sphere_filter'}]

class Flush(ast.NodeTransformer):
    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        name = {ast.Add: 'flushed_add', ast.Sub: 'flushed_sub', ast.Mult: 'flushed_mul'}[type(node.op)]
        return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[node.left, node.right], keywords=[])

    def visit_UnaryOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.USub):
            return ast.Call(func=ast.Name(id='flushed_neg', ctx=ast.Load()), args=[node.operand], keywords=[])
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'wp' and node.func.attr == 'abs':
            node.func = ast.Name(id='flushed_abs', ctx=ast.Load())
        return node

tree = Flush().visit(ast.Module(body=functions, type_ignores=[]))
header = '''"""Test-only explicit flush variant of the bounded binary32 arithmetic."""
import warp as wp
from warpower import _filtered_predicates as fp
from bounded_float32_filters import bounded, unavailable, ORIENT_BOUND, SPHERE_BOUND, ABS_FLOOR

wp.set_module_options({"enable_backward": False, "fast_math": False, "fuse_fp": False})
MIN_NORMAL = wp.constant(wp.float32(2.0**-126))

@wp.func(inline=True)
def flush(value: float):
    if wp.abs(value) < MIN_NORMAL:
        return 0.0
    return value

@wp.func(inline=True)
def flushed_add(a: float, b: float):
    return flush(flush(a) + flush(b))

@wp.func(inline=True)
def flushed_sub(a: float, b: float):
    return flush(flush(a) - flush(b))

@wp.func(inline=True)
def flushed_mul(a: float, b: float):
    return flush(flush(a) * flush(b))

@wp.func(inline=True)
def flushed_neg(a: float):
    return flush(-flush(a))

@wp.func(inline=True)
def flushed_abs(a: float):
    return flush(wp.abs(flush(a)))

'''
out = w / 'flushed_float32_filters.py'
assert not out.exists()
out.write_text(header + ast.unparse(ast.fix_missing_locations(tree)) + '\n')
print(out)
