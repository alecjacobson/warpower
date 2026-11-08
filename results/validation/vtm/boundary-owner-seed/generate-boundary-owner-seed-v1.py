"""Private candidate seeding: original arithmetic/traversal with one current-site upper bound."""
import ast
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'boundary_owner_seed_kernels_v1.py';assert not out.exists()
a=ast.parse((root/'warpower/_site_bvh.py').read_text());near=next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name=='nearest_site')
near.name='nearest_seeded';near.args.args.append(ast.arg(arg='seed',annotation=ast.Name(id='int',ctx=ast.Load())))
initial=ast.parse("""
if seed>=0 and seed<active[0] and seed<points.shape[0]:
    allowed=True
    if only_free and occupied[seed]>=0:allowed=False
    if allowed:
        distance=wp.length_sq(wp.vec3d(points[seed])-p)
        if wp.isfinite(distance) and distance<wp.float64(1.0e300):
            result.index=seed
            result.distance2=distance
""").body
assert isinstance(near.body[2],ast.Assign) and ast.unparse(near.body[2].targets[0])=='result.distance2'
near.body[3:3]=initial
b=ast.parse((root/'warpower/_boundary.py').read_text());kernel=next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name=='accumulate_samples');kernel.name='accumulate_seeded'
assert ast.unparse(kernel.body[1])=='owners[s] = -1'
kernel.body[1:1]=ast.parse('seed=owners[s]').body
class Replace(ast.NodeTransformer):
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id=='nearest_site':node.func.id='nearest_seeded';node.args.append(ast.Name(id='seed',ctx=ast.Load()))
  return node
kernel=Replace().visit(kernel)
header='import warp as wp\nfrom warpower import _bvh\nfrom warpower._site_bvh import SiteHit\nwp.set_module_options({"enable_backward":False,"fast_math":False})\n\n'
out.write_text(header+ast.unparse(ast.fix_missing_locations(near))+'\n\n'+ast.unparse(ast.fix_missing_locations(kernel))+'\n')
