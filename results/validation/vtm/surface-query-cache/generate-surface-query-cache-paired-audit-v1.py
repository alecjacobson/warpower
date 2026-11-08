"""Generate private paired original/cached queries in complete classifier trajectories."""
import ast
from pathlib import Path
root=Path.cwd();src=root/'warpower/_surface_query_cache.py';out=root/'.work/vtm/surface_query_cache_paired_audit_kernels_v1.py';assert not out.exists()
header='import warp as wp\nfrom warpower._extraction import SelectionCenter\nfrom warpower._surface_bvh import Closest, SurfaceTree, closest_point, closest_triangle, material_side\nfrom warpower._wide_center import exact_geometry, uncertain\nfrom warpower._surface_query_cache import QueryState, lookup, same_center, cached_side, cached_nearest\nwp.set_module_options({"enable_backward": False, "fast_math": False})\n\n@wp.func(inline=True)\ndef audit_side(tree: SurfaceTree, center: wp.vec3d, previous: QueryState, table: wp.array[int], stats: wp.array[int]):\n    j=lookup(center, previous, table)\n    wp.atomic_add(stats,0,1)\n    if j>=0 and previous.rays[j]!=0:wp.atomic_add(stats,1,1)\n    a=cached_side(tree,center,previous,table)\n    b=material_side(tree,center)\n    if a!=b:wp.atomic_add(stats,2,1)\n    return a\n\n@wp.func(inline=True)\ndef audit_nearest(tree: SurfaceTree, center: wp.vec3d, limit: wp.float64, previous: QueryState, table: wp.array[int], stats: wp.array[int]):\n    j=lookup(center,previous,table)\n    wp.atomic_add(stats,3,1)\n    if j>=0 and previous.nearest[j]!=0 and wp.cast(limit,wp.uint64)==wp.cast(previous.limits[j],wp.uint64):wp.atomic_add(stats,4,1)\n    a=cached_nearest(tree,center,limit,previous,table)\n    b=closest_point(tree,center,limit)\n    if a.face!=b.face:wp.atomic_add(stats,5,1)\n    if wp.cast(a.distance2,wp.uint64)!=wp.cast(b.distance2,wp.uint64):wp.atomic_add(stats,6,1)\n    if not same_center(a.point,b.point):wp.atomic_add(stats,7,1)\n    return a\n\n'
class Calls(ast.NodeTransformer):
 def visit_Call(self,node):
  node=self.generic_visit(node)
  if isinstance(node.func,ast.Name) and node.func.id in ['cached_side','cached_nearest','selection_center_cached']:
   if node.func.id=='cached_side':node.func.id='audit_side'
   if node.func.id=='cached_nearest':node.func.id='audit_nearest'
   node.args.append(ast.Name(id='stats',ctx=ast.Load()))
  return node
nodes=[]
for node in ast.parse(src.read_text()).body:
 if isinstance(node,ast.FunctionDef) and node.name in ['classify_cached','retained_cached','selection_center_cached','selection_cached']:
  node.args.args.append(ast.arg(arg='stats',annotation=ast.parse('wp.array[int]',mode='eval').body));nodes.append(Calls().visit(node))
assert len(nodes)==4
out.write_text(header+'\n\n'.join(ast.unparse(ast.fix_missing_locations(n)) for n in nodes)+'\n')
