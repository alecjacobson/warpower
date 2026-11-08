"""Private block-reduced global diagnostics; original per-sample nearest/scatter arithmetic."""
import ast,copy
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'boundary_global_reduce_kernels_v1.py';assert not out.exists()
old=next(n for n in ast.parse((root/'warpower/_boundary.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='accumulate_samples')
contribution=copy.deepcopy(old);contribution.name='contribution';contribution.decorator_list=ast.parse('@wp.func(inline=True)\ndef f():pass').body[0].decorator_list
assert ast.unparse(contribution.body[0])=='s = wp.tid()';contribution.body.pop(0);contribution.args.args.append(ast.arg(arg='s',annotation=ast.Name(id='int',ctx=ast.Load())))
class Returns(ast.NodeTransformer):
 def visit_Return(self,n):
  assert n.value is None;n.value=ast.parse('wp.vec3d(0.0)',mode='eval').body;return n
contribution=Returns().visit(contribution)
assert [ast.unparse(n.value.func) for n in contribution.body[-3:]]==['wp.atomic_add','wp.atomic_add','wp.atomic_max']
assert [ast.unparse(n.value.args[0]) for n in contribution.body[-3:]]==['energies','total','maximum']
contribution.body[-3:]=ast.parse('return wp.vec3d(weight * hit.distance2, weight, wp.sqrt(hit.distance2))').body
kernel=copy.deepcopy(old);kernel.name='accumulate_reduced';args=', '.join(a.arg for a in old.args.args)
body='s = wp.tid()\nvalues = contribution('+args+', s)\n'
body+='energy_tile=wp.tile_sum(wp.tile(values[0]))\nweight_tile=wp.tile_sum(wp.tile(values[1]))\nmaximum_tile=wp.tile_max(wp.tile(values[2]))\nmaximum_value=maximum_tile[0]\n'
body+='wp.tile_atomic_add(energies,energy_tile,offset=(component,))\nwp.tile_atomic_add(total,weight_tile,offset=(component,))\nif s % 32 == 0:\n    wp.atomic_max(maximum,component,maximum_value)\n'
kernel.body=ast.parse(body).body
header='import warp as wp\nfrom warpower import _bvh\nfrom warpower._site_bvh import nearest_site\nwp.set_module_options({"enable_backward":False,"fast_math":False})\n\n'
out.write_text(header+ast.unparse(ast.fix_missing_locations(contribution))+'\n\n'+ast.unparse(ast.fix_missing_locations(kernel))+'\n')
