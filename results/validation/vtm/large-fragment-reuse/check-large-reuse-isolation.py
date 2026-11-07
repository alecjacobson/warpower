"""Check exact source scope and preserve all existing Warp kernel/function ASTs."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();base='cca7a1de4f5ddf3c190f321cfda53c66f0525948';runtime='f8b5cfaf2625f264dd95448a272849b88c0c353b'
changed=subprocess.check_output(['git','diff','--name-only',base,'--','warpower'],text=True).splitlines()
assert changed==['warpower/_active_delaunay.py','warpower/_reuse_delaunay.py'],changed
assert not subprocess.check_output(['git','diff',runtime,'--','warpower'],text=True)
def gpu_nodes(data):
 t=ast.parse(data);r={}
 for n in ast.walk(t):
  if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and any(ast.unparse(d).startswith(('wp.kernel','wp.func')) for d in n.decorator_list):
   assert n.name not in r,n.name
   r[n.name]=ast.dump(n,include_attributes=False)
 return r
unchanged={};asts={}
for p in sorted((root/'warpower').glob('*.py')):
 name=str(p.relative_to(root));old=subprocess.check_output(['git','show',base+':'+name])
 new=p.read_bytes()
 if name not in changed:
  assert old==new,name
  unchanged[name]=hashlib.sha256(new).hexdigest()
 else:
  a,b=gpu_nodes(old),gpu_nodes(new);assert a==b,name
  asts[name]=dict(count=len(a),all_equal=True)
assert not subprocess.check_output(['git','diff',base,'--','benchmarks','examples','README.md'],text=True)
r=dict(baseline=base,runtime=runtime,production_changed=changed,unchanged_module_sha256=unchanged,changed_module_device_ast=asts,benchmarks_examples_readme_unchanged=True,scope='Only host construction/routing changes. Active full route uses Reuse for 2048<n<=4096 and clears its fragment at fresh solve. Reuse enables existing same-current-input proof consumption; every run recertifies retained cells against current original input. Ordinary standalone APIs and all Warp kernels/functions unchanged.')
p=root/'.work/vtm/large-reuse-isolation.json';p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='unchanged_module_sha256'},indent=2))
