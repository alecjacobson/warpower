"""Bind the production setup policy to the measured prototype and unchanged kernels."""
import ast,copy,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';baseline='df74493c97633324147cb39ef4c41a7d34c7b90a';dest=w/'larger-prefix-production-source-audit-v2.json';assert not dest.exists()
def old(name):return subprocess.check_output(['git','show',baseline+':'+name],text=True)
def dump(node):return ast.dump(node)
name='warpower/_active_delaunay.py';current=(root/name).read_text()
normalized=current.replace('n > 8192','n > 2048').replace('in [4, 8192]','in [4, 2048]')
assert dump(ast.parse(normalized))==dump(ast.parse(old(name)))
prod=ast.parse(current);prototype=ast.parse((w/'larger_prefix_v1.py').read_text())
def constructor(tree):
 c=next(n for n in tree.body if isinstance(n,ast.ClassDef));return next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
a=copy.deepcopy(constructor(prototype));b=constructor(prod)
assert isinstance(a.body[0],ast.If) and 'prefix_capacities' in ast.unparse(a.body[0].test)
a.body.pop(0);a.args=copy.deepcopy(b.args)
for node in ast.walk(a):
 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='load_module':
  k=next(k for k in node.keywords if k.arg=='module');assert ast.literal_eval(k.value)=='warpower._active_delaunay';k.value=ast.Name(id='__name__',ctx=ast.Load())
assert dump(a)==dump(b)
name='warpower/_vtm.py';changed=ast.parse((root/name).read_text());count=0
for node in ast.walk(changed):
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mesh_options' for t in node.targets) and isinstance(node.value,ast.Call) and any(k.arg=='prefix_capacities' for k in node.value.keywords):
  count+=1;node.value=ast.parse('dict(active_count=active_count)',mode='eval').body
assert count==1 and dump(changed)==dump(ast.parse(old(name)))
files=subprocess.check_output(['git','ls-files','warpower','benchmarks','examples'],text=True).splitlines()
unchanged=[]
for name in files:
 if name in ['warpower/_active_delaunay.py','warpower/_vtm.py']:continue
 assert (root/name).read_bytes()==subprocess.check_output(['git','show',baseline+':'+name]);unchanged.append(name)
assert (root/'README.md').read_text()==old('README.md')
files+=['README.md','tests/test_larger_prefixes.py','tests/test_active_delaunay.py','tests/test_active_sites.py','tests/test_large_reuse.py']
r=dict(baseline=baseline,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sorted(files)},unchanged_files=sorted(unchanged),all_device_kernel_ASTs_unchanged=True,all_other_methods_unchanged=True,production_constructor_equals_measured_prototype_after_explicit_policy=True,vtm_change_is_only_setup_level_selection=True,readme_unchanged=True,prototype_sha256=hashlib.sha256((w/'larger_prefix_v1.py').read_bytes()).hexdigest())
dest.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['source_sha256','unchanged_files']});print('source files',len(files),'unchanged runtime/benchmark/example files',len(unchanged))
