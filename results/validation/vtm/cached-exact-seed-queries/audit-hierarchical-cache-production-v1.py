"""Verify private/production kernels and all other runtime/benchmark/example sources."""
import ast,hashlib,json,subprocess
from pathlib import Path
w=Path('.work/vtm');base='7c3dfc74acb4e947636e428bc0ea866f66dfdf77'
def nodes(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
p=nodes((w/'hierarchical_split_v2.py').read_text());q=nodes(Path('warpower/_stackless_seeds.py').read_text())
assert all(p[k]==q[k] for k in ['strict_check','exact_check'])
old=subprocess.check_output(['git','ls-tree','-r','--name-only',base,'warpower','benchmarks','examples'],text=True).splitlines()
changed=['warpower/_stackless_seeds.py']
for name in old:
 original=subprocess.check_output(['git','show',base+':'+name])
 current=Path(name).read_bytes()
 if name not in changed:assert current==original,name
 else:assert nodes(original.decode())['successor']==q['successor']
assert Path('README.md').read_bytes()==subprocess.check_output(['git','show',base+':README.md'])
paths=[*old,'tests/test_stackless_seeds.py','tests/test_stackless_dispatch.py','tests/test_stackless_traversal.py','tests/test_large_reuse.py','README.md']
r=dict(baseline=base,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),measured_kernel_AST_equal=True,unchanged_successor=True,unmodified_source_files=len(old)-1,changed_runtime_modules=changed,source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
p=w/'hierarchical-cache-production-source-audit-v1.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n')
print({k:v for k,v in r.items() if k!='source_sha256'})
