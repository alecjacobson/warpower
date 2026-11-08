"""Verify measured/private traversal and unchanged predicate/kernel source."""
import ast,hashlib,json,subprocess
from pathlib import Path
w=Path('.work/vtm');base='5998b1384e97c04532fd5f928feb0e09c2fa702a'
def nodes(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
p=nodes((w/'large_stackless_v1.py').read_text());q=nodes(Path('warpower/_stackless_seeds.py').read_text())
assert all(p[k]==q[k] for k in ['successor','check_stackless'])
old=subprocess.check_output(['git','ls-tree','-r','--name-only',base,'warpower','benchmarks','examples'],text=True).splitlines()
changed=['warpower/_facet_seeds.py','warpower/_reuse_delaunay.py']
for name in old:
 original=subprocess.check_output(['git','show',base+':'+name])
 current=Path(name).read_bytes()
 if name not in changed:assert current==original,name
 else:
  a,b=nodes(original.decode()),nodes(current.decode())
  for k in a:
   if not k.endswith('Workspace') and k!='CertifiedSeeds':assert a[k]==b[k],(name,k)
paths=[*old,'warpower/_stackless_seeds.py','tests/test_stackless_seeds.py','tests/test_stackless_traversal.py','tests/test_large_reuse.py']
r=dict(baseline=base,measured_kernel_AST_equal=True,unmodified_source_files=len(old)-2,setup_only_existing_module_changes=changed,source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
(w/'large-stackless-production-source-audit-v1.json').write_text(json.dumps(r,indent=2)+'\n')
print({k:v for k,v in r.items() if k!='source_sha256'})
