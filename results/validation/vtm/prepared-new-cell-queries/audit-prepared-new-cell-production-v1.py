"""Bind measured simple kernels to production and verify all independent runtime paths."""
import ast,hashlib,json,subprocess
from pathlib import Path
w=Path('.work/vtm');base='a4f7215fbf99249d11f765a4902c586daa606829';runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();assert runtime=='3b68a6043860209d3c507951b0e51309ed8bb8c5'
def nodes(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
p=nodes((w/'prepared_new_cell_v1.py').read_text());q=nodes(Path('warpower/_unproved_queries.py').read_text());names=['prepare_unproved','check_prepared_unproved'];assert all(p[k]==q[k] for k in names)
old=subprocess.check_output(['git','ls-tree','-r','--name-only',base,'warpower','benchmarks','examples'],text=True).splitlines();changed=['warpower/_proved_completion.py'];unchanged=[]
for name in old:
 original=subprocess.check_output(['git','show',base+':'+name]);current=Path(name).read_bytes()
 if name not in changed:assert current==original,name;unchanged.append(name)
 else:
  a,b=nodes(original.decode()),nodes(current.decode())
  for key in a:
   if key not in ['ProvedCertificate','SeededFacetCompletion']:assert a[key]==b[key],key
assert Path('README.md').read_bytes()==subprocess.check_output(['git','show',base+':README.md'])
paths=[*old,'warpower/_unproved_queries.py','tests/test_unproved_queries.py','tests/test_large_reuse.py','tests/test_predicates.py','tests/test_prepared_query_masks.py','tests/test_vtm_core.py','README.md']
r=dict(baseline=base,runtime=runtime,measured_kernel_AST_equal={k:True for k in names},original_completion_kernel_ASTs_unchanged=True,unmodified_source_files=len(unchanged),changed_runtime_modules=changed+['warpower/_unproved_queries.py'],source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths})
p=w/'prepared-new-cell-production-source-audit-v1.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k!='source_sha256'})
