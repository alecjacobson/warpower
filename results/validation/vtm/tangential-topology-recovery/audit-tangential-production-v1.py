"""Pin the production recovery sources and independent-API nonchange boundary."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=w/'tangential-production-source-audit-v1.json';assert not out.exists()
base='ceb5ea1420b4c9e9a7b16c89a0432e567962b376';head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
files=subprocess.check_output(['git','ls-files','warpower','tests','examples','benchmarks','README.md','pyproject.toml'],text=True).splitlines()
changed=subprocess.check_output(['git','diff','--name-only',base,head],text=True).splitlines()
expected=['tests/test_topology_recovery.py','warpower/_exterior_band.py','warpower/_extraction.py','warpower/_mesher.py','warpower/_quality_moves.py','warpower/_topology_recovery.py'];assert sorted(changed)==expected
unchanged=[]
for n in files:
 if n not in changed:
  assert (root/n).read_bytes()==subprocess.check_output(['git','show',base+':'+n]);unchanged.append(n)
def methods(code):
 return {(c.name,m.name):ast.dump(m,include_attributes=False) for c in ast.parse(code).body if isinstance(c,ast.ClassDef) for m in c.body if isinstance(m,ast.FunctionDef)}
for n in ['warpower/_extraction.py','warpower/_quality_moves.py']:
 before=methods(subprocess.check_output(['git','show',base+':'+n],text=True));after=methods((root/n).read_text())
 assert before.keys()==after.keys()
 assert all(before[k]==after[k] for k in before if k[1]!='__init__')
for n in ['warpower/_extraction.py','warpower/_quality_moves.py','warpower/_mesher.py']:
 def funcs(code):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(code).body if isinstance(x,ast.FunctionDef)}
 assert funcs(subprocess.check_output(['git','show',base+':'+n],text=True))==funcs((root/n).read_text())
report=dict(scope=__doc__,baseline=base,runtime=head,changed_files=changed,unchanged_files=unchanged,source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files},readme_unchanged=True,existing_extraction_quality_and_mesher_kernels_unchanged=True,extraction_and_quality_runtime_methods_unchanged=True,independent_diagram_and_delaunay_sources_unchanged=True)
out.write_text(json.dumps(report,indent=2)+'\n');print(len(files),'pinned files;',len(unchanged),'unchanged; standalone sources unchanged')
