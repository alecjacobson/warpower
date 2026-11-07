"""Archive rejected private search experiments; no runtime adoption."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/facet-pruning-probes'
assert not out.exists();out.mkdir(parents=True)
assert not subprocess.check_output(['git','diff','5f3e6d7','--','warpower','tests','benchmarks','examples','README.md'],text=True)
for prefix in ['facet-tree','facet-linear']:
 assert '51 passed' in (w/f'{prefix}-existing-tests.log').read_text()
 assert '36 passed' in (w/f'{prefix}-query-tests.log').read_text()
 rows=json.loads((w/f'{prefix}-pilot/commands.json').read_text())
 assert len(rows)==4 and all(r['exit_code']==0 and r['quota_clean'] for r in rows)
 checks=json.loads((w/f'{prefix}-pilot-independent.json').read_text())
 assert len(checks)==2 and all(r['after_ms']>r['before_ms'] and r['before_pool_bytes']==r['after_pool_bytes'] for r in checks.values())
files=[]
def copy(p,relative=None):
 dest=out/(relative or p.name);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
 os.link(p,dest)
 files.append(dict(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
for p in sorted(w.iterdir()):
 if p.is_file() and p.name.startswith(('facet-tree-','facet-linear-')) and p.suffix in ['.json','.log']:copy(p)
for name in ['facet_tree_search.py','facet_linear_box_search.py','install_facet_tree.py','install_facet_linear.py','test-facet-tree-existing.py','test-facet-linear-existing.py','test_facet_tree_queries.py','test_facet_linear_queries.py','run-facet-tree.py','run-facet-linear.py','run-facet-tree-pilot.py','run-facet-linear-pilot.py','check-facet-tree-pilot.py','check-facet-linear-pilot.py','publish-facet-pruning.py']:
 copy(w/name)
for folder in ['facet-tree-pilot','facet-linear-pilot']:
 for p in sorted((w/folder).iterdir()):
  if p.is_file():copy(p,p.relative_to(w))
sources=['warpower/_proved_completion.py','warpower/_facet_completion.py','warpower/_sphere_bounds.py','warpower/_delaunay_certificate.py','warpower/_filtered_predicates.py','benchmarks/mesher.py','benchmarks/mesh_quality.py','benchmarks/surface_distance.py']
helper=root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py'
assert helper.read_bytes()==(w/'check-large-reuse-proof-pairs.py').read_bytes()
manifest=dict(runtime='5f3e6d71e93a492aeead9fc100ba8ef89097f177',scope='Two private conservative search variants rejected after slower complete-solve pilots. No production/tests/benchmark/example/README changes; no adoption audit claimed.',source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},shared_validation_helper=dict(path=str(helper.relative_to(root)),sha256=hashlib.sha256(helper.read_bytes()).hexdigest()),native_reference_binary_sha256={p:hashlib.sha256((w/p).read_bytes()).hexdigest() for p in ['check-delaunay','surface-distance']},files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
print('verified',len(files),'artifacts',sum(r['bytes'] for r in files),'bytes')
