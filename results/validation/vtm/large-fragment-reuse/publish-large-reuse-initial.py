"""Publish the integrated larger-route adoption only after every validation gate."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/large-fragment-reuse'
read=lambda p:json.loads((w/p).read_text())
checks=read('large-reuse-final-independent.json')
assert len(checks['cases'])==10 and len(checks['standalone'])==5
assert '650 passed' in (w/'large-reuse-full-suite.log').read_text()
assert '7 passed' in (w/'large-reuse-new-tests.log').read_text()
s=(w/'large-reuse-memcheck.log').read_text()
assert s.count('1 passed')==7 and s.count('ERROR SUMMARY: 0 errors')==7
lifetime=read('large-reuse-lifetime-analysis.json')
assert len(lifetime['tests'])==7 and all(r['api_memory_and_afterfree_pass'] for r in lifetime['tests'].values())
assert len(lifetime['controls'])==3 and len(lifetime['strict_failures'])>=4
audit=read('large-reuse-native-adoption-commands.json')
assert len(audit)==3 and all(r['exit_code']==0 for r in audit)
for kind in ['full','loop']:
 assert read(f'large-reuse-{kind}-native-audit.json')['gpu_kernel_activities']>0
assert read('large-reuse-loop-native.json')['phase']==8
assert read('large-reuse-full-native.json')['valid']==1
assert not subprocess.check_output(['git','diff','f8b5cfa','--','warpower','tests'],text=True)
isolation=read('large-reuse-isolation.json')
assert all(v['all_equal'] for v in isolation['changed_module_device_ast'].values())
assert not out.exists()
out.mkdir(parents=True)
files=[];input_refs=[]
def copy(source,relative=None,compress=False):
 source=Path(source);dest=out/(relative or source.name)
 if compress:dest=dest.with_name(dest.name+'.gz')
 assert not dest.exists(),dest
 dest.parent.mkdir(parents=True,exist_ok=True)
 data=source.read_bytes()
 if compress:dest.write_bytes(gzip.compress(data,mtime=0))
 else:os.link(source,dest)  # Immutable artifacts share persistent storage, avoiding duplicate quota use.
 check=gzip.decompress(dest.read_bytes()) if compress else dest.read_bytes()
 assert check==data,dest
 files.append(dict(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),uncompressed_sha256=hashlib.sha256(data).hexdigest()))
# Retain every private wrapper, probe, failed export/helper, log and diagnostic.
skip={'large-reuse-pr-draft.md','large-reuse-161-initial-comment.md','large-reuse-161-proof-comment.md','large-reuse-adoption-plan.md','large-reuse-baseline-checkout.log'}
for p in sorted(w.iterdir()):
 if p.is_file() and 'large-reuse' in p.name and p.name not in skip:
  copy(p,compress=p.suffix in ['.nsys-rep','.dot','.sqlite'])
for name in ['seed_tree_queries.py','test-seed-tree-existing.py','test_seed_tree_large.py','seed-tree-existing-tests.log','seed-tree-large-tests.log']:
 copy(w/name)
for folder in ['large-reuse-proof-pairs','large-reuse-final']:
 for p in sorted((w/folder).rglob('*')):
  if not p.is_file():continue
  relative=p.relative_to(w)
  if p.name=='input.off':
   canonical=root/('results/examples/vtm/armadillo-uniform/input.off' if 'armadillo-' in str(p) else 'results/examples/vtm/graded-torus/input.off')
   assert canonical.exists(),canonical
   assert p.read_bytes()==canonical.read_bytes()
   input_refs.append(dict(original=str(p.relative_to(root)),canonical=str(canonical.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  else:copy(p,relative,compress=p.suffix in ['.vtu','.off'])
# The uniform 4k control is shared with the already-published extraction experiment.
control=root/'results/validation/vtm/extraction-split-probe/extraction-split-4096'
for ext in ['.json','.npz','.log']:
 assert (w/'extraction-split-4096'/('control'+ext)).read_bytes()==(control/('control'+ext)).read_bytes()
sources=['warpower/_active_delaunay.py','warpower/_reuse_delaunay.py','warpower/_proved_completion.py','warpower/_facet_seeds.py','warpower/_prepared_seeds.py','warpower/_facet_completion.py','tests/test_large_reuse.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/mesh_quality.py','benchmarks/graph_capture.py','benchmarks/cpu/check_delaunay.cpp','benchmarks/cpu/surface_intersections.cpp','benchmarks/cpu/surface_distance.cpp']
manifest=dict(runtime='f8b5cfaf2625f264dd95448a272849b88c0c353b',baseline='cca7a1de4f5ddf3c190f321cfda53c66f0525948',baseline_runtime='9228a3e7b3804b0cf1b191976477e35c25dc47ef',production_scope=isolation['production_changed'],source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources},native_reference_binary_sha256={p:hashlib.sha256((w/p).read_bytes()).hexdigest() for p in ['check-delaunay','surface-reference','surface-distance']},shared_uniform4096_control=str(control.relative_to(root)),shared_inputs=input_refs,lifetime_validation_scope=lifetime['coverage'],scope='Bounded active full-route reuse through 4096 sites. Existing same-current-input proofs consumed beyond former query threshold. All numerical device kernels unchanged. No cross-input proof cache. CPU outperformance goal remains unmet.',files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
print('published',len(files),'hash-verified artifacts',sum(r['bytes'] for r in files),'bytes',flush=True)
