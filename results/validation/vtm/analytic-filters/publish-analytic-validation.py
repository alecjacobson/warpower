"""Publish closed evidence only; generated caches and unrelated metadata stay private."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/analytic-filters'
runtime='0590626a96ad3e4bf8df18d7fb5a0184e5ce21b1';baseline='fb795ac1b83f4bbf0c5759fb6f800ff41e1c403e'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==runtime
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
assert '694 passed' in (w/'analytic-full-suite.log').read_text()
final=json.loads((w/'analytic-final-independent.json').read_text());assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(abs(r['relative_change'])<0.01 for r in final['standalone'].values())
assert all(r['after_ms']<r['before_ms'] for r in final['cases'].values())
assert len(json.loads((w/'analytic-face-broad-independent.json').read_text()))==6
rows=json.loads((w/'analytic-audit-commands.json').read_text());assert len(rows)==19
assert all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in rows[:18])
memory=json.loads((w/'analytic-memory-coverage.json').read_text());assert memory['all_those_scopes_zero_error']
assert memory['strict_route_control']['exit_code']==99 and memory['strict_route_control']['all_printed_invalid_diagnostics_match_prior_cub_histogram_allocation_order']
native=json.loads((w/'analytic-native-commands.json').read_text());assert len(native)==3 and all(r['exit_code']==0 for r in native)
assert 'AUDIT_VALIDATION_PENDING' not in (root/'docs/analytic-facet-search.md').read_text()
for name in ['full','loop']:
    audit=json.loads((w/f'analytic-{name}-native-audit.json').read_text())
    assert set(audit['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
    assert set(audit['gpu_memcpy_kinds']) <= {'CUDA_MEMCPY_KIND_DTOD'}
    assert set(audit['dot_node_types']) <= {'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
    assert audit['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
    assert audit['cuda_api_calls'].get('cuCtxSynchronize')==1
assert not out.exists();out.mkdir(parents=True)
files=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(p,relative=None):
    relative=Path(relative or p.name);dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists(),dest
    compressed=p.suffix in ['.off','.vtu','.nsys-rep']
    record=dict(original=str(p.relative_to(root)),original_sha256=sha(p))
    if compressed:
        dest=dest.with_name(dest.name+'.gz');data=p.read_bytes();dest.write_bytes(gzip.compress(data,mtime=0))
        assert gzip.decompress(dest.read_bytes())==data
        record['encoding']='gzip; decompress to recover exact original bytes'
    else:os.link(p,dest)
    record.update(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=sha(dest));files.append(record)
exclude_prefixes=('analytic-pr-','analytic-parent-','analytic-filter-issue','analytic-filter-progress','analytic-push','analytic-lifetime-issue')
exclude_names={'analytic-compiler-cache-relocation.json'}
for p in sorted(w.iterdir()):
    if not p.is_file() or not p.name.startswith('analytic-'):continue
    if p.name.startswith(exclude_prefixes) or p.name in exclude_names:continue
    if p.suffix in ['.py','.json','.log','.npz','.gz','.nsys-rep','.md']:copy(p)
extra=['analytic_face_search.py','install_analytic_face.py','test-analytic-existing.py','test_analytic_filters.py','test_analytic_queries.py','integrate-analytic-filters.py','relocate-analytic-compiler-cache.py','repeat-analytic-controls.py','deduplicate-analytic-inputs.py','boundary-timing-commands.json','shewchuk-predicates.c','publish-analytic-validation.py']
extra +=[p.name for p in sorted(w.iterdir()) if p.is_file() and p.suffix=='.py' and p.name.startswith(('run-analytic','check-analytic','analyze-analytic'))]
for name in extra:copy(w/name)
for directory in ['analytic-face-pilot','analytic-face-broad','analytic-face-broad-recheck','analytic-face-broad-recheck2','analytic-hand-control-repeat','analytic-cavity-control-repeat','analytic-disconnected-control-repeat','analytic-final']:
    for p in sorted((w/directory).rglob('*')):
        if p.is_file():copy(p,p.relative_to(w))
canonical=root/'results/examples/vtm/armadillo-uniform/input.off'
assert sha(canonical)=='b4e07d0cea859ec879b46ed632d11206db0d63705c2d4399edd88ead1e1a601a'
shared_inputs=[dict(original=r['removed_exact_duplicate'],canonical=str(canonical.relative_to(root)),sha256=r['sha256']) for r in json.loads((w/'analytic-input-deduplication.json').read_text())]
shared_helpers=[]
for name in ['check-large-reuse-proof-pairs.py','large-reuse-isolated-test.py']:
    p=root/'results/validation/vtm/large-fragment-reuse'/name
    assert p.read_bytes()==(w/name).read_bytes()
    shared_helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
sources=['warpower/_analytic_predicates.py','warpower/_proved_completion.py','tests/test_analytic_predicates.py','tests/test_analytic_face_search.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
manifest=dict(runtime=runtime,baseline=baseline,scope='Certified analytic FP64 filters only in VTM exposed-face search. Complete GPU gains; no CPU speedup claim. Full routed allocation-order limitation retained, not relabeled a pass.',source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared_inputs,shared_helpers=shared_helpers,reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},reference_predicates_url='https://www.cs.cmu.edu/afs/cs/project/quake/public/code/predicates.c',reference_predicates_sha256=sha(w/'shewchuk-predicates.c'),files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for record in files:assert sha(root/record['path'])==record['sha256']
print('verified',len(files),'artifacts',sum(r['bytes'] for r in files),'bytes',flush=True)
