"""Publish completed #172 evidence; verify every copied or compressed artifact."""
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/grouped-seed-queries'
revisions=json.loads((w/'prepared-group-runtime.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==revisions['runtime']
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
assert '786 passed' in (w/'prepared-group-full-suite.log').read_text()
assert '119 passed' in (w/'prepared-group-focused-tests.log').read_text()
final=json.loads((w/'prepared-group-final-independent.json').read_text())
assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(abs(r['relative_change'])<0.02 for r in final['standalone'].values())
assert len(json.loads((w/'prepared-group-broad-independent.json').read_text())['cases'])==6
counter=json.loads((w/'prepared-group-counter-equivalence.json').read_text())
assert counter['points_bitwise_equal'] and all(counter['work_equal'].values())
assert counter['totals']['calls']==624 and counter['totals']['cells']==12882912
assert counter['totals']['groups_skipped']==1402978068 and counter['totals']['exact_fallback_calls']==0
memory=json.loads((w/'prepared-group-memory-commands.json').read_text())
assert len(memory)==6 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
native=json.loads((w/'prepared-group-native-commands.json').read_text())
assert len(native)==3 and all(r['exit_code']==0 for r in native)
for name in ['full','loop']:
    d=json.loads((w/f'prepared-group-{name}-native-audit.json').read_text())
    assert set(d['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
    assert set(d['gpu_memcpy_kinds']) <= {'CUDA_MEMCPY_KIND_DTOD'}
    assert set(d['dot_node_types']) <= {'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
    assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
    assert d['cuda_api_calls'].get('cuCtxSynchronize')==1
assert 'VALIDATION_PENDING' not in (root/'docs/grouped-seed-queries.md').read_text()
assert not out.exists();out.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
canonical=root/'results/examples/vtm/armadillo-uniform/input.off'
canonical_hash=sha(canonical)
assert canonical_hash=='b4e07d0cea859ec879b46ed632d11206db0d63705c2d4399edd88ead1e1a601a'
files=[];shared_inputs=[];removed_duplicates=[]
def copy(p,relative=None):
    relative=Path(relative or p.name)
    original_hash=sha(p)
    if p.name=='input.off' and original_hash==canonical_hash:
        shared_inputs.append(dict(original=str(p.relative_to(root)),canonical=str(canonical.relative_to(root)),sha256=original_hash))
        return
    dest=out/relative;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists(),dest
    record=dict(original=str(p.relative_to(root)),original_sha256=original_hash)
    compressed=p.suffix in ['.off','.vtu','.nsys-rep']
    if compressed:
        dest=dest.with_name(dest.name+'.gz');data=p.read_bytes()
        dest.write_bytes(gzip.compress(data,mtime=0))
        assert gzip.decompress(dest.read_bytes())==data
        record['encoding']='gzip; decompress to recover exact original bytes'
    else:
        os.link(p,dest)
    record.update(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=sha(dest));files.append(record)
    if compressed:
        # All checks are finished. The exact persistent archive retains unique
        # data; remove only its verified raw duplicate to stay within quota.
        removed_duplicates.append(dict(raw=str(p.relative_to(root)),archive=str(dest.relative_to(root)),sha256=original_hash))
        p.unlink()
exclude_prefixes=('prepared-group-pr-','prepared-group-push','prepared-group-parent-','prepared-group-git-storage','prepared-group-pack-')
exclude_names={'prepared-group-verified-storage-cleanup.json','prepared-group-lossless-text-archives.json','prepared-group-profile-storage.json','prepared-group-native-duplicate-cleanup.json'}
prefixes=('prepared-group-',)
for p in sorted(w.iterdir()):
    if not p.is_file() or not p.name.startswith(prefixes):continue
    if p.name.startswith(exclude_prefixes) or p.name in exclude_names:continue
    if p.suffix in ['.py','.json','.log','.npz','.gz','.nsys-rep','.md']:copy(p)
extra=['repeat-prepared-group-hand.py','prepared_group_counters.py','prepared_group_queries.py','test_prepared_group_masks.py','test_prepared_group_bounds.py','publish-prepared-group-validation.py']
script_prefixes=('run-prepared-group','check-prepared-group','analyze-prepared-group','test-prepared-group')
extra += [p.name for p in sorted(w.iterdir()) if p.is_file() and p.suffix=='.py' and p.name.startswith(script_prefixes)]
assert len(extra)==len(set(extra))
for name in extra:copy(w/name)
for name in ['prepared-group-pilot','prepared-group-broad','prepared-group-final','prepared-group-hand-repeat']:
    for p in sorted((w/name).rglob('*')):
        if p.is_file():copy(p,p.relative_to(w))
shared_helpers=[]
for private,public in [('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json'),('prepared-query-final/commands.json','prepared-point-queries/prepared-query-final/commands.json')]:
    p=root/'results/validation/vtm'/public
    assert p.read_bytes()==(w/private).read_bytes()
    shared_helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
sources=['warpower/_grouped_seeds.py','warpower/_facet_seeds.py','warpower/_prepared_seeds.py','tests/test_grouped_query_bounds.py','tests/test_grouped_query_masks.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
manifest=dict(**revisions,scope='Current-input Morton groups in private VTM seed queries. Exact group min/max boxes, existing conservative sphere-box and point/symbolic tests. Adds 24 bytes per group and one preparation launch per invocation; no temporal proof or CPU outperformance claim. The existing routed full allocation-order instrumentation limitation remains.',source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared_inputs,shared_helpers=shared_helpers,reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},verified_raw_duplicates_removed=removed_duplicates,files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print('Verified',len(files),'artifacts;',sum(r['bytes'] for r in files),'bytes',flush=True)
