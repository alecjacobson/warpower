"""Publish completed #185 evidence, preserving all source variants and checks."""
import gzip, hashlib, json, os, subprocess
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/bounded-hull-support'
runtime='6eb1882d8af126e65ee88fe0cbdd5f3520f8cc87'
baseline='94056844cf9d55a44b4454744b0e07dcd20fd9a7'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==runtime
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
for name,count in [('existing',69),('masks',36),('integrated-focused',105),('full-suite',831)]:
    assert f'{count} passed' in (w/f'bounded-hull-support-{name}.log').read_text()
pilot=json.loads((w/'bounded-hull-support-pilot-independent.json').read_text());assert len(pilot)==2
for folder,count in [('bounded-hull-support-pilot',4),('bounded-hull-support-final',26)]:
    jobs=json.loads((w/folder/'commands.json').read_text())
    assert len(jobs)==count and all(r['exit_code']==0 and r['quota_clean'] for r in jobs)
final=json.loads((w/'bounded-hull-support-final-independent.json').read_text())
assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(abs(r['relative_change'])<.02 for r in final['standalone'].values())
for row in final['cases'].values():
    assert all(row['work_equal'].values()) and row['independent']['full_hull']['valid']
    assert row['independent']['all_boundary_vertex_links_single_cycles']
    if 'before_pool_bytes' in row:assert row['before_pool_bytes']==row['after_pool_bytes']
    assert row['after_ms']<row['before_ms']
memory=json.loads((w/'bounded-hull-support-memory-commands.json').read_text())
assert len(memory)==6 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
native=json.loads((w/'bounded-hull-support-native-commands.json').read_text())
assert len(native)==2 and [r['exit_code'] for r in native]==[0,1]
assert "AttributeError: module 'numpy' has no attribute 'bounded'" in (w/'bounded-hull-support-loop-native.log').read_text()
resumed=json.loads((w/'bounded-hull-support-native-resume-commands.json').read_text())
assert len(resumed)==2 and all(r['exit_code']==0 for r in resumed)
for name in ['full','loop']:
    d=json.loads((w/f'bounded-hull-support-{name}-native-audit.json').read_text())
    assert set(d['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
    assert set(d['gpu_memcpy_kinds'])<={'CUDA_MEMCPY_KIND_DTOD'}
    assert set(d['dot_node_types'])<={'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
    assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
    assert d['cuda_api_calls'].get('cuCtxSynchronize')==1
assert 'PENDING' not in (root/'docs/bounded-hull-support.md').read_text()
assert not out.exists();out.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
canonical=root/'results/examples/vtm/armadillo-uniform/input.off';canonical_hash=sha(canonical)
files=[];shared=[]
def copy(p):
    original_hash=sha(p)
    if p.name=='input.off' and original_hash==canonical_hash:
        shared.append(dict(original=str(p.relative_to(root)),canonical=str(canonical.relative_to(root)),sha256=original_hash));return
    dest=out/p.relative_to(w);dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists();os.link(p,dest)
    files.append(dict(original=str(p.relative_to(root)),path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=original_hash))
prefixes=('bounded-hull-support','bounded_hull_support','build_bounded_hull_support',
          'test_bounded_hull_support','test-bounded-hull-support','run-bounded-hull-support',
          'check-bounded-hull-support','analyze-bounded-hull-support','publish-bounded-hull-support',
          'write-bounded-hull-support')
for p in sorted(w.iterdir()):
    if not p.name.startswith(prefixes):continue
    if p.is_file() and p.suffix!='.md':copy(p)
    elif p.is_dir():
        for child in sorted(p.rglob('*')):
            if child.is_file():copy(child)
helpers=[]
for private,public in [('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),
                       ('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),
                       ('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json')]:
    p=root/'results/validation/vtm'/public
    assert p.read_bytes()==(w/private).read_bytes()
    helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
profile=root/'results/validation/vtm/bounded-predicate-filters/float32-native-audit-summary.json'
helpers.append(dict(path=str(profile.relative_to(root)),sha256=sha(profile)))
sources=['warpower/_proved_completion.py','warpower/_bounded_predicates.py','warpower/_analytic_predicates.py',
         'benchmarks/bounded_predicate_proof.py','tests/test_bounded_hull_support.py','tests/test_support_reuse.py',
         'benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
manifest=dict(baseline=baseline,runtime=runtime,
    scope='Existing bounded binary32 and analytic binary64 orientation stages in private VTM hull-support leaf queries, followed by original Ball64/exact fallback. One runtime call change; unchanged storage/launch shape and standalone defaults. No CPU-outperformance claim; routed full allocation-order checker limitation remains documented.',
    source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared,shared_helpers=helpers,
    reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},
    lossless_archives=json.loads((w/'bounded-hull-support-lossless-archives.json').read_text()),files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in files:assert sha(root/row['path'])==row['sha256']
print('Verified',len(files),'artifacts;',sum(r['bytes'] for r in files),'bytes',flush=True)
