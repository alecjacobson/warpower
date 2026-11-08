"""Publish certified capacity extension evidence, including unchanged failed cases."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/extended-fragment-reuse'
assert not (out/'manifest.json').exists()
runtime='f8e2834140274936f5f8c525edc21b7da4e27088';baseline='689fd31382c490251b953046176b15ccc35e5753'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
assert subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])==b''
assert '1121 passed' in (w/'extended-reuse-full-suite-v1.log').read_text()
checks=json.loads((w/'extended-reuse-final-v1-independent.json').read_text());assert checks['complete'] and len(checks['cases'])==10 and len(checks['standalone'])==5
assert all(r['exit_code']==0 and r['zero_memcheck_errors'] and r['expected_tests']==1 for r in json.loads((w/'extended-reuse-memory-v2-commands.json').read_text()))
assert len(json.loads((w/'extended-reuse-memory-v2-commands.json').read_text()))==30
qualification=json.loads((w/'extended-reuse-qualification-v2.json').read_text())
assert len(qualification)==3 and all(r['exit_code']==0 for r in qualification)
assert json.loads((w/'extended-reuse-native-loop-independent-v1.json').read_text())['complete']
for label in ['full','loop']:
    audit=json.loads((w/f'extended-reuse-{label}-native-v1-audit.json').read_text())
    assert audit['gpu_kernel_activities']>0 and set(audit['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'}
    assert set(audit['dot_copy_kinds'])<={'DtoD (DEVICE to DEVICE)'}
    assert all(not any(word in name.lower() for word in ['alloc','free','memcpy']) for name in audit['cuda_api_calls'])
assert json.loads((w/'extended-reuse-pilot-v1-independent.json').read_text())['complete']
assert json.loads((w/'extended-reuse-gallery-pilot-v2-independent.json').read_text())['complete']
assert json.loads((w/'extended-reuse-dense-failure-v1.json').read_text())['complete']
source_audit=json.loads((w/'extended-reuse-source-audit-v1.json').read_text())
for name,digest in source_audit['source_sha256'].items():assert sha(root/name)==digest,name
prefixes=('extended-reuse','extended_reuse','run-extended-reuse','check-extended-reuse','profile-extended-reuse','test_extended_reuse','proposed-test-extended-reuse','audit-extended-reuse','prepare-extended-reuse','capture-graded-armadillo-8192','analyze-graded-armadillo-8192','analyze-graded-armadillo-transaction','analyze-graded-armadillo-pinched','graded-armadillo-8192')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-pr-','-issue-','-doc-draft-','-stackless-followup-'])]
folders=['extended-reuse-pilot-v1','extended-reuse-gallery-pilot-v1','extended-reuse-gallery-dense-v1','extended-reuse-final-v1','graded-armadillo-8192-diagnostic-v1','graded-armadillo-8192-transaction-v1']
for name in folders:files += [p for p in (w/name).rglob('*') if p.is_file()]
files += [Path(__file__).resolve(),w/'write-extended-reuse-doc-v1.py'];files=sorted(set(files))
meta=dict(scope=__doc__,runtime=runtime,baseline=baseline,device='NVIDIA L40',warp='1.19.0',files=[],source_sha256=source_audit['source_sha256'],shared_helpers=[],shared_prior_evidence=[],limitations=['GPU implementation speedups are not comparable-quality native CPU speedups; the overall CPU goal remains open.','Both versions fail uniform Armadillo8192 at quadrature1 and graded Armadillo8192 at quadrature1/2. Failed-case timing is not successful-mesh evidence; one denser baseline also has quota throttling.','A diagnostic six-cell removal repairs boundary edges but leaves a disconnected vertex link. Existing rollback guards are unchanged; this is not an accepted repaired mesh.','Native trace and sanitizer durations are diagnostic, not clean performance results.','Fresh solve timing excludes setup, compilation and transfers. Retained baseline measurements and private prototypes are separately identified.'],reproduction=['Restore the recorded original .work/vtm paths and shared helper/prior artifacts in separate checkouts of each pinned commit. Replace the workspace root as needed; drivers refuse overwriting outputs.','Private prototypes require their recorded pre-extension source. Production validation requires runtime f8e2834. No private prototype is imported by production.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU process at a time, no heavy CPU work during clean timing.','Build native CGAL validation tools from benchmarks/cpu; their binary hashes are retained. Recreate /opt scratch freely; unique artifacts and source stay persistent.'])
for p in files:
    assert p.is_file() and not p.is_symlink(),p
    target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():os.link(p,target)
    target.chmod(0o444);assert target.stat().st_ino==p.stat().st_ino
    meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py')]:
    p=w/original;q=root/'results/validation/vtm'/canonical;assert sha(p)==sha(q);meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
prior=root/'results/validation/vtm/fixed-site-current/manifest.json';old=json.loads(prior.read_text());meta['shared_prior_evidence'].append(dict(path=rel(prior),sha256=sha(prior)))
for row in old['files']:
    if 'fixed-site-current-larger-v1' in row['original']:
        assert sha(root/row['path'])==row['sha256'];meta['shared_prior_evidence'].append(row)
meta['native_binary_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
for name in folders:
    record=w/name/'lossless-archives.json'
    if record.exists():
        for row in json.loads(record.read_text()):
            packed=Path(row['path']);assert sha(packed)==row['sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==row['raw_sha256']
for label in ['profile','full-native','loop-native']:
    stem=w/f'extended-reuse-{label}-v1';r=json.loads(stem.with_name(stem.name+'-kernels.json').read_text());packed=stem.with_suffix('.nsys-rep.gz')
    assert sha(packed)==r['trace_gz_sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==r['trace_raw_sha256']
    dot=stem.with_suffix('.dot.gz');audit=json.loads(stem.with_name(stem.name+'-audit.json').read_text());assert hashlib.sha256(gzip.decompress(dot.read_bytes())).hexdigest()==audit['dot_sha256']
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
for r in meta['files']:assert sha(root/r['path'])==sha(root/r['original'])==r['sha256']
result=dict(files=len(files),manifest_sha256=sha(p),all_artifact_source_archive_hashes_verified=True,runtime=runtime)
(w/'verified-extended-reuse-publication-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
