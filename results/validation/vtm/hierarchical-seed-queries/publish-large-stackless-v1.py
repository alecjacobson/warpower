"""Publish production hierarchical-query evidence with immutable local aliases."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/hierarchical-seed-queries'
assert not (out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
runtime=json.loads((w/'large-stackless-full-suite-v1.json').read_text())['source'];baseline='5998b1384e97c04532fd5f928feb0e09c2fa702a'
assert subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])==b''
assert '1227 passed' in (w/'large-stackless-full-suite-v1.log').read_text()
checks=json.loads((w/'large-stackless-final-v1-independent.json').read_text());assert checks['complete'] and len(checks['cases'])==6 and len(checks['standalone'])==5
assert all(r['comparison']['points_bitwise_equal'] for r in checks['cases'].values())
mem=json.loads((w/'large-stackless-memory-v1-commands.json').read_text());assert len(mem)==24 and all(r['exit_code']==0 and r['zero_memcheck_errors'] and r['expected_tests']==1 for r in mem)
qualification=json.loads((w/'large-stackless-qualification-v1.json').read_text());assert len(qualification)==3 and all(r['exit_code']==0 for r in qualification)
assert json.loads((w/'large-stackless-native-loop-independent-v1.json').read_text())['complete']
for label in ['full','loop']:
 audit=json.loads((w/f'large-stackless-{label}-native-v1-audit.json').read_text())
 assert audit['gpu_kernel_activities']>0 and set(audit['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'}
 assert set(audit['dot_copy_kinds'])<={'DtoD (DEVICE to DEVICE)'}
 assert all(not any(word in name.lower() for word in ['alloc','free','memcpy']) for name in audit['cuda_api_calls'])
assert json.loads((w/'large-stackless-pilot-v1-independent.json').read_text())['complete']
source_audit=json.loads((w/'large-stackless-production-source-audit-v1.json').read_text())
for name,digest in source_audit['source_sha256'].items():assert sha(root/name)==digest,name
prefixes=('large-stackless','large_stackless','run-large-stackless','check-large-stackless','profile-large-stackless','test_large_stackless','test-large-stackless','audit-large-stackless','prepare-large-stackless','promote-large-stackless','write-large-stackless')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-pr-','-issue-','-doc-draft-'])]
folders=['large-stackless-pilot-v1','large-stackless-final-v1']
for name in folders:files += [p for p in (w/name).rglob('*') if p.is_file()]
files += [Path(__file__).resolve()];files=sorted(set(files))
meta=dict(scope=__doc__,runtime=runtime,baseline=baseline,device='NVIDIA L40',warp='1.19.0',files=[],source_sha256=source_audit['source_sha256'],shared_helpers=[],shared_prior_evidence=[],limitations=['GPU-versus-GPU traversal gains do not meet the comparable-quality native CPU speed goal.','Native trace and sanitizer durations are diagnostic, not clean timing samples.','Fresh solve timing excludes setup, compilation and transfers.','Previous Armadillo extraction failures are unchanged and not presented as successful meshes.','The first captured-loop launch includes deferred graph instantiation and upload before its single graph launch; driver-internal preparation is not claimed host-free or allocation-free.'],reproduction=['Restore original .work/vtm paths and shared helpers/prior artifacts in checkouts of the pinned source commits; replace absolute workspace root as needed. Drivers refuse output overwrites.','Private probes use baseline5998b13; production validation uses the recorded runtime commit. Production imports no private prototype.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU process at a time; no heavy CPU work during clean timing.','Build native CGAL validators from benchmarks/cpu; binary hashes are retained. /opt contains only regenerable cache/scratch.'])
for p in files:
 assert p.is_file() and not p.is_symlink(),p
 target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True)
 assert not target.exists(),target
 os.link(p,target);target.chmod(0o444);assert target.stat().st_ino==p.stat().st_ino
 meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py')]:
 p=w/original;q=root/'results/validation/vtm'/canonical;assert sha(p)==sha(q);meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
prior=root/'results/validation/vtm/extended-fragment-reuse/manifest.json';old=json.loads(prior.read_text());meta['shared_prior_evidence'].append(dict(path=rel(prior),sha256=sha(prior)))
needed=('extended-reuse-final-v1','extended-reuse-pilot-v1','extended-reuse-collection-v1','profile-extended-reuse','run-extended-reuse','check-extended-reuse')
for row in old['files']:
 if any(x in row['original'] for x in needed):
  assert sha(root/row['path'])==row['sha256'];meta['shared_prior_evidence'].append(row)
meta['native_binary_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
for name in folders:
 for row in json.loads((w/name/'lossless-archives.json').read_text()):
  packed=Path(row['path']);assert sha(packed)==row['sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==row['raw_sha256']
for label in ['profile','full-native','loop-native']:
 stem=w/f'large-stackless-{label}-v1';r=json.loads(stem.with_name(stem.name+'-kernels.json').read_text());packed=stem.with_suffix('.nsys-rep.gz')
 assert sha(packed)==r['trace_gz_sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==r['trace_raw_sha256']
 dot=stem.with_suffix('.dot.gz');audit=json.loads(stem.with_name(stem.name+'-audit.json').read_text());assert hashlib.sha256(gzip.decompress(dot.read_bytes())).hexdigest()==audit['dot_sha256']
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
for r in meta['files']:assert sha(root/r['path'])==sha(root/r['original'])==r['sha256']
result=dict(files=len(files),manifest_sha256=sha(p),all_artifact_source_archive_hashes_verified=True,runtime=runtime)
(w/'verified-large-stackless-publication-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
