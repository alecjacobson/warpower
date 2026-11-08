"""Publish shared new-cell geometry and the unadopted strict/exact experiment."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/prepared-new-cell-queries'
assert not (out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
def read(n):return json.loads((w/n).read_text())
runtime=read('prepared-new-cell-full-suite-v1.json')['source'];baseline='a4f7215fbf99249d11f765a4902c586daa606829'
assert subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])==b''
assert '1279 passed' in (w/'prepared-new-cell-full-suite-v1.log').read_text()
assert read('prepared-new-cell-throttled-pair-independent-v1.json')['complete']
assert read('prepared-new-cell-hand-repeat-v1/comparison.json')['complete']
hand=read('prepared-new-cell-hand-trace-v1/nonregression-conclusion-v1.json');assert hand['complete'] and hand['modified_modules_not_loaded'] and hand['kernel_signature_identical']
assert read('prepared-new-cell-hand-import-audit-v1/comparison.json')['complete']
checks=read('prepared-new-cell-final-v2-independent.json');assert checks['complete'] and len(checks['cases'])==7 and len(checks['standalone'])==5
assert all(r['comparison']['points_bitwise_equal'] for r in checks['cases'].values())
assert read('prepared-new-cell-grid-v1-independent.json')['complete']
mem=read('prepared-new-cell-memory-v1-commands.json');assert len(mem)==28 and all(r['exit_code']==0 and r['zero_memcheck_errors'] and r['expected_tests']==1 for r in mem)
qualification=read('prepared-new-cell-qualification-v1.json');assert len(qualification)==3 and all(r['exit_code']==0 for r in qualification)
assert read('prepared-new-cell-native-loop-independent-v1.json')['complete']
for label in ['full','loop']:
 audit=read(f'prepared-new-cell-{label}-native-v1-audit.json')
 assert audit['gpu_kernel_activities']>0 and set(audit['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'}
 assert set(audit['dot_copy_kinds'])<={'DtoD (DEVICE to DEVICE)'}
 assert all(not any(word in name.lower() for word in ['alloc','free','memcpy']) for name in audit['cuda_api_calls'])
for version in [1,2]:
 assert read(f'prepared-new-cell-pilot-v{version}-independent.json')['complete']
assert read('prepared-new-cell-grid-v2-independent.json')['complete']
assert read('prepared-new-cell-simple-guards-v1-independent.json')['complete']
assert read('prepared-new-cell-simple-grid-retry-v1.json')['exit_code']==0
source_audit=read('prepared-new-cell-production-source-audit-v1.json')
for name,digest in source_audit['source_sha256'].items():assert sha(root/name)==digest,name
prefixes=('prepared-new-cell','prepared_new_cell','run-prepared-new-cell','check-prepared-new-cell','profile-prepared-new-cell','test_prepared_new_cell','test-prepared-new-cell','build-prepared-new-cell','draft-prepared-new-cell','promote-prepared-new-cell','write-prepared-new-cell','bench-prepared-new-cell','audit-prepared-new-cell','trace-prepared-new-cell','analyze-prepared-new-cell','conclude-prepared-new-cell','prepare-prepared-new-cell','update-prepared-new-cell','new-cell-query')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-pr-','-issue-','-report-draft-'])]
files += [w/n for n in ['proposed_unproved_queries_v1.py','larger-prefix-routing-design-v1.md','compare-larger-prefix-reservations-v1.py','compare-larger-prefix-reservations-v2.py','larger-prefix-reservation-first-attempt-v1.json','larger-prefix-reservation-comparison-v2.json']]
folders=['prepared-new-cell-pilot-v1','prepared-new-cell-pilot-v2','prepared-new-cell-simple-guards-v1','prepared-new-cell-final-v1','prepared-new-cell-final-v2','prepared-new-cell-hand-repeat-v1','prepared-new-cell-hand-trace-v1','prepared-new-cell-hand-import-audit-v1']
for name in folders:files += [p for p in (w/name).rglob('*') if p.is_file()]
files += [Path(__file__).resolve()];files=sorted(set(files))
meta=dict(scope=__doc__,runtime=runtime,baseline=baseline,device='NVIDIA L40',warp='1.19.0',files=[],source_sha256=source_audit['source_sha256'],shared_helpers=[],shared_prior_evidence=[],limitations=['GPU timing improvements do not meet the wide comparable-quality native CPU goal.','The initial production graded candidate had one CPU quota-throttled period; its mesh/timings are retained separately and a fresh complete pair was required. Three preceding clean pairs are retained in the primary cohort.', 'The private strict/exact v2 experiment adds no useful benefit over simple v1 and is not adopted. Its full evidence is retained. The simple grid harness system-Python import failure and explicit-venv retry are retained.','Retained-grid stress timing excludes initial full construction and is not fresh VTM timing.','Native trace/observer/sanitizer times are diagnostic, not clean performance samples.','Fresh solves exclude setup, compilation, upload and serialization; workspace pool memory is not process peak memory.','The first captured-loop launch includes deferred graph instantiation/upload before its single graph launch; no claim of zero driver-internal host work.','Known Armadillo extraction failures and comparable-quality CPU limitations remain open.'],reproduction=['Restore original .work/vtm paths, shared helpers and prior inputs in checkouts of pinned source commits. Replace the absolute workspace root as needed; drivers refuse overwrites.','Private prototypes use baselinea4f7215. Production has no private installer. Source/kernel AST checks bind the promoted kernels to simple v1. Additional simple8k/grid candidates use retained fresh v2 controls, not new interleaved pairs.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU process at a time; no heavy CPU during clean timing.','Build native validators from benchmarks/cpu; binary hashes retained. /opt holds only regenerable caches/scratch.'])
meta['experiment_mapping']={
 'simple':dict(source='prepared_new_cell_v1.py',sha256=sha(w/'prepared_new_cell_v1.py'),integration='prepared_new_cell_integration_v1.py',integration_sha256=sha(w/'prepared_new_cell_integration_v1.py'),pilot='prepared-new-cell-pilot-v1',guards='prepared-new-cell-simple-guards-v1',adopted_kernel_layout=True),
 'split':dict(source='prepared_new_cell_v2.py',sha256=sha(w/'prepared_new_cell_v2.py'),integration='prepared_new_cell_integration_v2.py',integration_sha256=sha(w/'prepared_new_cell_integration_v2.py'),pilot='prepared-new-cell-pilot-v2',grid='prepared-new-cell-grid-v2',adopted=False),
 'production':dict(source='warpower/_unproved_queries.py',runtime=runtime,final='prepared-new-cell-final-v2',prior_stopped_batch='prepared-new-cell-final-v1',retained_clean_jobs=6)}

for p in files:
 assert p.is_file() and not p.is_symlink(),p
 target=out/p.relative_to(w);assert not target.exists(),target
 meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','boundary-launch/commands/boundary-timing-commands.json')]:
 p=w/original;q=root/'results/validation/vtm'/canonical;assert sha(p)==sha(q);meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
for folder,needed in [('cached-exact-seed-queries',('hierarchical-split-pilot-v2/commands.json','hierarchical-cache-final-v1/commands.json')),('hierarchical-seed-queries',('large-stackless-pilot-v1/commands.json','large-stackless-final-v1/commands.json')),('extended-fragment-reuse',('extended-reuse-final-v1/commands.json','extended-reuse-collection-v1.log'))]:
 prior=root/'results/validation/vtm'/folder/'manifest.json';old=json.loads(prior.read_text());meta['shared_prior_evidence'].append(dict(path=rel(prior),sha256=sha(prior)))
 for row in old['files']:
  if any(row['original'].endswith(x) for x in needed):
   assert sha(root/row['path'])==row['sha256'];meta['shared_prior_evidence'].append(row)
meta['input_fixtures']=[dict(path='tests/data/hand-fit256.npz',sha256=sha(root/'tests/data/hand-fit256.npz'))]
meta['native_binary_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
for name in folders:
 p=w/name/'lossless-archives.json'
 if not p.exists():continue
 for row in json.loads(p.read_text()):
  packed=Path(row['path']);assert sha(packed)==row['sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==row['raw_sha256']
for row in read('prepared-new-cell-final-v1-throttled-archives.json'):
 packed=Path(row['path']);assert sha(packed)==row['sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==row['raw_sha256']
for label in ['prepared-new-cell-profile-v1','prepared-new-cell-profile-v2','prepared-new-cell-full-native-v1','prepared-new-cell-loop-native-v1']:
 stem=w/label;r=read(label+'-kernels.json');packed=stem.with_suffix('.nsys-rep.gz')
 assert sha(packed)==r['trace_gz_sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==r['trace_raw_sha256']
 dot=stem.with_suffix('.dot.gz');audit=read(label+'-audit.json');assert hashlib.sha256(gzip.decompress(dot.read_bytes())).hexdigest()==audit['dot_sha256']
for phase in ['before','after']:
 stem=w/'prepared-new-cell-hand-trace-v1'/phase;r=read('prepared-new-cell-hand-trace-v1/'+phase+'-archives.json')
 assert sha(stem.with_suffix('.nsys-rep.gz'))==r['gzip_sha256'];assert hashlib.sha256(gzip.decompress(stem.with_suffix('.nsys-rep.gz').read_bytes())).hexdigest()==r['raw_sha256']
 assert sha(stem.with_suffix('.dot.gz'))==r['dot_gzip_sha256'];assert hashlib.sha256(gzip.decompress(stem.with_suffix('.dot.gz').read_bytes())).hexdigest()==r['dot_sha256']
for row in meta['files']:
 source=root/row['original'];target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
 assert sha(source)==row['sha256'] and not target.exists()
 os.link(source,target);target.chmod(0o444);assert target.stat().st_ino==source.stat().st_ino
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
for r in meta['files']:assert sha(root/r['path'])==sha(root/r['original'])==r['sha256']
result=dict(files=len(files),manifest_sha256=sha(p),all_artifact_source_archive_hashes_verified=True,runtime=runtime)
(w/'verified-prepared-new-cell-publication-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
