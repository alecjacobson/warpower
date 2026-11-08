"""Publish split/cached seed-query experiments, including the rejected regression."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/cached-exact-seed-queries'
assert not (out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
def read(n):return json.loads((w/n).read_text())
runtime=read('hierarchical-cache-full-suite-v1.json')['source'];baseline='7c3dfc74acb4e947636e428bc0ea866f66dfdf77'
assert subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])==b''
assert '1237 passed' in (w/'hierarchical-cache-full-suite-v1.log').read_text()
checks=read('hierarchical-cache-final-v1-independent.json');assert checks['complete'] and len(checks['cases'])==7 and len(checks['standalone'])==5
assert all(r['comparison']['points_bitwise_equal'] for r in checks['cases'].values())
assert read('hierarchical-cache-grid-v1-independent.json')['complete']
mem=read('hierarchical-cache-memory-v1-commands.json');assert len(mem)==28 and all(r['exit_code']==0 and r['zero_memcheck_errors'] and r['expected_tests']==1 for r in mem)
qualification=read('hierarchical-cache-qualification-v1.json');assert len(qualification)==3 and all(r['exit_code']==0 for r in qualification)
assert read('hierarchical-cache-native-loop-independent-v1.json')['complete']
for label in ['full','loop']:
 audit=read(f'hierarchical-cache-{label}-native-v1-audit.json')
 assert audit['gpu_kernel_activities']>0 and set(audit['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'}
 assert set(audit['dot_copy_kinds'])<={'DtoD (DEVICE to DEVICE)'}
 assert all(not any(word in name.lower() for word in ['alloc','free','memcpy']) for name in audit['cuda_api_calls'])
for version in [1,2,3]:assert read(f'hierarchical-split-pilot-v{version}-independent.json')['complete']
for version in [1,2]:
 assert read(f'hierarchical-split-grid-v{version}-independent.json')['complete']
 assert read(f'hierarchical-split-grid-observer-v{version}-comparison.json')['passed']
 observed=read(f'hierarchical-split-observer-v{version}-comparison.json')
 if version==1:
  assert all(observed['work_equal'].values())
  assert all(observed['geometry'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
  assert observed['geometry']['maximum_coordinate_difference']==0
 else:assert observed['passed']
source_audit=read('hierarchical-cache-production-source-audit-v1.json')
for name,digest in source_audit['source_sha256'].items():assert sha(root/name)==digest,name
prefixes=('hierarchical-split','hierarchical_split','hierarchical-cache','run-hierarchical','check-hierarchical','profile-hierarchical','test_hierarchical','test-hierarchical','audit-hierarchical','prepare-hierarchical','promote-hierarchical','write-hierarchical','bench-hierarchical')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-pr-','-issue-','-doc-draft-'])]
folders=['hierarchical-split-pilot-v1','hierarchical-split-pilot-v2','hierarchical-split-pilot-v3','hierarchical-cache-final-v1']
for name in folders:files += [p for p in (w/name).rglob('*') if p.is_file()]
files += [Path(__file__).resolve()];files=sorted(set(files))
meta=dict(scope=__doc__,runtime=runtime,baseline=baseline,device='NVIDIA L40',warp='1.19.0',files=[],source_sha256=source_audit['source_sha256'],shared_helpers=[],shared_prior_evidence=[],limitations=['GPU timing improvements do not meet the wide comparable-quality native CPU goal.','The split-only private v1 grid regression is retained. Only the bounded-cache v2 layout was promoted.','Retained-grid stress timing excludes initial full construction and is not fresh VTM timing.','Native trace/observer/sanitizer times are diagnostic, not clean performance samples.','Fresh solves exclude setup, compilation, upload and serialization; workspace pool memory is not process peak memory.','The first captured-loop launch includes deferred graph instantiation/upload before its single graph launch; no claim of zero driver-internal host work.','Known Armadillo extraction failures and comparable-quality CPU limitations remain open.'],reproduction=['Restore original .work/vtm paths, shared helpers and prior inputs in checkouts of pinned source commits. Replace the absolute workspace root as needed; drivers refuse overwrites.','Private prototypes use baseline7c3dfc7. Production has no private installer. Source/kernel AST checks bind the promoted implementation to the cached prototype.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU process at a time; no heavy CPU during clean timing.','Build native validators from benchmarks/cpu; binary hashes retained. /opt holds only regenerable caches/scratch.'])
meta['experiment_mapping']={
 'split_only':dict(source='hierarchical_split_v1.py',sha256=sha(w/'hierarchical_split_v1.py'),pilots=['hierarchical-split-pilot-v1','hierarchical-split-pilot-v2'],grid='hierarchical-split-grid-v1',adopted=False),
 'cached':dict(source='hierarchical_split_v2.py',sha256=sha(w/'hierarchical_split_v2.py'),pilots=['hierarchical-split-pilot-v3'],grid='hierarchical-split-grid-v2',adopted_kernel_layout=True),
 'production':dict(source='warpower/_stackless_seeds.py',runtime=runtime,final='hierarchical-cache-final-v1')}
for p in files:
 assert p.is_file() and not p.is_symlink(),p
 target=out/p.relative_to(w);assert not target.exists(),target
 meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','boundary-launch/commands/boundary-timing-commands.json')]:
 p=w/original;q=root/'results/validation/vtm'/canonical;assert sha(p)==sha(q);meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
for folder,needed in [('hierarchical-seed-queries',('large-stackless-pilot-v1/commands.json','large-stackless-final-v1/commands.json')),('extended-fragment-reuse',('extended-reuse-final-v1/commands.json','extended-reuse-collection-v1.log'))]:
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
for label in ['hierarchical-split-profile-v1','hierarchical-cache-full-native-v1','hierarchical-cache-loop-native-v1']:
 stem=w/label;r=read(label+'-kernels.json');packed=stem.with_suffix('.nsys-rep.gz')
 assert sha(packed)==r['trace_gz_sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==r['trace_raw_sha256']
 dot=stem.with_suffix('.dot.gz');audit=read(label+'-audit.json');assert hashlib.sha256(gzip.decompress(dot.read_bytes())).hexdigest()==audit['dot_sha256']
for row in meta['files']:
 source=root/row['original'];target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
 assert sha(source)==row['sha256'] and not target.exists()
 os.link(source,target);target.chmod(0o444);assert target.stat().st_ino==source.stat().st_ino
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
for r in meta['files']:assert sha(root/r['path'])==sha(root/r['original'])==r['sha256']
result=dict(files=len(files),manifest_sha256=sha(p),all_artifact_source_archive_hashes_verified=True,runtime=runtime)
(w/'verified-hierarchical-cache-publication-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
