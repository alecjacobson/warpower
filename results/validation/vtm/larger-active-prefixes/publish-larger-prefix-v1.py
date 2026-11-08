"""Publish qualified larger-prefix routing with exact sources, costs and rejection guards."""
import gzip,hashlib,json,os,subprocess,sys
import numpy as np
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/larger-active-prefixes';assert not (out/'manifest.json').exists()
def read(n):return json.loads((w/n).read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
runtime='670bd3b1a801e5397f55837d3caf685c20286656';baseline='df74493c97633324147cb39ef4c41a7d34c7b90a'
assert read('larger-prefix-full-suite-v1.json')['source']==runtime and read('larger-prefix-full-suite-v1.json')['tests']==1292
assert '47 passed' in (w/'larger-prefix-focused-v1.log').read_text()
assert subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])==b''
source=read('larger-prefix-production-source-audit-v2.json');assert source['source']==runtime and len(source['source_sha256'])==148
for name,digest in source['source_sha256'].items():assert sha(root/name)==digest
commands=read('larger-prefix-final-v1/commands.json');assert len(commands)==34 and all(r['exit_code'] in [0,1] and r['quota_clean'] for r in commands)
report=read('larger-prefix-final-v1-independent.json');assert report['complete'] and len(report['cases'])==11 and len(report['standalone'])==5
successful=0
for name,r in report['cases'].items():
 assert all(r['work_equal'].values())
 if r['before_valid']==r['after_valid']==1:
  successful+=1;assert all(r['comparison'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
 else:
  assert 'armadillo' in name and r['before_valid']==r['after_valid']==0 and all(r['failed_state_equal'].values())
assert successful==9
assert read('larger-prefix-grid-v1-independent.json')['complete']
assert read('larger-prefix-pilot-v1-independent.json')['complete']
assert read('larger-prefix-current-reservation-comparison-v1.json')['complete']
assert read('larger-prefix-tests-v1.json')['exit_code']==read('larger-prefix-edge-tests-v1.json')['exit_code']==0
mem=read('larger-prefix-memory-v1-commands.json');assert len(mem)==28 and all(r['exit_code']==0 and r['zero_memcheck_errors'] and r['expected_tests']==1 for r in mem)
qualification=read('larger-prefix-qualification-v1.json');assert len(qualification)==3 and all(r['exit_code']==0 for r in qualification)
assert read('larger-prefix-native-loop-independent-v1.json')['complete']
for kind in ['full','loop']:
 a=read(f'larger-prefix-{kind}-native-v1-audit.json');assert a['gpu_kernel_activities']>0 and set(a['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'}
 assert set(a['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert all(not any(word in k.lower() for word in ['alloc','free','memcpy']) for k in a['cuda_api_calls'])
 prefix=f'larger-prefix-{kind}-native-v1';r=read(prefix+'-kernels.json');trace=w/(prefix+'.nsys-rep.gz');dot=w/(prefix+'.dot.gz')
 assert sha(trace)==r['trace_gz_sha256'] and hashlib.sha256(gzip.decompress(trace.read_bytes())).hexdigest()==r['trace_raw_sha256']
 assert hashlib.sha256(gzip.decompress(dot.read_bytes())).hexdigest()==a['dot_sha256']
files=[]
prefixes=('larger-prefix','larger_prefix','run-larger-prefix','check-larger-prefix','profile-larger-prefix','test_larger_prefix','build-larger-prefix','update-larger-prefix','audit-larger-prefix','promote-larger-prefix','write-larger-prefix','bench-larger-prefix','compare-larger-prefix')
for p in w.iterdir():
 if p.is_file() and p.name.startswith(prefixes) and not any(s in p.name for s in ['-comment-','-issue-','-pr-','-premerge-','-merge-audit-']):files.append(p)
for folder in ['larger-prefix-pilot-v1','larger-prefix-final-v1']:
 files += [p for p in (w/folder).rglob('*') if p.is_file()]
 p=w/folder/'lossless-archives.json'
 if p.exists():
  for row in json.loads(p.read_text()):
   packed=Path(row['path']);assert sha(packed)==row['sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==row['raw_sha256']
files.append(Path(__file__).resolve());files=sorted(set(files))
meta=dict(scope=__doc__,runtime=runtime,baseline=baseline,device='NVIDIA L40',warp='1.19.0',python_runtime=sys.version,numpy=np.__version__,files=[],source_sha256=source['source_sha256'],shared_helpers=[],shared_prior_evidence=[],input_fixtures=[],native_binary_sha256={},limitations=['Wide comparable-quality native CPU speedup remains unmet; these timings compare GPU revisions.','Additional children add 1,853,169,562 persistent workspace bytes at 16K and 894,484,877 at 8K in measured configurations, plus setup; not scratch sharing or process peak memory.','Known uniform/graded 8K Armadillo error 64 remains; rejection timings are not successful-mesh speedups.','Native trace/sanitizer times are diagnostic; cold loop launch can instantiate/upload before its single graph launch.','Retained-grid timing excludes initial construction.','Original-coordinate proof does not imply bitwise floating VTM trajectories for every input; strict equality is established for saved measured pairs.'],reproduction=['Check out the pinned runtime/baseline; restore original .work/vtm artifact paths and referenced helpers/prior inputs. Replace the absolute workspace root as needed. Drivers refuse overwrites.','Private pilot uses larger_prefix_v1.py before unchanged benchmark setup; production uses the committed VTM-owned policy. Source/AST audit binds their constructor bodies.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. (append .work/vtm for private prototype). One GPU process; no heavy CPU validation during clean timings.','Build native validators from retained benchmarks/cpu source; executable hashes and input hashes retained. /opt is regenerable scratch only.'])
for p in files:
 assert p.is_file() and not p.is_symlink();q=out/p.relative_to(w);assert not q.exists()
 meta['files'].append(dict(original=rel(p),path=rel(q),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py')]:
 p=w/original;q=root/'results/validation/vtm'/canonical;assert sha(p)==sha(q);meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
for folder,needed in [('prepared-new-cell-queries',('prepared-new-cell-final-v2/commands.json','larger-prefix-routing-design-v1.md','larger-prefix-reservation-comparison-v2.json')),('extended-fragment-reuse',('extended-reuse-final-v1/commands.json','extended-reuse-gallery-dense-v1/commands.json'))]:
 p=root/'results/validation/vtm'/folder/'manifest.json';prior=json.loads(p.read_text());meta['shared_prior_evidence'].append(dict(path=rel(p),sha256=sha(p)))
 for row in prior['files']:
  if any(row['original'].endswith(n) for n in needed):
   assert sha(root/row['path'])==row['sha256'];meta['shared_prior_evidence'].append(row)
for name in ['tests/data/hand-fit256.npz','.work/vtm/scanned-armadillo-input/armadillo.off']:
 p=root/name;meta['input_fixtures'].append(dict(path=name,sha256=sha(p)))
for name in ['check-delaunay','surface-distance','surface-reference']:meta['native_binary_sha256'][name]=sha(w/name)
for row in meta['files']:
 p=root/row['original'];q=root/row['path'];assert sha(p)==row['sha256'];q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q);q.chmod(0o444);assert p.stat().st_ino==q.stat().st_ino
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
for row in meta['files']:assert sha(root/row['original'])==sha(root/row['path'])==row['sha256']
r=dict(files=len(files),manifest_sha256=sha(p),runtime=runtime,all_sources_archives_aliases_verified=True)
(w/'verified-larger-prefix-publication-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
