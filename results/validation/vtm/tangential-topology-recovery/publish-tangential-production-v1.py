"""Publish qualified device topology-recovery evidence with immutable source bindings."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/tangential-topology-recovery';assert not out.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((w/n).read_text())
audit=read('tangential-production-source-audit-v1.json');runtime=audit['runtime']
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==runtime
assert not subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md','pyproject.toml'])
assert '1326 passed' in (w/'tangential-production-full-suite-v1.log').read_text()
commands=read('tangential-production-final-v1/commands.json');assert len(commands)==28 and all(d['quota_clean'] and d['exit_code'] in [0,1] for d in commands)
assert sum(d['valid']==0 for d in commands)==2
r=read('tangential-production-final-v1-independent.json');assert r['complete'] and len(r['cases'])==9 and len(r['standalone'])==5
for n in ['tangential-production-loop-v1-independent.json','tangential-production-loop-v2-independent.json','tangential-production-loop-native-v1-independent.json']:assert read(n)['complete']
assert read('tangential-production-native-full-v1-independent.json')['complete']
assert read('tangential-production-memory-v3-comparison.json')['complete']
assert read('tangential-production-reference-v1.json')['complete']
for name in ['tangential-production-loop-v2/result.json','tangential-production-loop-native-v1-frames/result.json']:
 for path,digest in read(name)['production_source_sha256'].items():assert audit['source_sha256'][path]==digest
q=read('tangential-production-qualification-v3.json');assert len(q)==3 and all(d['exit_code']==0 for d in q)
m=read('tangential-production-memory-v3-commands.json');assert len(m)==10 and all(d['exit_code']==0 and d['zero_memcheck_errors'] for d in m)
for stem in ['tangential-production-uniform-native-v1','tangential-production-loop-native-v1']:
 assert read(stem+'-command.json')['exit_code']==0
 a=read(stem+'-audit.json');calls=a['cuda_api_calls']
 assert set(a['dot_node_types'])<={'KERNEL','MEMCPY','MEMSET','CONDITIONAL'} and set(a['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert calls['cudaGraphLaunch_v10000']==1
 assert sum(v for k,v in calls.items() if 'Synchronize' in k)==1
 assert not any(any(term in k.lower() for term in ['alloc','free','memcpy']) for k in calls)
 k=read(stem+'-kernels.json');p=w/(stem+'.nsys-rep.gz')
 assert sha(p)==k['trace_gz_sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==k['trace_raw_sha256']
 assert hashlib.sha256(gzip.decompress((w/(stem+'.dot.gz')).read_bytes())).hexdigest()==a['dot_sha256']
for a in read('tangential-production-final-v1/lossless-archives.json'):
 p=Path(a['path']);assert sha(p)==a['sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==a['raw_sha256']
prefixes=('tangential-production-','run-tangential-production-','profile-tangential-production-','check-tangential-production-','test_tangential_production_','audit-tangential-production-','build-tangential-','render-tangential-production-','write-tangential-production-report-','update-tangential-production-docs-')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(k in p.name for k in ['-pr-','-merge-','-publication-'])]
folders=['tangential-production-final-v1','tangential-production-loop-v1','tangential-production-loop-v2','tangential-production-loop-native-v1-frames','tangential-production-memory-v3-loop-padded-frames','tangential-production-memory-v3-loop-lifetime-frames']
for n in folders:files += [p for p in (w/n).rglob('*') if p.is_file()]
files += [w/n for n in ['large-reuse-isolated-test.py','test_tangential_policy_isolated_v1.py','build-tangential-policy-isolation-v1.py','build-tangential-production-v1.py','tangential-topology-recovery-design-v2.md','test_tangential_rollback_loop_v1.py','check-tangential-rollback-loop-v1.py','tangential-rollback-loop-v1.log','tangential-rollback-loop-v1-independent.log','tangential-rollback-loop-v1-independent.json']]
files += [p for p in (w/'tangential-rollback-loop-v1').rglob('*') if p.is_file()]
files.append(Path(__file__).resolve())
meta=dict(scope=__doc__,runtime=runtime,baseline=audit['baseline'],files=[],shared_references=[],runtime_source_sha256=audit['source_sha256'],limitations=['Finite32proposal quality recovery is not guaranteed to fix all inputs.','Global fitting energy-decrease guards are unchanged; quality moves may slightly increase fitting within existing bounds.','The two8K baseline failures have no successful-mesh speedup ratio.','Extra persistent recovery storage and setup are disclosed; standalone geometry sources/allocations unchanged.','Native and sanitizer times are diagnostic; setup/compilation/readback excluded from clean replay timings.','Wide comparable-quality native CPU speedup remains unmet; README flagship unchanged.','Initial production v1 loop predates complete diagnostic reset; current-source v2 loop and fullsuite are separate.'],reproduction=['Checkout pinned runtime/baseline and restore artifacts to original .work/vtm paths, retaining input/reference hashes. Adjust absolute workspace root in drivers as needed; drivers refuse overwrites.','Use the repository venv Python with Warp1.19.0; PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. One GPU process, no CPU validation during clean timing.','Compile offline native validators from benchmarks/cpu; their executable hashes record the measured tools, not portable binaries.','Restore referenced frozen helper sources and original scan from prior published proof-hierarchy evidence. Asset provenance/terms remain separate from code license.'])
prior=root/'results/validation/vtm/exterior-proof-hierarchy'
references=[root/'results/validation/vtm/surface-bound-reuse/check-surface-bound-reuse-final-v2.py',root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py',prior/'tangential_topology_recovery_v2.py',prior/'uniform-armadillo-8192-transaction-v1/failed-candidate.npz',prior/'scanned-armadillo-input/armadillo.off',prior/'scanned-armadillo-input/provenance.json']
for case in ['armadillo-graded8192-recovery','armadillo-uniform8192-recovery']:
 references += [prior/'exterior-bvh-final-v2'/(case+'-after')/n for n in ['mesh.npz','result.json']]
for p in references:meta['shared_references'].append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
meta['native_validators']={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']}
for p in sorted(set(files)):
 q=out/p.relative_to(w);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
 row=dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
 os.link(p,q);q.chmod(0o444);assert sha(q)==row['sha256'];meta['files'].append(row)
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(dict(files=len(meta['files']),manifest_sha256=sha(p),source_files=len(meta['runtime_source_sha256'])))
