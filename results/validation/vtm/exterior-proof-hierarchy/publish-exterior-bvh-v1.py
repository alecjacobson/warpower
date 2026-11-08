"""Publish verified proof acceleration evidence without adopting private topology recovery."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/exterior-proof-hierarchy';assert not out.exists()
runtime='431958fc3efa127f99f9c298cc66be6834232783'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==runtime
assert not subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((w/n).read_text())
assert '1317 passed' in (w/'exterior-bvh-full-suite-v1.log').read_text()
pipeline=read('exterior-bvh-pipeline-v2.json');assert len(pipeline)==2 and all(r['exit_code']==0 for r in pipeline)
final=read('exterior-bvh-final-v2/commands.json');assert len(final)==28 and all(r['exit_code']==0 and r['quota_clean'] for r in final)
r=read('exterior-bvh-final-v3-independent.json');assert r['complete'] and len(r['cases'])==9 and len(r['standalone'])==5
assert read('exterior-bvh-native-loop-independent-v1.json')['complete']
m=read('exterior-bvh-memory-v1-commands.json');assert len(m)==8 and all(x['exit_code']==0 and x['zero_memcheck_errors'] for x in m)
for stem in ['tangential-topology-uniform-native-v2','exterior-bvh-uniform-native-v1','exterior-bvh-loop-native-v1']:
 assert read(stem+'-command.json')['exit_code']==0
 a=read(stem+'-audit.json');assert set(a['dot_node_types'])<={'KERNEL','MEMCPY','MEMSET','CONDITIONAL'}
 assert set(a['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert a['cuda_api_calls']['cudaGraphLaunch_v10000']==a['cuda_api_calls']['cuCtxSynchronize']==1
 assert not any(any(term in k.lower() for term in ['alloc','free','memcpy']) for k in a['cuda_api_calls'])
 k=read(stem+'-kernels.json');p=w/(stem+'.nsys-rep.gz')
 assert sha(p)==k['trace_gz_sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==k['trace_raw_sha256']
 assert hashlib.sha256(gzip.decompress((w/(stem+'.dot.gz')).read_bytes())).hexdigest()==a['dot_sha256']
for a in read('exterior-bvh-final-v2/lossless-archives.json'):
 p=Path(a['path']);assert sha(p)==a['sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==a['raw_sha256']
prefixes=('exterior-bvh','exterior-plane-bvh','exterior-ray-bvh','exterior-ray-bounds','exterior_plane_bvh','exterior_ray_bvh',
          'run-exterior-bvh','profile-exterior-bvh','check-exterior-bvh','build-exterior-bvh','build-exterior-ray-bvh','compare-exterior-',
          'test_exterior_bvh','test_exterior_plane_bvh','test_exterior_ray_bvh','test_exterior_ray_bounds',
          'tangential-plane-bvh','tangential-ray-bvh','run-tangential-plane-bvh','run-tangential-ray-bvh',
          'tangential-topology-uniform-native-v2','profile-tangential-topology-uniform-v2','run-tangential-topology-uniform-native-v2')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(k in p.name for k in ['-pr-','-merge-audit-','-comment-'])]
for folder in ['exterior-bvh-final-v1','exterior-bvh-final-v2','tangential-topology-pilot-v2','tangential-plane-bvh-pilot-v1','tangential-ray-bvh-pilot-v1']:
 files += [p for p in (w/folder).rglob('*') if p.is_file()]
files += [w/n for n in ['tangential_topology_recovery_v2.py','run-tangential-topology-gallery-v2.py','run-tangential-topology-pilot-v2.py','check-tangential-topology-pilot-v2.py','tangential-topology-pilot-v2-independent.json','write-exterior-bvh-report-v1.py','large-reuse-isolated-test.py']]
files.append(Path(__file__).resolve())
meta=dict(scope=__doc__,runtime=runtime,baseline='6f878c7098b525e4cfdbda5e8d2ed45cf8b7e736',files=[],shared_references=[],runtime_source_sha256={},limitations=['Tangential topology recovery remains private and is used identically on both sides of the experimental8K timing pairs.','Native traces and initial pilot replays are diagnostic, not the clean timing samples.','Disjoint coplanar triangle planes can stop causing a conservative ray ambiguity; geometric proof remains conservative.','Wide comparable-quality CPU speedup and issue228 remain open; README flagship unchanged.'])
for name in subprocess.check_output(['git','ls-files','warpower','tests','benchmarks','examples','README.md'],text=True).splitlines():
 p=root/name
 if p.is_file():meta['runtime_source_sha256'][name]=sha(p)
for p in [root/'results/validation/vtm/surface-bound-reuse/check-surface-bound-reuse-final-v2.py',root/'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py',w/'check-delaunay',w/'surface-distance',w/'surface-reference',w/'uniform-armadillo-8192-transaction-v1/failed-candidate.npz']:
 meta['shared_references'].append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
# The real Armadillo sanitizer fixture must be available in the public archive.
p=w/'uniform-armadillo-8192-transaction-v1/failed-candidate.npz';files.append(p)
files.append(w/'scanned-armadillo-input/armadillo.off')
files.append(w/'scanned-armadillo-input/provenance.json')
files.append(w/'update-exterior-bvh-docs-v1.py')
for p in sorted(set(files)):
 q=out/p.relative_to(w);q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists()
 row=dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),sha256=sha(p),bytes=p.stat().st_size)
 os.link(p,q);q.chmod(0o444);assert sha(q)==row['sha256'];meta['files'].append(row)
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(dict(files=len(meta['files']),manifest_sha256=sha(p),source_files=len(meta['runtime_source_sha256'])))
