"""Archive verified integration results without modifying prior evidence."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/shared-wall'
runtime='cc0a020f5f7c586370191d0246eb116c92cd0ac8';baseline='6eb1882d8af126e65ee88fe0cbdd5f3520f8cc87'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==runtime
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','README.md'],text=True)
assert '75 passed' in (w/'shared-wall-integrated-focused.log').read_text()
assert '844 passed' in (w/'shared-wall-full-suite.log').read_text()
assert len(json.loads((w/'shared-wall-private-independent-v3.json').read_text()))==10
jobs=json.loads((w/'shared-wall-final/commands.json').read_text());assert len(jobs)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in jobs)
final=json.loads((w/'shared-wall-final-independent.json').read_text());assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(abs(r['relative_change'])<.02 for r in final['standalone'].values())
for r in final['cases'].values():
 assert all(r['work_equal'].values()) and r['independent']['full_hull']['valid']
 assert r['independent']['all_boundary_vertex_links_single_cycles'] and r['after_ms']<r['before_ms']
for name,count in [('memory-v2',6),('native',3)]:
 rows=json.loads((w/('shared-wall-'+name+'-commands.json')).read_text())
 assert len(rows)==count and all(r['exit_code']==0 for r in rows)
 if name=='memory-v2':assert all(r['zero_memcheck_errors'] for r in rows)
interrupted=json.loads((w/'shared-wall-memory-commands.json').read_text())
assert len(interrupted)==1 and interrupted[0]['exit_code']==-2
assert '9 passed' in (w/'shared-wall-wall-padded.log').read_text()
for name in ['full','loop']:
 d=json.loads((w/f'shared-wall-{name}-native-audit.json').read_text())
 assert set(d['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert set(d['gpu_memcpy_kinds'])<={'CUDA_MEMCPY_KIND_DTOD'}
 assert set(d['dot_node_types'])<={'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
 assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
 assert d['cuda_api_calls'].get('cuCtxSynchronize')==1
assert 'PENDING' not in (root/'docs/shared-wall-reconstruction.md').read_text()
assert not out.exists();out.mkdir(parents=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
canonical=root/'results/examples/vtm/armadillo-uniform/input.off';canonical_hash=sha(canonical)
files=[];shared=[]
def copy(p):
 h=sha(p)
 if p.name=='input.off' and h==canonical_hash:
  shared.append(dict(original=str(p.relative_to(root)),canonical=str(canonical.relative_to(root)),sha256=h));return
 dest=out/p.relative_to(w);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();os.link(p,dest)
 files.append(dict(original=str(p.relative_to(root)),path=str(dest.relative_to(root)),bytes=p.stat().st_size,sha256=h))
prefixes=('shared-wall','run-shared-wall','check-shared-wall','analyze-shared-wall','write-shared-wall','publish-shared-wall',
          'local-dewall-vtm-v3','local-dewall-v3','local-dewall-shared-v1-focused')
explicit=['local_dewall_integration_v3.py','local_dewall_shared_v1.py','run-local-dewall-vtm-pilots-v3.py','run-local-dewall-mesher-v3.py','test_local_dewall_v3.py','test_local_dewall_shared_v1.py']
for p in sorted(w.iterdir()):
 if not p.name.startswith(prefixes) and p.name not in explicit:continue
 if any(x in p.name for x in ['git-storage','pack-verify','packed-object','cache-cleanup','identical-storage','progress188','parent29','open-issues']):continue
 if p.is_file() and p.suffix!='.md':copy(p)
 elif p.is_dir():
  for child in sorted(p.rglob('*')):
   if child.is_file():copy(child)
helpers=[]
for private,public in [('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json'),('local_dewall_v4.py','local-dewall/local_dewall_v4.py'),('local_dewall_integration_v2.py','local-dewall/local_dewall_integration_v2.py'),('test_local_dewall_v2.py','local-dewall/test_local_dewall_v2.py')]:
 p=root/'results/validation/vtm'/public;assert p.read_bytes()==(w/private).read_bytes();helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
sources=['warpower/_wall_candidates.py','warpower/_wall_reuse.py','warpower/_active_delaunay.py','warpower/_vtm.py','tests/test_wall_reuse.py','NOTICE','LICENSE-Local-DeWall','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
manifest=dict(baseline=baseline,runtime=runtime,scope='Private VTM wall fallback shares exact filtering/completion and publishes current proof plus topology metadata once; standalone public defaults unchanged. No quality-matched CPU speedup claim.',source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared,shared_helpers=helpers,reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print(len(files),sum(r['bytes'] for r in files),flush=True)
