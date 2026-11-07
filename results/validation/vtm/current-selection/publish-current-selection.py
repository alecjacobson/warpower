"""Publish closed validation records, retaining original failed harnesses."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/current-selection'
runtime='0b67386';baseline='80d6ded503ac799048c3a82d17003951aa0f8aa6'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip().startswith(runtime)
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','README.md'],text=True)
assert '46 passed' in (w/'current-selection-production-focused-v2.log').read_text()
assert '851 passed' in (w/'current-selection-full-suite.log').read_text()
jobs=json.loads((w/'current-selection-final/commands.json').read_text());assert len(jobs)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in jobs)
final=json.loads((w/'current-selection-final-independent.json').read_text());assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(r['relative_change']<.02 for r in final['standalone'].values())
for r in final['cases'].values():
 assert all(r['work_equal'].values()) and r['independent']['full_hull']['valid']
 assert r['independent']['all_boundary_vertex_links_single_cycles']
for name,count in [('memory-v2',4),('native',3)]:
 rows=json.loads((w/('current-selection-'+name+'-commands.json')).read_text())
 assert len(rows)==count and all(r['exit_code']==0 for r in rows)
 if name=='memory-v2':assert all(r['zero_memcheck_errors'] for r in rows)
for name in ['full','loop']:
 d=json.loads((w/f'current-selection-{name}-native-audit.json').read_text())
 assert set(d['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert set(d['gpu_memcpy_kinds'])<={'CUDA_MEMCPY_KIND_DTOD'}
 assert set(d['dot_node_types'])<={'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
 assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
 assert d['cuda_api_calls'].get('cuCtxSynchronize')==1
profiles=json.loads((w/'current-selection-armadillo-profile-summary.json').read_text())
assert set(profiles)=={'before','after'} and all(r['exit_code']==0 for r in profiles.values())
assert profiles['before']['result']==profiles['after']['result']
assert 'PENDING' not in (root/'docs/current-material-selection.md').read_text()
assert not out.exists();out.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=root/'results/examples/vtm/armadillo-uniform/input.off';canonical_hash=sha(canonical)
files=[];shared=[];private_meshes=[]
def copy(p):
 h=sha(p)
 if p.name=='input.off' and h==canonical_hash:
  shared.append(dict(original=str(p.relative_to(root)),canonical=str(canonical.relative_to(root)),sha256=h));return
 dest=out/p.relative_to(w);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();os.link(p,dest)
 files.append(dict(original=str(p.relative_to(root)),path=str(dest.relative_to(root)),bytes=p.stat().st_size,sha256=h))
prefixes=('current-selection','current_selection','run-current-selection','check-current-selection','analyze-current-selection','test_current_selection','write-current-selection','publish-current-selection')
for p in sorted(w.iterdir()):
 if not p.name.startswith(prefixes):continue
 if any(x in p.name for x in ['storage','archived-trace','bytecode','progress190','publish.log']):continue
 if p.is_file() and p.suffix!='.md':copy(p)
 elif p.is_dir():
  for child in sorted(p.rglob('*')):
   if not child.is_file():continue
   if p.name!='current-selection-final' and child.name.endswith(('.npz','.vtu.gz','.off.gz','.off')):
    private_meshes.append(dict(path=str(child.relative_to(root)),sha256=sha(child),bytes=child.stat().st_size));continue
   copy(child)
helpers=[]
for private,public in [('check-shared-wall-final.py','shared-wall/check-shared-wall-final.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json')]:
 p=root/'results/validation/vtm'/public;assert p.read_bytes()==(w/private).read_bytes();helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
sources=['warpower/_selective.py','warpower/_vtm.py','tests/test_current_selection.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
manifest=dict(baseline=baseline,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scope='Accepted material evaluation reused only inside uninterrupted VTM step schedule. Full production meshes published; earlier instrumented/pilot meshes retained privately with hashes. No quality-matched CPU speedup claim.',source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared,shared_helpers=helpers,reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},private_pilot_meshes=private_meshes,files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print(len(files),sum(r['bytes'] for r in files),'public bytes',len(private_meshes),'private pilot meshes',flush=True)
