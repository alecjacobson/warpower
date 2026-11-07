"""Publish closed validation records, retaining original failed harnesses."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/recovery-diagnostics'
runtime='987568a054a0';baseline='6f1601e2f960fb75dde63e9421bf518c9dca20f9'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip().startswith(runtime)
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','README.md'],text=True)
assert '82 passed' in (w/'recovery-diagnostics-production-focused-v2.log').read_text()
assert '894 passed' in (w/'recovery-diagnostics-full-suite.log').read_text()
jobs=json.loads((w/'recovery-diagnostics-final/commands.json').read_text());assert len(jobs)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in jobs)
final=json.loads((w/'recovery-diagnostics-final-independent.json').read_text());assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(r['relative_change']<.02 for r in final['standalone'].values())
for r in final['cases'].values():
 assert all(r['work_equal'].values()) and r['independent']['full_hull']['valid']
 assert r['independent']['all_boundary_vertex_links_single_cycles']
for name,count in [('memory-v2',4),('native',3)]:
 rows=json.loads((w/('recovery-diagnostics-'+name+'-commands.json')).read_text())
 assert len(rows)==count and all(r['exit_code']==0 for r in rows)
 if name=='memory-v2':assert all(r['zero_memcheck_errors'] for r in rows)
for name in ['full','loop']:
 d=json.loads((w/f'recovery-diagnostics-{name}-native-audit.json').read_text())
 assert set(d['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert set(d['gpu_memcpy_kinds'])<={'CUDA_MEMCPY_KIND_DTOD'}
 assert set(d['dot_node_types'])<={'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
 assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
 assert d['cuda_api_calls'].get('cuCtxSynchronize')==1
quality=json.loads((w/'recovery-diagnostics-public-quality.json').read_text())
assert len(quality['cases'])==8 and all(r['all_fields_match'] and r['public_rms']>0 for r in quality['cases'].values())
assert 'PENDING' not in (root/'docs/selection-surface-status.md').read_text()
summary=json.loads((w/'recovery-diagnostics-native-audit-summary.json').read_text())
kernels=summary['full']['kernels']
for prefix,calls in [('classify_',110),('surface_error_',39),('selection_surface_status_',71)]:
 assert sum(r['calls'] for r in kernels if r['kernel'].startswith(prefix))==calls
loop=json.loads((w/'recovery-diagnostics-loop-native.json').read_text())
assert loop['device_comparison_error']==0 and loop['phase']==8
for name in ['full','loop']:
 audit=json.loads((w/f'recovery-diagnostics-{name}-native-audit.json').read_text())
 assert set(audit['cuda_api_calls']) <= {'cuCtxSynchronize','cuProfilerStart','cudaGraphInstantiateWithFlags_v11040','cudaGraphLaunch_v10000','cudaGraphUpload_v10000','cudaStreamIsCapturing_v10000'}
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
prefixes=('recovery-diagnostics','recovery_diagnostics','run-recovery-diagnostics','check-recovery-diagnostics','analyze-recovery-diagnostics','test_recovery_diagnostics','write-recovery-diagnostics','publish-recovery-diagnostics')
for p in sorted(w.iterdir()):
 if not p.name.startswith(prefixes) and p.name!='test_selection_extraction_initial.py':continue
 if any(x in p.name for x in ['storage','archived-mesh','mesh-fixture-restore','progress192','next-classification-issue','publish.log']):continue
 if p.is_file() and (p.suffix!='.md' or p.name=='recovery-diagnostics-witness-v1-proof.md'):copy(p)
 elif p.is_dir():
  for child in sorted(p.rglob('*')):
   if not child.is_file():continue
   if p.name!='recovery-diagnostics-final' and child.name.endswith(('.npz','.vtu.gz','.off.gz','.off')):
    private_meshes.append(dict(path=str(child.relative_to(root)),sha256=sha(child),bytes=child.stat().st_size));continue
   copy(child)
helpers=[]
for private,public in [('check-current-selection-final.py','current-selection/check-current-selection-final.py'),('check-shared-wall-final.py','shared-wall/check-shared-wall-final.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json')]:
 p=root/'results/validation/vtm'/public;assert p.read_bytes()==(w/private).read_bytes();helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
sources=['warpower/_selective.py','warpower/_extraction.py','tests/test_selection_extraction.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
prior=root/'results/validation/vtm/current-selection/current-selection-armadillo-profile-summary.json'
helpers.append(dict(path=str(prior.relative_to(root)),sha256=sha(prior)))
manifest=dict(baseline=baseline,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scope='Private selection surface-query status preserves full-query errors while omitting unused distance reductions; public full diagnostics stay available. Full production meshes published; earlier instrumented/pilot meshes retained privately with hashes. No quality-matched CPU speedup claim.',source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared,shared_helpers=helpers,reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},private_pilot_meshes=private_meshes,files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print(len(files),sum(r['bytes'] for r in files),'public bytes',len(private_meshes),'private pilot meshes',flush=True)
