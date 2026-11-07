"""Publish closed validation records, retaining original failed harnesses."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/recovery-classification'
runtime='57c42cb5d196';baseline='f338eb5c5d90c58806d7b794769c309729567392'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip().startswith(runtime)
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','README.md'],text=True)
assert '157 passed' in (w/'recovery-classification-production-focused-v2.log').read_text()
assert '981 passed' in (w/'recovery-classification-full-suite.log').read_text()
jobs=json.loads((w/'recovery-classification-final/commands.json').read_text());assert len(jobs)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in jobs)
final=json.loads((w/'recovery-classification-final-independent.json').read_text());assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(r['relative_change']<.02 for r in final['standalone'].values())
for r in final['cases'].values():
 assert all(r['work_equal'].values()) and r['independent']['full_hull']['valid']
 assert r['independent']['all_boundary_vertex_links_single_cycles']
for name,count in [('memory-v1',4),('native',3)]:
 rows=json.loads((w/('recovery-classification-'+name+'-commands.json')).read_text())
 assert len(rows)==count and all(r['exit_code']==0 for r in rows)
 if name=='memory-v1':assert all(r['zero_memcheck_errors'] for r in rows)
for name in ['full','loop']:
 d=json.loads((w/f'recovery-classification-{name}-native-audit.json').read_text())
 assert set(d['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert set(d['gpu_memcpy_kinds'])<={'CUDA_MEMCPY_KIND_DTOD'}
 assert set(d['dot_node_types'])<={'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
 assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
 assert d['cuda_api_calls'].get('cuCtxSynchronize')==1
quality=json.loads((w/'recovery-classification-public-quality.json').read_text())
assert len(quality['cases'])==8 and all(r['all_fields_match'] and r['public_rms']>0 for r in quality['cases'].values())
assert 'PENDING' not in (root/'docs/selection-interior-status.md').read_text()
summary=json.loads((w/'recovery-classification-native-audit-summary.json').read_text())
kernels=summary['full']['kernels']
for prefix,calls in [('classify_',39),('classify_selection_',71),('surface_error_',39),('selection_surface_status_',71)]:
 matches=[r for r in kernels if r['kernel'].startswith(prefix) and (prefix!='classify_' or not r['kernel'].startswith('classify_selection_'))]
 assert sum(r['calls'] for r in matches)==calls
loop=json.loads((w/'recovery-classification-loop-native.json').read_text())
assert loop['device_comparison_error']==0 and loop['phase']==8
for name in ['full','loop']:
 audit=json.loads((w/f'recovery-classification-{name}-native-audit.json').read_text())
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
prefixes=('recovery-classification','recovery_classification','run-recovery-classification','check-recovery-classification','analyze-recovery-classification','test_recovery_classification','write-recovery-classification','publish-recovery-classification')
for p in sorted(w.iterdir()):
 if not p.name.startswith(prefixes):continue
 if any(x in p.name for x in ['storage','archived-mesh','archived-dot','progress194','publish.log']):continue
 if p.is_file() and (p.suffix!='.md' or p.name=='recovery-classification-inside-v1-proof.md'):copy(p)
 elif p.is_dir():
  for child in sorted(p.rglob('*')):
   if not child.is_file():continue
   if p.name!='recovery-classification-final' and child.name.endswith(('.npz','.vtu.gz','.off.gz','.off')):
    private_meshes.append(dict(path=str(child.relative_to(root)),sha256=sha(child),bytes=child.stat().st_size));continue
   copy(child)
helpers=[]
for private,public in [('check-recovery-diagnostics-final.py','recovery-diagnostics/check-recovery-diagnostics-final.py'),('check-current-selection-final.py','current-selection/check-current-selection-final.py'),('check-shared-wall-final.py','shared-wall/check-shared-wall-final.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json')]:
 p=root/'results/validation/vtm'/public;assert p.read_bytes()==(w/private).read_bytes();helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
sources=['warpower/_selective.py','warpower/_extraction.py','tests/test_selection_extraction.py','tests/test_selection_classification.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
prior=root/'results/validation/vtm/recovery-diagnostics/recovery-diagnostics-native-audit-summary.json'
helpers.append(dict(path=str(prior.relative_to(root)),sha256=sha(prior)))
public_source=(root/'warpower/_extraction.py').read_text()
start=public_source.index('@wp.kernel\ndef classify(');end=public_source.index('\n\n',public_source.index('    quality[i] = ',start))
public_kernel_sha256=hashlib.sha256(public_source[start:end].encode()).hexdigest()
assert public_kernel_sha256=='0275c34e9053e9780fe27c0ef17d424d5e44d3414e21b7850475caec4e241e87'
manifest=dict(public_classify_source_sha256=public_kernel_sha256,baseline=baseline,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scope='Private selection skips nearest distances only for proved inside centers; its distance buffer has only a sign contract. All outside/unsupported queries and public full distances/diagnostics retain original behavior. Full production meshes published; earlier instrumented/pilot meshes retained privately with hashes. No quality-matched CPU speedup claim.',source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared,shared_helpers=helpers,reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},private_pilot_meshes=private_meshes,files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print(len(files),sum(r['bytes'] for r in files),'public bytes',len(private_meshes),'private pilot meshes',flush=True)
