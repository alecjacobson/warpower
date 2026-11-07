"""Publish completed bounded-filter evidence; verify every retained artifact."""
import gzip,hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/bounded-predicate-filters'
revisions=json.loads((w/'float32-runtime.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==revisions['runtime']
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks','examples','README.md'],text=True)
assert '795 passed' in (w/'float32-full-suite.log').read_text()
assert '150 passed' in (w/'float32-focused-tests.log').read_text()
for name,n in [('float32-filter-bound-tests-v2.log',8),('float32-face-existing.log',69),('float32-face-query-tests.log',36),('float32-seed-existing.log',69),('float32-seed-mask-tests.log',36),('float32-combined-existing.log',69)]:assert f'{n} passed' in (w/name).read_text()
final=json.loads((w/'float32-final-independent.json').read_text())
assert len(final['cases'])==8 and len(final['standalone'])==5
assert all(abs(r['relative_change'])<.02 for r in final['standalone'].values())
for row in final['cases'].values():
 assert all(row['work_equal'].values()) and row['independent']['full_hull']['valid']
 if 'before_pool_bytes' in row:assert row['before_pool_bytes']==row['after_pool_bytes']
 assert row['after_ms']<row['before_ms']
assert sum('before_pool_bytes' in row for row in final['cases'].values())==2
for consumer in ['face','seed','combined']:
 rows=json.loads((w/f'float32-{consumer}-pilot/commands.json').read_text());assert len(rows)==4 and all(r['exit_code']==0 and r['quota_clean'] for r in rows)
 assert len(json.loads((w/f'float32-{consumer}-pilot-independent.json').read_text()))==2
assert len(json.loads((w/'float32-broad-independent.json').read_text())['cases'])==6
counter=json.loads((w/'float32-counter-equivalence.json').read_text());assert all(counter['comparison'].values()) and all(counter['work_equal'].values())
assert counter['totals']['face']['orientation_f64_uncertain']==counter['totals']['face']['sphere_f64_uncertain']==counter['totals']['seed']['prepared_ball64_uncertain']==0
memory=json.loads((w/'float32-memory-commands.json').read_text());assert len(memory)==8 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
native=json.loads((w/'float32-native-commands.json').read_text());assert len(native)==3 and all(r['exit_code']==0 for r in native)
for name in ['full','loop']:
 d=json.loads((w/f'float32-{name}-native-audit.json').read_text())
 assert set(d['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert set(d['gpu_memcpy_kinds']) <= {'CUDA_MEMCPY_KIND_DTOD'}
 assert set(d['dot_node_types']) <= {'MEMCPY','KERNEL','MEMSET','CONDITIONAL'}
 assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1 and d['cuda_api_calls'].get('cuCtxSynchronize')==1
assert 'PENDING' not in (root/'docs/bounded-predicate-filters.md').read_text()
assert not out.exists();out.mkdir(parents=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=root/'results/examples/vtm/armadillo-uniform/input.off';canonical_hash=sha(canonical)
files=[];shared_inputs=[];removed=[]
def copy(p,relative=None):
 original_hash=sha(p)
 if p.name=='input.off' and original_hash==canonical_hash:
  shared_inputs.append(dict(original=str(p.relative_to(root)),canonical=str(canonical.relative_to(root)),sha256=original_hash));return
 dest=out/(relative or p.name);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists(),dest
 record=dict(original=str(p.relative_to(root)),original_sha256=original_hash)
 compressed=p.suffix in ['.off','.vtu','.nsys-rep']
 if compressed:
  dest=dest.with_name(dest.name+'.gz');data=p.read_bytes();dest.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(dest.read_bytes())==data
  record['encoding']='gzip; decompress to recover exact original bytes'
 else:os.link(p,dest)
 record.update(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=sha(dest));files.append(record)
 if compressed:
  removed.append(dict(raw=str(p.relative_to(root)),archive=str(dest.relative_to(root)),sha256=original_hash));p.unlink()
exclude=('cache','cleanup','duplicate','bytecode','preflight','relocation','parent','issue','initial-progress','publication','pr-','push','staged-bound')
prefixes=('float32-','float32_','run-float32','check-float32','test-float32','test_float32','build_float32','check_float32','install_float32','integrate_float32','stage_float32','write_float32','publish_float32','prove_float32','bounded_float32','flushed_float32')
for p in sorted(w.iterdir()):
 if p.is_file() and p.name.startswith(prefixes) and not any(term in p.name for term in exclude):
  if p.suffix in ['.py','.json','.log','.npz','.gz','.nsys-rep','.md']:copy(p)
for name in ['reference-predicates-1996.c','reference-predicates-1996-source.json']:copy(w/name)
for folder in ['float32-face-pilot','float32-seed-pilot','float32-combined-pilot','float32-broad','float32-final','float32-staged']:
 for p in sorted((w/folder).rglob('*')):
  if p.is_file():copy(p,p.relative_to(w))
helpers=[]
for private,public in [('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json'),('prepared-query-final/commands.json','prepared-point-queries/prepared-query-final/commands.json')]:
 p=root/'results/validation/vtm'/public;assert p.read_bytes()==(w/private).read_bytes();helpers.append(dict(path=str(p.relative_to(root)),sha256=sha(p)))
sources=['warpower/_bounded_predicates.py','warpower/_analytic_predicates.py','warpower/_proved_completion.py','warpower/_grouped_seeds.py','benchmarks/bounded_predicate_proof.py','tests/test_bounded_predicates.py','tests/_flushed_bounded_predicates.py','tests/test_bounded_predicate_proof.py','tests/test_analytic_face_search.py','tests/test_grouped_query_masks.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/graph_capture.py']
manifest=dict(**revisions,scope='Bounded binary32 strict first stages in private VTM face and retained-cell queries. Original-coordinate exact sign acceptance with complete former fallback; unchanged storage/launch counts and standalone defaults. Includes separate private consumer pilots and all completed validation. No CPU outperformance claim; existing routed full allocation-order instrumentation limitation remains.',source_sha256={p:sha(root/p) for p in sources},shared_inputs=shared_inputs,shared_helpers=helpers,reference_binaries_sha256={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']},verified_raw_duplicates_removed=removed,files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:assert sha(root/r['path'])==r['sha256']
print('Verified',len(files),'artifacts;',sum(r['bytes'] for r in files),'bytes',flush=True)
