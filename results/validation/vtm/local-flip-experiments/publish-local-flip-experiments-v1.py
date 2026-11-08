"""Publish unadopted local-flip experiments, including observed failures."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/local-flip-experiments';assert not out.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
baseline='74ae277572d0a6e07cfc1442760f45b9178eee87'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==baseline
assert not subprocess.check_output(['git','diff','--name-only','--','warpower','tests','benchmarks','examples','README.md'],text=True).strip()
for version in [1,2]:
 d=json.loads((w/f'local-flips-v{version}/results.json').read_text());assert d['complete'] and len(d['cases'])==35 and sum(r['accepted'] for r in d['cases'].values())==12
 independent=json.loads((w/f'local-flips-v{version}/independent.json').read_text());assert independent['complete'] and len(independent['cases'])==12 and all(r['exit_code']==0 for r in independent['cases'].values())
 for name,row in d['cases'].items():assert sha(w/f'local-flips-v{version}'/(name+'.npz'))==row['output_sha256']
 p=w/f'local-flip-pilot-v{version}/commands.json';commands=json.loads(p.read_text());assert len(commands)==4 and all(r['exit_code']==0 and r['quota_clean'] and r['valid']==1 and r['error']==0 for r in commands)
 for r in commands:assert sha(root/r['result'])==r['result_sha256']
 p=w/('local-flip-pilot-v1-independent-v2.json' if version==1 else 'local-flip-pilot-v2-independent-v1.json');d=json.loads(p.read_text());assert d['complete'] and len(d['cases'])==2
 for row in d['cases'].values():
  assert all(row['work_equal'].values());assert row['comparison']['material_tets_equal'] and row['comparison']['oriented_boundary_equal'] and row['comparison']['roles_equal']
  for phase in ['before','after']:
   c=row[phase]['independent'];assert c['full_hull']['valid'] and c['all_boundary_vertex_links_single_cycles'] and c['boundary_self_intersection_pairs']==0 and c['vtu_round_trip_exact']
checks=[]
for name,n in [('snapshot-validation.json',26),('snapshot-validation-graded.json',14)]:
 d=json.loads((w/'local-flip-observer-v2'/name).read_text());assert d['complete'] and len(d['cases'])==n
 for path,row in d['cases'].items():
  assert sha(root/path)==row['sha256'];assert row['previous']['exit_code']==row['reconstructed']['exit_code']==0
 checks.extend(d['cases'])
assert len(set(checks))==40
prefixes=('local-flip','local_flips_','local_flip_','run-local-flip','check-local-flip','build-local-flip','diagnose-local-flip','measure-local-flip','publish-local-flip')
files=[]
for p in sorted(w.iterdir()):
 if not p.name.startswith(prefixes) or any(s in p.name for s in ['archived','storage']):continue
 if p.is_file() and p.suffix!='.md':files.append(p)
 elif p.is_dir():files.extend(sorted(q for q in p.rglob('*') if q.is_file()))
assert len(files)==len(set(files))
meta=dict(scope=__doc__,baseline=baseline,status='Both private schedules regress full VTM performance and remain unadopted. No production/default API/README change. Wide native CPU speed target remains unmet.',device='NVIDIA L40',warp='1.19.0',numpy='2.2.6',files=[],shared_helpers=[],shared_inputs=[],source_sha256={},limitations=[
 'Two private35-case GPU runs and8 full VTM outputs; no new full-suite, memory instrumentation, native trace, broad-gallery or standalone timing claim.',
 'Observer timings include extra diagnostics and are not speed comparisons.40 saved snapshots each have old and reconstructed meshes independently checked.',
 'Some setup windows are CPU-quota throttled. All first and timed replay windows are clean; the original overly broad offline quota checker failure is retained.',
 'v1 hand full-hull exterior connectivity varies after near-zero coordinate changes. Both outputs independently validate; material mesh, boundary, roles and work agree. No causal kernel claim.',
 'Prototype dispatch does not separately enforce the unchanged-owner checkpoint measured by the observer; current-input coverage and final certificate are mandatory. No prototype was adopted.',
 'Strict five-site flips can stall; three exact coplanar four-cell replacements are retained for issue212. Full production reconstruction succeeds.',
 'Finite-source quality distances are sampled; this is not a Hausdorff guarantee or CPU meshing speed comparison.'
],reproduction=[
 'Use baseline74ae277 in a separate persistent checkout. Restore modules/helpers to their recorded original .work/vtm paths, adjusting workspace roots. Output directories must be new; the harnesses refuse overwriting.',
 'Build independent CGAL validators from benchmarks/cpu. The shared Armadillo input is already retained in results/examples/vtm/armadillo-uniform/input.off.',
 'Run one GPU process at a time. Do not run heavy CPU validators during clean timing. Env: PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm.',
 'Observe using run-local-flip-observer-v2.py; run private trials with run-local-flips-v1.py/v2.py and full comparisons with run-local-flip-pilot-v1.py/v2.py. The v1 observer failed compilation and is retained.',
 'Correct complete-output validators are check-local-flip-pilot-v2.py forpilotv1 and check-local-flip-pilot-v3.py forpilotv2; their predecessor stopped on setup throttling.',
 'Export files may be lossless gzip, verified by local-flip-observer-exports-compression-v1.json and the pilot drivers. Reproduction can leave fresh exports uncompressed or use the retained checks.',
 'Publisher stdout must be outside collected prefixes so it cannot hash its own still-open log.'
])
for p in files:
 q=out/p.relative_to(w);q.parent.mkdir(parents=True,exist_ok=True);os.link(p,q);q.chmod(0o444);assert sha(p)==sha(q)
 meta['files'].append(dict(original=rel(p),path=rel(q),bytes=q.stat().st_size,sha256=sha(q)))
for name,public in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py')]:
 p=w/name;q=root/'results/validation/vtm'/public;assert sha(p)==sha(q);meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
p=w/'scanned-armadillo-input/armadillo.off';q=root/'results/examples/vtm/armadillo-uniform/input.off';assert sha(p)==sha(q);meta['shared_inputs'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
for folder in ['warpower','benchmarks','examples','tests']:
 for p in sorted((root/folder).rglob('*.py')):meta['source_sha256'][rel(p)]=sha(p)
meta['reference_binaries_sha256']={n:sha(w/n) for n in ['check-delaunay','surface-distance','surface-reference']}
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in meta['files']),source_files=len(meta['source_sha256']),manifest=rel(p),manifest_sha256=sha(p))))
