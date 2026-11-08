"""Publish qualified exact push pruning with immutable evidence and proof dependencies."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path.cwd(); w=root/'.work/vtm'; out=root/'results/validation/vtm/surface-push-pruning'; assert not out.exists()
def read(name): return json.loads((w/name).read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
source=read('surface-push-production-source-v1.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source['runtime']
for p,h in source['source_sha256'].items(): assert sha(root/p)==h
assert set(source['changed'])=={'warpower/_surface_bvh.py'}
tests=read('surface-push-production-tests-v1.json'); assert len(tests)==2 and all(r['passed'] and r['exit_code']==0 for r in tests)
assert '1387 passed' in (w/'surface-push-production-full-suite-v1.log').read_text()
commands=read('surface-push-final-v1/commands.json'); assert len(commands)==32 and all(r['quota_clean'] and r['exit_code']==0 for r in commands)
proof=read('surface-push-final-v1-independent.json'); assert proof['complete'] and len(proof['cases'])==11 and len(proof['standalone'])==5
review=read('surface-push-variation-review-v1.json'); assert review['complete'] and review['proof_sha256']==sha(w/'surface-push-final-v1-independent.json')
memory=read('surface-push-memory-v1-commands.json'); assert len(memory)==4 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
qualified=read('surface-push-qualified-v1.json'); assert len(qualified)==5 and all(r['exit_code']==0 for r in qualified)
for name in ['surface-push-uniform-native-v1-audit.json','surface-push-loop-native-v1-audit.json']:
    audit=read(name); assert audit['gpu_kernel_activities']>0
    assert not any(k in audit['dot_node_types'] for k in ['HOST','MEM_ALLOC','MEM_FREE'])
    assert all('DEVICE to DEVICE' in k for k in audit['dot_copy_kinds'])
for name,row in proof['cases'].items():
    assert all(row['work_equal'].values()),(name,row['work_equal'])
    assert row['timing']['additional_pool_bytes']==0
    c=row['comparison']; assert all(c[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal','full_oriented_tets_equal','material_oriented_tets_equal']) and all(c['input_arrays_equal'].values()),(name,c)
    assert c['maximum_coordinate_difference']<1e-9,(name,c)
    for phase in ['before','after']:
        v=row['independent'][phase]; assert v['full_hull']['valid'] and v['all_boundary_vertex_links_single_cycles'] and v['boundary_self_intersection_pairs']==0
    qa,qb=[row['independent'][phase]['quality']['radius_ratio_percentiles']['0'] for phase in ['before','after']]
    assert abs(qa-qb)<1e-12,(name,qa,qb)
for name,row in proof['standalone'].items(): assert all(row['unchanged'].values()) and row['relative_change']<.01,(name,row)
prefixes=('surface-push-','surface_push_','build-surface-push-','profile-surface-push-','run-surface-push-','check-surface-push-','test_surface_push_','test-surface-push-','audit-surface-push-','publish-surface-push-','verify-surface-push-','summarize-surface-push-','prepare-surface-push-','review-surface-push-')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(s in p.name for s in ['-issue','-progress','-pr-','-premerge','-merge-audit','-postmerge','-publication-review','-prior-evidence'])]
folders=['surface-push-pruning-static-v1','surface-push-pruning-pilot-v1','surface-push-final-v1','surface-push-production-loop-v1','surface-push-loop-native-v1-frames','surface-push-memory-v1-loop-padded-frames','surface-push-memory-v1-loop-lifetime-frames']
for folder in folders: files.extend(p for p in (w/folder).rglob('*') if p.is_file())
assert (w/'verify-surface-push-v1.py').exists()
shared=[w/n for n in ['test_tangential_production_loop_v2.py','large-reuse-isolated-test.py','run-tangential-production-gallery-v1.py','check-large-reuse-proof-pairs.py','check-surface-bound-reuse-final-v2.py','check-surface-query-order-final-v1.py','check-tangential-production-memory-v3.py','check-surface-sample-pairs-v4.py','surface-query-order-final-v1-independent.json','surface-query-order-final-v1/commands.json','surface-sample-final-v1/commands.json','tangential-production-loop-v2-independent.json','scanned-armadillo-input/armadillo.off']]
shared.extend(w/'tangential-production-loop-v2'/n for n in ['result.json','frame0.npz','frame1.npz','frame4.npz'])
for name in proof['cases']:
    for phase in ['before','after']:
        stem=w/'surface-query-order-final-v1'/(name+'-'+phase)
        shared.append(stem/'mesh.npz' if stem.is_dir() else stem.with_suffix('.npz'))
for name in ['torus1000','torus16384','armadillo-uniform8192','armadillo-graded8192']:
    shared.append(Path(read('surface-push-pruning-static-v1/'+name+'-before.json')['input']))
shared.extend(root/p for p in ['results/validation/vtm/ordered-surface-queries/manifest.json','results/validation/vtm/tangential-topology-recovery/manifest.json','results/validation/vtm/surface-sample-quadrature/manifest.json'])
shared.extend(Path(row['source']) for row in read('surface-push-qualification-preparation-v1.json')['files'])
shared.extend(w/name for name in ['surface-sample-uniform-native-v1-kernels.json','surface-sample-production-source-v2.json'])
meta=dict(scope=__doc__,baseline=source['baseline'],runtime=source['runtime'],runtime_source_sha256=source['source_sha256'],files=[],shared_references=[],limits=['GPU revision improvements do not establish comparable-quality CPU outperformance.','The exact traversal preserves surviving visits and arithmetic; floating-point atomic reductions are not promised deterministic.','Setup/upload excluded from warmed replay; setup and first calls are separate.','Sampled surface distances are not Hausdorff certificates.','Previous independent proofs transfer only through exact array/oriented-connectivity/quality comparisons; current exports receive round-trip checks.','The graded retained static regression is retained.'],reproduction=['Use pinned baseline/runtime sources, NVIDIA L40 and Warp 1.19.0+cu12. Restore original .work/vtm paths or adapt copies of retained scripts.','Use .venv/bin/python with PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU job; no heavy CPU work during clean timing.','Run private nearest/extraction contracts, static profiles, independent cache contracts, then complete pilot and its geometry checks; integrate and run production tests/final timings/native/memory/paired-frame qualification.','Restore shared inputs/helpers/proof records from prior published evidence. Native validators are documented in benchmarks/cpu/README.md.','Keep original Stanford/libigl scanned-input provenance; the input is not relicensed under MIT.'])
out.mkdir(parents=True)
for path in sorted(set(files)):
    target=out/path.relative_to(w); target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists(); os.link(path,target); path.chmod(0o444)
    meta['files'].append(dict(original=str(path.relative_to(root)),path=str(target.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size))
for path in sorted(set(shared)): meta['shared_references'].append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
manifest=out/'manifest.json'; manifest.write_text(json.dumps(meta,indent=2)+'\n'); manifest.chmod(0o444)
print(dict(artifacts=len(meta['files']),shared_references=len(meta['shared_references']),manifest_sha256=sha(manifest)))
