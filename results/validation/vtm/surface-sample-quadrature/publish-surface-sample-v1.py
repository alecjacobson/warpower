"""Publish qualified parallel quadrature with immutable raw experiments and shared proof bindings."""
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/surface-sample-quadrature';assert not out.exists()
def read(name):return json.loads((w/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
source=read('surface-sample-production-source-v2.json')
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source['runtime']
for p,h in source['source_sha256'].items():assert sha(root/p)==h
tests=read('surface-sample-production-tests-v2.json');assert len(tests)==2 and all(r['passed'] and r['exit_code']==0 for r in tests)
assert '1387 passed' in (w/'surface-sample-production-full-suite-v2.log').read_text()
commands=read('surface-sample-final-v1/commands.json');assert len(commands)==32 and all(r['quota_clean'] and r['exit_code']==0 for r in commands)
proof=read('surface-sample-final-v1-independent.json');assert proof['complete'] and len(proof['cases'])==11 and len(proof['standalone'])==5
memory=read('surface-sample-memory-v1-commands.json');assert len(memory)==4 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
qualified=read('surface-sample-qualified-v1.json');assert len(qualified)==5 and all(r['exit_code']==0 for r in qualified)
for name in ['surface-sample-uniform-native-v1-audit.json','surface-sample-loop-native-v1-audit.json']:
    audit=read(name);assert audit['gpu_kernel_activities']>0
    assert not any(k in audit['dot_node_types'] for k in ['HOST','MEM_ALLOC','MEM_FREE'])
    assert all('DEVICE to DEVICE' in k for k in audit['dot_copy_kinds'])
for name,row in proof['cases'].items():
    assert all(row['work_equal'].values()),(name,row['work_equal'])
    c=row['comparison'];assert all(c[k] for k in ['roles_equal','material_tets_equal','oriented_boundary_equal','material_oriented_tets_equal']) and all(c['input_arrays_equal'].values()),(name,c)
    assert c['maximum_coordinate_difference']<1e-9,(name,c)
    for phase in ['before','after']:
        v=row['independent'][phase];assert v['full_hull']['valid'] and v['all_boundary_vertex_links_single_cycles'] and v['boundary_self_intersection_pairs']==0
    qa,qb=[row['independent'][phase]['quality']['radius_ratio_percentiles']['0'] for phase in ['before','after']]
    assert abs(qa-qb)<1e-12,(name,qa,qb)
for name,row in proof['standalone'].items():assert all(row['unchanged'].values()) and row['relative_change']<.01,(name,row)
out.mkdir(parents=True)
prefixes=('surface-sample-','surface_sample_','build-surface-sample-','profile-surface-sample-','run-surface-sample-','check-surface-sample-','test_surface_sample_','test-surface-sample-','audit-surface-sample-','publish-surface-sample-','verify-surface-sample-')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(s in p.name for s in ['-issue','-pr-','-premerge','-merge-audit','-postmerge'])]
folders=['surface-sample-static-v1','surface-sample-pilot-v1','surface-sample-pilot-v2','surface-sample-final-v1','surface-sample-production-loop-v2','surface-sample-loop-native-v1-frames','surface-sample-memory-v1-loop-padded-frames','surface-sample-memory-v1-loop-lifetime-frames']
for folder in folders:files.extend(p for p in (w/folder).rglob('*') if p.is_file())
assert (w/'verify-surface-sample-v1.py').exists()
shared=[w/n for n in ['test_tangential_production_loop_v2.py','large-reuse-isolated-test.py','run-tangential-production-gallery-v1.py','check-large-reuse-proof-pairs.py','check-surface-bound-reuse-final-v2.py','check-surface-query-order-final-v1.py','check-tangential-production-memory-v3.py','surface-query-order-final-v1-independent.json','surface-query-order-final-v1/commands.json','tangential-production-loop-v2-independent.json','scanned-armadillo-input/armadillo.off']]
shared.extend(w/'tangential-production-loop-v2'/n for n in ['result.json','frame0.npz','frame1.npz','frame4.npz'])
for name in proof['cases']:
    for phase in ['before','after']:
        stem=w/'surface-query-order-final-v1'/(name+'-'+phase)
        shared.append(stem/'mesh.npz' if stem.is_dir() else stem.with_suffix('.npz'))
shared.extend(root/p for p in ['results/validation/vtm/ordered-surface-queries/manifest.json','results/validation/vtm/tangential-topology-recovery/manifest.json'])
meta=dict(scope=__doc__,baseline=source['baseline'],runtime=source['runtime'],runtime_source_sha256=source['source_sha256'],files=[],shared_references=[],limits=['The GPU revision improvement does not establish comparable-quality CPU outperformance.','Atomic integral accumulation order changes; no bitwise global reduction guarantee.','Setup and transfer are excluded from warmed replay; setup and first calls are recorded separately.','Independent sampled surface distances are not Hausdorff certificates.','Exact previous geometry proofs transfer only through complete array/oriented-connectivity checks; current exports receive their own round trips.'],reproduction=['Use the pinned baseline/runtime sources, NVIDIA L40 and Warp 1.19.0+cu12. Preserve original .work/vtm paths or adapt copies of the retained scripts.','Use .venv/bin/python with PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. One GPU job; no heavy CPU work during clean timing.','Run private contracts/static/pilot qualification, then current production tests v2, final timing, native/memory qualification and the paired/frame checks. Do not overwrite historical or failed attempts.','Restore shared inputs/helpers/proof records from the linked prior evidence; native validators are documented in benchmarks/cpu/README.md.','The scanned input remains under the original Stanford/libigl provenance terms, not the repository MIT license.'])
for path in sorted(set(files)):
    rel=path.relative_to(w);target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
    assert not target.exists();os.link(path,target);path.chmod(0o444)
    meta['files'].append(dict(original=str(path.relative_to(root)),path=str(target.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size))
for path in sorted(set(shared)):meta['shared_references'].append(dict(path=str(path.relative_to(root)),sha256=sha(path)))
manifest=out/'manifest.json';manifest.write_text(json.dumps(meta,indent=2)+'\n');manifest.chmod(0o444)
print(dict(artifacts=len(meta['files']),shared_references=len(meta['shared_references']),manifest_sha256=sha(manifest)))
