"""Publish constructor-only recovery storage qualification without altering raw evidence."""
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = root / 'results/validation/vtm/recovery-query-storage'
assert not out.exists()

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(n):
    return json.loads((w / n).read_text())

s = read('quality-query-storage-production-source-v1.json')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == s['runtime']
assert len(s['source_sha256']) == 262 and len(s['unchanged_files']) == 259
assert s['readme_unchanged'] and s['independent_diagram_and_delaunay_sources_unchanged']
for n, h in s['source_sha256'].items():
    assert sha(root / n) == h
assert '1327 passed' in (w / 'quality-query-storage-production-full-suite-v1.log').read_text()
assert read('quality-query-storage-production-full-suite-v1.json')['exit_code'] == 0
jobs = read('quality-query-storage-final-v1/commands.json')
assert len(jobs) == 20 and all(j['exit_code'] == 0 and j['quota_clean'] for j in jobs)
r = read('quality-query-storage-final-v2-independent.json')
assert r['complete'] and len(r['cases']) == len(r['standalone']) == 5
for name, c in r['cases'].items():
    assert c['before_valid'] == c['after_valid'] == 1
    assert all(c['work_equal'].values()) and all(c['input_arrays_equal'].values())
    for phase in ['before', 'after']:
        d = c['independent_' + phase]
        assert d['full_hull']['valid'] and d['boundary_self_intersection_pairs'] == 0
    d = c['comparison']
    assert all(d[k] for k in ['roles_equal', 'material_tets_equal', 'oriented_boundary_equal'])
    assert d['full_tets_equal'] == (name != 'thin-hand-seed7')
    if name == 'thin-hand-seed7':
        assert d['changed_full_cells_are_exterior_only'] and d['full_hull_matches_other_certified_unchanged_baseline']
    if 'saved_workspace_bytes' in c:
        assert c['saved_workspace_bytes'] == c['predicted_payload_bytes']
for c in r['standalone'].values():
    assert all(c['unchanged'].values())
for n in ['quality-query-storage-pilot-v1-independent.json', 'quality-query-storage-hand-variation-v2.json', 'quality-query-storage-native-full-v1-comparison.json', 'quality-query-storage-production-loop-v1-comparison.json', 'quality-query-storage-loop-native-v1-comparison.json', 'quality-query-storage-memory-v1-loop-padded-comparison.json', 'quality-query-storage-memory-v1-loop-lifetime-comparison.json']:
    assert read(n)['complete']
for n, length in [('quality-query-storage-pipeline-v1.json', 2), ('quality-query-storage-qualification-v1.json', 3)]:
    jobs = read(n)
    assert len(jobs) == length and all(j['exit_code'] == 0 for j in jobs)
jobs = read('quality-query-storage-memory-v1-commands.json')
assert len(jobs) == 4 and all(j['exit_code'] == 0 and j['zero_memcheck_errors'] for j in jobs)
assert {(j['name'], j['mode']) for j in jobs} == {(n, m) for n in ['storage', 'loop'] for m in ['padded', 'lifetime']}
for stem in ['quality-query-storage-uniform-native-v1', 'quality-query-storage-loop-native-v1']:
    a = read(stem + '-audit.json')
    calls = a['cuda_api_calls']
    assert set(a['dot_node_types']) <= {'KERNEL', 'MEMCPY', 'MEMSET', 'CONDITIONAL'}
    assert set(a['dot_copy_kinds']) == {'DtoD (DEVICE to DEVICE)'}
    assert calls['cudaGraphLaunch_v10000'] == 1 and sum(v for k, v in calls.items() if 'Synchronize' in k) == 1
    assert not any(any(t in k.lower() for t in ['alloc', 'free', 'memcpy']) for k in calls)
    k = read(stem + '-kernels.json')
    p = w / (stem + '.nsys-rep.gz')
    assert sha(p) == k['trace_gz_sha256']
    assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() == k['trace_raw_sha256']
    assert hashlib.sha256(gzip.decompress((w / (stem + '.dot.gz')).read_bytes())).hexdigest() == a['dot_sha256']
for folder in ['quality-query-storage-pilot-v1', 'quality-query-storage-final-v1']:
    for a in read(folder + '/lossless-archives.json'):
        p = Path(a['path'])
        assert sha(p) == a['sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() == a['raw_sha256']

prefixes = ('quality-query-storage-', 'run-quality-query-storage-', 'profile-quality-query-storage-', 'check-quality-query-storage-', 'audit-quality-query-storage-')
files = [p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-pr-', '-merge-', '-publication-', '-issue'])]
folders = ['quality-query-storage-pilot-v1', 'quality-query-storage-final-v1', 'quality-query-storage-production-loop-v1', 'quality-query-storage-loop-native-v1-frames', 'quality-query-storage-memory-v1-loop-padded-frames', 'quality-query-storage-memory-v1-loop-lifetime-frames']
for folder in folders:
    files += [p for p in (w / folder).rglob('*') if p.is_file()]
files += [w / 'quality_query_storage_v1.py', Path(__file__).resolve()]
meta = dict(scope=__doc__, runtime=s['runtime'], baseline=s['baseline'], runtime_source_sha256=s['source_sha256'], files=[], shared_references=[], native_validators={n: sha(w/n) for n in ['check-delaunay', 'surface-distance', 'surface-reference']}, limitations=['Storage savings only; no replay speedup or comparable-quality CPU speedup claimed.', 'Hand full-hull variation is explicitly retained and demonstrated on unchanged baseline runs; material cells and oriented boundary agree.', 'Failed strict v1 comparison and incorrectly serialized v1 diagnostic remain retained, with v2 corrections and proof bindings.', 'Native and memory diagnostic times are not clean performance samples.', 'Finite recovery and previous combined-process instrumentation limitations remain.'], reproduction=['Use pinned baseline/runtime with Warp 1.19.0 on NVIDIA L40; one GPU process and no heavy CPU validators during clean timings.', 'Restore originals to their .work/vtm paths from this manifest and referenced prior publications; adjust absolute workspace paths as needed. Drivers refuse overwrites.', 'Use venv Python, PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. Compile offline validators from benchmarks/cpu.', 'Original Stanford/libigl scan provenance and terms are retained; no asset relicensing.'])
prior = root / 'results/validation/vtm/tangential-topology-recovery'
refs = [prior / n for n in ['manifest.json', 'run-tangential-production-gallery-v1.py', 'large-reuse-isolated-test.py', 'test_tangential_production_loop_v2.py', 'check-tangential-production-memory-v3.py', 'tangential-production-loop-v2-independent.json', 'tangential-production-source-audit-v1.json', 'tangential-production-memory-v3-commands.json', 'tangential-production-final-v1/commands.json', 'tangential-production-loop-v2/result.json', 'tangential-production-loop-v2/frame0.npz', 'tangential-production-loop-v2/frame1.npz', 'tangential-production-loop-v2/frame4.npz']]
refs += [root / 'results/validation/vtm' / n for n in ['surface-bound-reuse/check-surface-bound-reuse-final-v2.py', 'large-fragment-reuse/check-large-reuse-proof-pairs.py', 'larger-active-prefixes/larger-prefix-final-v1/commands.json', 'exterior-proof-hierarchy/scanned-armadillo-input/armadillo.off', 'exterior-proof-hierarchy/scanned-armadillo-input/provenance.json']]
for p in refs:
    meta['shared_references'].append(dict(path=str(p.relative_to(root)), sha256=sha(p)))
for p in sorted(set(files)):
    q = out / p.relative_to(w)
    q.parent.mkdir(parents=True, exist_ok=True)
    row = dict(original=str(p.relative_to(root)), path=str(q.relative_to(root)), sha256=sha(p), bytes=p.stat().st_size)
    assert not q.exists()
    os.link(p, q)
    q.chmod(0o444)
    assert sha(q) == row['sha256']
    meta['files'].append(row)
p = out / 'manifest.json'
p.write_text(json.dumps(meta, indent=2) + '\n')
p.chmod(0o444)
print(dict(files=len(meta['files']), manifest_sha256=sha(p)))
