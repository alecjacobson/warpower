"""Verify published storage evidence and runtime bindings before merge."""
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

root = Path.cwd()
out = root / 'results/validation/vtm/recovery-query-storage'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(n):
    return json.loads((out / n).read_text())

m = read('manifest.json')
for row in m['files']:
    p, q = root / row['path'], root / row['original']
    assert sha(p) == sha(q) == row['sha256']
    assert p.stat().st_ino == q.stat().st_ino and not p.stat().st_mode & 0o222
for row in m['shared_references']:
    assert sha(root / row['path']) == row['sha256']
for n, h in m['runtime_source_sha256'].items():
    assert sha(root / n) == h
    assert hashlib.sha256(subprocess.check_output(['git', 'show', m['runtime'] + ':' + n])).hexdigest() == h
assert not subprocess.check_output(['git', 'diff', m['runtime'], '--', 'warpower', 'tests', 'benchmarks', 'examples', 'README.md', 'pyproject.toml'])
s = read('quality-query-storage-production-source-v1.json')
assert len(s['source_sha256']) == 262 and len(s['unchanged_files']) == 259
assert s['readme_unchanged'] and s['independent_diagram_and_delaunay_sources_unchanged']
suite = read('quality-query-storage-production-full-suite-v1.json')
assert suite['runtime'] == m['runtime'] and suite['exit_code'] == 0
assert sha(out / 'quality-query-storage-production-full-suite-v1.log') == suite['log_sha256']
assert '1327 passed' in (out / 'quality-query-storage-production-full-suite-v1.log').read_text()
commands = read('quality-query-storage-final-v1/commands.json')
assert len(commands) == 20 and all(j['exit_code'] == 0 and j['quota_clean'] for j in commands)
for j in commands:
    assert j['source'] == m['baseline' if j['phase'] == 'before' else 'runtime']
    p = out / Path(j['result']).relative_to(root / '.work/vtm')
    assert sha(p) == j['result_sha256']
r = read('quality-query-storage-final-v2-independent.json')
assert r['complete'] and len(r['cases']) == len(r['standalone']) == 5
for name, c in r['cases'].items():
    assert all(c['work_equal'].values()) and all(c['input_arrays_equal'].values())
    for phase in ['before', 'after']:
        d = c['independent_' + phase]
        assert d['full_hull']['valid'] and d['boundary_self_intersection_pairs'] == 0
        p = out / 'quality-query-storage-final-v1' / (name + '-' + phase)
        p = p / 'mesh.npz' if p.is_dir() else p.with_suffix('.npz')
        assert sha(p) == d['mesh_sha256']
    d = c['comparison']
    assert all(d[k] for k in ['roles_equal', 'material_tets_equal', 'oriented_boundary_equal'])
    assert d['full_tets_equal'] == (name != 'thin-hand-seed7')
    assert d['relative_coordinate_difference'] < 1e-9
    if 'maximum_radius_ratio_difference' in d:
        assert d['maximum_radius_ratio_difference'] < 1e-7 and abs(d['minimum_radius_ratio_before'] - d['minimum_radius_ratio_after']) < 1e-12
    if 'saved_workspace_bytes' in c:
        assert c['saved_workspace_bytes'] == c['predicted_payload_bytes']
for c in r['standalone'].values():
    assert all(c['unchanged'].values())
hand = read('quality-query-storage-hand-variation-v2.json')
assert hand['complete'] and hand['baseline_variation_matches_production_difference']
assert hand['all_changed_full_cells_are_outside_material'] and hand['production_full_hull_matches_other_unchanged_baseline']
assert len(hand['removed_full_cells']) == len(hand['added_full_cells']) == 4
assert all(type(v) is int for k in ['removed_full_cells', 'added_full_cells'] for row in hand[k] for v in row)
assert read('quality-query-storage-independent-v1.json')['exit_code'] == 1
assert read('quality-query-storage-independent-v2.json')['exit_code'] == 0
assert not read('quality-query-storage-final-v1-independent.json').get('complete', False)
for n in ['quality-query-storage-production-loop-v1', 'quality-query-storage-loop-native-v1', 'quality-query-storage-memory-v1-loop-padded', 'quality-query-storage-memory-v1-loop-lifetime']:
    c = read(n + '-comparison.json')
    assert c['complete'] and c['rows_equal'] and c['stable_pointers'] and len(c['frames']) == 3
    for f in c['frames'].values():
        assert all(v['equal'] for v in f['fields'].values())
assert read('quality-query-storage-native-full-v1-comparison.json')['complete']
jobs = read('quality-query-storage-memory-v1-commands.json')
assert len(jobs) == 4 and all(j['exit_code'] == 0 and j['zero_memcheck_errors'] for j in jobs)
for j in jobs:
    log = (out / ('quality-query-storage-memory-v1-' + j['name'] + '-' + j['mode'] + '.log')).read_text()
    assert 'ERROR SUMMARY: 0 errors' in log and '1 passed' in log
for stem, count in [('quality-query-storage-uniform-native-v1', 90971), ('quality-query-storage-loop-native-v1', 156956)]:
    a = read(stem + '-audit.json')
    calls = a['cuda_api_calls']
    assert a['gpu_kernel_activities'] == count
    assert calls['cudaGraphLaunch_v10000'] == 1 and sum(v for k, v in calls.items() if 'Synchronize' in k) == 1
    assert not any(any(t in k.lower() for t in ['alloc', 'free', 'memcpy']) for k in calls)
    assert set(a['dot_node_types']) <= {'KERNEL', 'MEMCPY', 'MEMSET', 'CONDITIONAL'} and set(a['dot_copy_kinds']) == {'DtoD (DEVICE to DEVICE)'}
    k = read(stem + '-kernels.json')
    p = out / (stem + '.nsys-rep.gz')
    assert sha(p) == k['trace_gz_sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest() == k['trace_raw_sha256']
    assert hashlib.sha256(gzip.decompress((out / (stem + '.dot.gz')).read_bytes())).hexdigest() == a['dot_sha256']
links = []
for name in ['recovery-query-storage.md', 'tangential-topology-recovery.md', 'architecture.md', 'mesher.md', 'vtm.md']:
    p = root / 'docs' / name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text()):
        if target.startswith(('http://', 'https://', '#')):
            continue
        q = (p.parent / target.split('#', 1)[0]).resolve()
        rel = str(q.relative_to(root))
        if not q.exists():
            subprocess.run(['git', 'cat-file', '-e', 'HEAD:' + rel], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        links.append(dict(document=name, target=rel))
result = dict(scope=__doc__, manifest_sha256=sha(out/'manifest.json'), artifact_hashes_aliases_readonly_verified=len(m['files']), runtime_files_verified=262, full_tests=1327, final_validated_meshes=10, memory_jobs=4, hand_exterior_variation_explicit=True, standalone_unchanged=True, readme_unchanged=True, links=links, verifier_sha256=sha(Path(__file__)))
p = out / 'premerge-integrity.json'
assert not p.exists()
p.write_text(json.dumps(result, indent=2) + '\n')
p.chmod(0o444)
print({k: v for k, v in result.items() if k != 'links'})
