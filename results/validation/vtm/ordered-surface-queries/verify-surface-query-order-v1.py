"""Check the completed ordering qualification and optional published aliases."""
import argparse
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--evidence', type=Path, default=Path('.work/vtm'))
p.add_argument('--output', type=Path)
args = p.parse_args()
root = Path.cwd()
w = root / '.work/vtm'
out = args.evidence.resolve()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(name):
    return json.loads((out / name).read_text())

s = read('surface-query-order-production-source-v1.json')
assert len(s['source_sha256']) == 264 and len(s['unchanged_files']) == 259
assert s['readme_unchanged'] and s['independent_diagram_and_delaunay_sources_unchanged']
assert s['original_scalar_cache_unchanged'] and s['minimum_query_count'] == 32768
assert s['centroid_keys_and_live_gate_identical'] and s['launch_ordered_sort_methods_identical']
for n, h in s['source_sha256'].items():
    assert sha(root / n) == h
    assert hashlib.sha256(subprocess.check_output(['git', 'show', s['runtime'] + ':' + n])).hexdigest() == h
assert not subprocess.check_output(['git', 'diff', s['runtime'], '--', 'warpower', 'tests', 'benchmarks', 'examples', 'README.md', 'pyproject.toml'])
suite = read('surface-query-order-production-full-suite-v1.json')
assert suite['runtime'] == s['runtime'] and suite['exit_code'] == 0
assert sha(out / 'surface-query-order-production-full-suite-v1.log') == suite['log_sha256']
assert '1346 passed' in (out / 'surface-query-order-production-full-suite-v1.log').read_text()
commands = read('surface-query-order-final-v1/commands.json')
assert len(commands) == 32 and all(j['exit_code'] == 0 and j['quota_clean'] for j in commands)
for j in commands:
    assert j['source'] == s['baseline' if j['phase'] == 'before' else 'runtime']
    assert sha(out / Path(j['result']).relative_to(w)) == j['result_sha256']
r = read('surface-query-order-final-v1-independent.json')
assert r['complete'] and len(r['cases']) == 11 and len(r['standalone']) == 5
assert r['exact_geometry_transfers'] == 4 and not r['strict_trajectory_parity']
for name, c in r['cases'].items():
    assert c['parameters_equal'] and all(c['work_equal'].values())
    d = c['comparison']
    assert all(d[k] for k in ['roles_equal', 'full_tets_equal', 'material_tets_equal', 'oriented_boundary_equal'])
    assert all(d['input_arrays_equal'].values()) and d['relative_coordinate_difference'] < 1e-9
    assert d['points_bitwise_equal'] == (name not in ['thin-hand-seed7', 'cavity'])
    assert d.get('maximum_stored_quality_difference', 0) < 1e-7
    assert abs(c['minimum_quality']['before'] - c['minimum_quality']['after']) < 1e-12
    for phase in ['before', 'after']:
        v = c['independent'][phase]
        assert v['full_hull']['valid'] and v['boundary_self_intersection_pairs'] == 0
        assert v['all_boundary_vertex_links_single_cycles']
        stem = out / 'surface-query-order-final-v1' / (name + '-' + phase)
        mesh = stem / 'mesh.npz' if stem.is_dir() else stem.with_suffix('.npz')
        assert sha(mesh) == v['mesh_sha256']
        if stem.is_dir():
            assert v['vtu_round_trip_exact']
        if v.get('transferred_by_exact_geometry'):
            assert sha(Path(v['reference_mesh'])) == v['reference_mesh_sha256']
            assert sha(Path(v['reference_report'])) == v['reference_proof_sha256']
for c in r['standalone'].values():
    assert all(c['unchanged'].values())
assert read('surface-query-order-contracts-v1.json')['exit_code'] == 1
for version in [2, 3, 4]:
    assert read(f'surface-query-order-contracts-v{version}.json')['exit_code'] == 0
for version in [1, 2]:
    assert read(f'surface-query-order-pilot-v{version}-independent.json')['complete']
for n in ['surface-query-order-pipeline-v1.json', 'surface-query-order-qualification-v1.json']:
    rows = read(n)
    assert len(rows) == (2 if 'pipeline' in n else 3) and all(j['exit_code'] == 0 for j in rows)
for stem in ['surface-query-order-production-loop-v1', 'surface-query-order-loop-native-v1', 'surface-query-order-memory-v1-loop-padded', 'surface-query-order-memory-v1-loop-lifetime']:
    c = read(stem + '-comparison.json')
    assert c['complete'] and c['rows_equal'] and c['stable_pointers'] and len(c['frames']) == 3
    for f in c['frames'].values():
        assert all(v['equal'] for v in f['fields'].values())
assert read('surface-query-order-native-full-v1-comparison.json')['complete']
jobs = read('surface-query-order-memory-v1-commands.json')
assert len(jobs) == 4 and all(j['exit_code'] == 0 and j['zero_memcheck_errors'] for j in jobs)
assert {(j['name'], j['mode']) for j in jobs} == {(n, m) for n in ['kernels', 'loop'] for m in ['padded', 'lifetime']}
for j in jobs:
    log = (out / ('surface-query-order-memory-v1-' + j['name'] + '-' + j['mode'] + '.log')).read_text()
    assert 'ERROR SUMMARY: 0 errors' in log and str(j['expected_tests']) + ' passed' in log
for stem, count in [('surface-query-order-uniform-native-v1', 93800), ('surface-query-order-loop-native-v1', 158566)]:
    a = read(stem + '-audit.json')
    calls = a['cuda_api_calls']
    assert a['gpu_kernel_activities'] == count
    assert calls['cudaGraphLaunch_v10000'] == 1 and sum(v for k, v in calls.items() if 'Synchronize' in k) == 1
    assert not any(any(t in k.lower() for t in ['alloc', 'free', 'memcpy']) for k in calls)
    assert set(a['dot_node_types']) <= {'KERNEL', 'MEMCPY', 'MEMSET', 'CONDITIONAL'}
    assert set(a['dot_copy_kinds']) == {'DtoD (DEVICE to DEVICE)'}
    k = read(stem + '-kernels.json')
    path = out / (stem + '.nsys-rep.gz')
    assert sha(path) == k['trace_gz_sha256']
    assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == k['trace_raw_sha256']
    assert hashlib.sha256(gzip.decompress((out / (stem + '.dot.gz')).read_bytes())).hexdigest() == a['dot_sha256']
for folder in ['surface-query-order-pilot-v1', 'surface-query-order-pilot-v2', 'surface-query-order-final-v1']:
    for a in read(folder + '/lossless-archives.json'):
        path = out / Path(a['path']).relative_to(w)
        assert sha(path) == a['sha256']
        assert hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() == a['raw_sha256']
result = dict(scope=__doc__, runtime=s['runtime'], runtime_files=264, full_tests=1346, validated_meshes=22, fresh_mesh_proofs=18, exact_proof_transfers=4, memory_jobs=4, readme_unchanged=True, standalone_sources_unchanged=True, explicit_coordinate_variations=['thin-hand-seed7', 'cavity'])
if (out / 'manifest.json').exists():
    m = read('manifest.json')
    for row in m['files']:
        a, b = root / row['original'], root / row['path']
        assert sha(a) == sha(b) == row['sha256']
        assert a.stat().st_ino == b.stat().st_ino and not b.stat().st_mode & 0o222
    for row in m['shared_references']:
        assert sha(root / row['path']) == row['sha256']
    result.update(artifact_hashes_aliases_readonly_verified=len(m['files']), manifest_sha256=sha(out/'manifest.json'))
    links = []
    for name in ['ordered-surface-queries.md', 'architecture.md', 'mesher.md', 'vtm.md']:
        path = root / 'docs' / name
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('http://', 'https://', '#')):
                continue
            q = (path.parent / target.split('#', 1)[0]).resolve()
            rel = str(q.relative_to(root))
            if not q.exists():
                subprocess.run(['git', 'cat-file', '-e', 'HEAD:' + rel], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            links.append(dict(document=name, target=rel))
    result['links'] = links
result['verifier_sha256'] = sha(Path(__file__))
if args.output:
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    args.output.chmod(0o444)
print({k:v for k,v in result.items() if k != 'links'})
