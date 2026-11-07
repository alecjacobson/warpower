"""Publish all private #130 sphere-cache variants without runtime adoption."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = root / 'results/validation/vtm/temporal-seed-probe'
baseline = '651e35f21d1fde33437c67e2721f9d190f7254c2'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == baseline
assert not subprocess.check_output(['git', 'diff', '--', 'warpower', 'tests', 'benchmarks', 'examples', 'README.md'], text=True)
assert '15 failed, 54 passed' in (w / 'temporal-seed-existing.log').read_text()
for version in [1, 2, 3]:
    assert '69 passed' in (w / f'temporal-seed-existing-v{version}.log').read_text()
for name, count in [('temporal-seed-mask-tests.log', 18), ('temporal-seed-mask-tests-v2.log', 18), ('temporal-seed-hybrid-masks-v3.log', 24)]:
    assert f'{count} passed' in (w / name).read_text()
folders = ['temporal-seed-pilot', 'temporal-seed-v2-pilot', 'temporal-seed-v3-pilot']
for folder in folders:
    rows = json.loads((w / folder / 'commands.json').read_text())
    assert len(rows) == 4 and all(r['exit_code'] == 0 and r['quota_clean'] and r['valid'] == 1 and r['error'] == 0 for r in rows)
    checks = json.loads((w / (folder + '-independent.json')).read_text())
    assert len(checks) == 2
    for record in checks.values():
        assert all(value for key, value in record['comparison'].items() if key != 'maximum_coordinate_difference')
        assert all(record['work_equal'].values()) and record['inputs_parameters_equal']
        assert record['independent']['full_hull']['valid'] and record['independent']['all_boundary_vertex_links_single_cycles']
        assert 0 < 1 - record['after_ms'] / record['before_ms'] < 0.01
diagnostic = json.loads((w / 'temporal-seed-diagnostic-equivalence.json').read_text())
assert diagnostic['points_bitwise_equal'] and diagnostic['roles_equal'] and diagnostic['tetrahedron_sets_equal']
assert all(diagnostic['work_equal'].values())
assert diagnostic['totals'] == dict(calls=624, cells=12882912, unchanged_proved_cells=4323564, changed_queries=1054374)
assert not out.exists()
out.mkdir(parents=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

files = []

def copy(p, relative=None):
    q = out / (relative or p.name)
    q.parent.mkdir(parents=True, exist_ok=True)
    assert not q.exists()
    os.link(p, q)
    files.append(dict(original=str(p.relative_to(root)), path=str(q.relative_to(root)), bytes=q.stat().st_size, sha256=sha(q)))

sources = [
    'temporal_seed_queries.py', 'temporal_seed_queries-v0-compile-error.py',
    'temporal_seed_queries-v1.py', 'temporal_seed_queries-v2.py',
    'build-temporal-prototype.py', 'build-temporal-prototype-v0-compile-error.py',
    'build-temporal-prototype-v1.py', 'build-temporal-prototype-v2.py',
    'test-temporal-seed-existing.py', 'test_temporal_seed_masks.py',
    'test_temporal_seed_hybrid_masks.py', 'run-temporal-seed.py',
    'run-temporal-seed-pilots.py', 'run-temporal-seed-v2-pilots.py', 'run-temporal-seed-v3-pilots.py',
    'check-temporal-seed-pilots.py', 'check-temporal-seed-v2-pilots.py', 'check-temporal-seed-v3-pilots.py',
    'profile_temporal_seed.py', 'profile_temporal_seed-v2.py', 'publish-temporal-seed-probe.py',
]
for name in sources:
    copy(w / name)
excluded = {'temporal-seed-preflight-mesh-archives.json', 'temporal-seed-publication.log'}
for p in sorted(w.glob('temporal-seed-*')):
    if p.is_file() and p.suffix in {'.log', '.json', '.npz'} and p.name not in excluded:
        copy(p)
for folder in folders:
    for p in sorted((w / folder).iterdir()):
        if p.is_file():
            copy(p, p.relative_to(w))
helper = root / 'results/validation/vtm/large-fragment-reuse/check-large-reuse-proof-pairs.py'
assert helper.read_bytes() == (w / 'check-large-reuse-proof-pairs.py').read_bytes()
manifest = dict(
    baseline=baseline,
    scope='Private temporal retained-cell sphere-proof experiments, unadopted. v0 compilation failure retained; v1/v2/v3 small complete-solve improvements below1%. No runtime/public-test/default/README change and no broad adoption memory/native audit claimed.',
    variants={'v0': 'temporal_seed_queries-v0-compile-error.py', 'v1': 'temporal_seed_queries-v1.py', 'v2': 'temporal_seed_queries-v2.py', 'v3': 'temporal_seed_queries.py'},
    diagnostic_variant='v2; use profile_temporal_seed-v2.py with temporal_seed_queries-v2.py restored to its original import name',
    runtime_source_sha256={str(p): sha(p) for p in sorted(Path('warpower').glob('*.py'))},
    shared_test_sources_sha256={p: sha(root / p) for p in ['tests/test_grouped_query_masks.py', 'tests/test_predicates.py']},
    shared_helper=dict(path=str(helper.relative_to(root)), sha256=sha(helper)),
    reference_binaries_sha256={name: sha(w / name) for name in ['check-delaunay', 'surface-distance']},
    files=files,
)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for record in files:
    assert sha(root / record['path']) == record['sha256']
print('Verified', len(files), 'artifacts;', sum(r['bytes'] for r in files), 'bytes')
