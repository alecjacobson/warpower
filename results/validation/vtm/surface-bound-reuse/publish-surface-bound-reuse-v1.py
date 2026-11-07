"""Publish completed query-bound experiments and production validation."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = root / 'results/validation/vtm/surface-bound-reuse'
runtime = '576ac5193125ae980da2314ed7502bd6938aeb79'
baseline = '18ecb03c10cb10e42a5089e8253ed92a24c3bae1'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == runtime
assert not subprocess.check_output(['git', 'diff', '--', 'warpower', 'tests', 'benchmarks', 'README.md'], text=True)
assert '1027 passed' in (w / 'surface-bound-reuse-full-suite-v1.log').read_text()
assert '236 passed' in (w / 'surface-bound-reuse-focused-v1.log').read_text()
jobs = json.loads((w / 'surface-bound-reuse-final/commands.json').read_text())
assert len(jobs) == 26 and all(r['exit_code'] == 0 and r['quota_clean'] for r in jobs)
final = json.loads((w / 'surface-bound-reuse-final-independent.json').read_text())
assert final['runtime'] == runtime and final['baseline'] == baseline
assert len(final['cases']) == 8 and len(final['standalone']) == 5
assert all(r['relative_change'] < .02 for r in final['standalone'].values())
for r in final['cases'].values():
    assert all(r['work_equal'].values()) and r['independent']['full_hull']['valid']
    assert r['independent']['all_boundary_vertex_links_single_cycles']
for name, count in [('memory-v1', 4), ('native', 3)]:
    rows = json.loads((w / ('surface-bound-reuse-' + name + '-commands.json')).read_text())
    assert len(rows) == count and all(r['exit_code'] == 0 for r in rows)
    if name == 'memory-v1':
        assert all(r['zero_memcheck_errors'] for r in rows)
for name in ['full', 'loop']:
    d = json.loads((w / f'surface-bound-reuse-{name}-native-audit.json').read_text())
    assert set(d['dot_copy_kinds']) == {'DtoD (DEVICE to DEVICE)'}
    assert set(d['gpu_memcpy_kinds']) <= {'CUDA_MEMCPY_KIND_DTOD'}
    assert set(d['dot_node_types']) <= {'MEMCPY', 'KERNEL', 'MEMSET', 'CONDITIONAL'}
    assert d['cuda_api_calls'].get('cudaGraphLaunch_v10000') == 1
    assert d['cuda_api_calls'].get('cuCtxSynchronize') == 1
    assert set(d['cuda_api_calls']) <= {'cuCtxSynchronize', 'cuProfilerStart', 'cudaGraphInstantiateWithFlags_v11040', 'cudaGraphLaunch_v10000', 'cudaGraphUpload_v10000', 'cudaStreamIsCapturing_v10000'}
quality = json.loads((w / 'surface-bound-reuse-public-quality.json').read_text())
assert len(quality['cases']) == 8 and all(r['all_fields_match'] and r['public_rms'] > 0 for r in quality['cases'].values())
loop = json.loads((w / 'surface-bound-reuse-loop-native.json').read_text())
assert loop['device_comparison_error'] == 0 and loop['phase'] == 8
for phase, expected in [('before', baseline), ('after', runtime)]:
    props = json.loads((w / f'surface-bound-reuse-kernel-properties-{phase}-v1.json').read_text())
    assert props['runtime'] == expected and len(props['kernels']) == 5
source_check = json.loads((w / 'surface-bound-reuse-source-check-v1.json').read_text())
assert source_check['reference_function_ast_identical_after_name_normalization']
assert source_check['standalone_geometry_sources_defaults_unchanged']
for name in ['surface-bound-reuse', 'surface-bound-investigation']:
    assert 'PENDING' not in (root / ('docs/' + name + '.md')).read_text()

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not out.exists()
out.mkdir(parents=True)
canonical = root / 'results/examples/vtm/armadillo-uniform/input.off'
canonical_hash = sha(canonical)
files, shared = [], []

def copy(p):
    h = sha(p)
    if p.name == 'input.off' and h == canonical_hash:
        shared.append(dict(original=str(p.relative_to(root)), canonical=str(canonical.relative_to(root)), sha256=h))
        return
    dest = out / p.relative_to(w)
    dest.parent.mkdir(parents=True, exist_ok=True)
    assert not dest.exists()
    os.link(p, dest)
    files.append(dict(original=str(p.relative_to(root)), path=str(dest.relative_to(root)), bytes=p.stat().st_size, sha256=h))

prefixes = (
    'surface-bound-reuse', 'run-surface-bound-reuse', 'check-surface-bound-reuse',
    'analyze-surface-bound-reuse', 'publish-surface-bound-reuse', 'write-surface-bound-reuse',
    'surface_work', 'surface-work', 'run_surface_work', 'build_surface_work',
    'surface_trajectory', 'surface_box_filter', 'surface-box-filter',
    'prove_surface_box', 'filtered_nearest', 'filtered-nearest',
    'profile_filtered', 'test_surface_box', 'cached_surface_bounds',
    'cached-surface-bounds', 'build_cached_surface', 'profile_cached_surface',
    'test_cached_surface',
)
for p in sorted(w.iterdir()):
    if not p.name.startswith(prefixes):
        continue
    if any(s in p.name for s in ['storage', 'pycache', 'protected-fixture', 'archived-mesh', 'publish.log']):
        continue
    if p.is_file() and p.suffix != '.md':
        copy(p)
    elif p.is_dir():
        for child in sorted(p.rglob('*')):
            if child.is_file():
                copy(child)
helpers = []
for private, public in [
    ('extraction_split.py', 'cooperative-surface-probe/extraction_split.py'),
    ('check-large-reuse-proof-pairs.py', 'large-fragment-reuse/check-large-reuse-proof-pairs.py'),
    ('large-reuse-isolated-test.py', 'large-fragment-reuse/large-reuse-isolated-test.py'),
    ('boundary-timing-commands.json', 'analytic-filters/boundary-timing-commands.json'),
]:
    p = root / 'results/validation/vtm' / public
    assert p.read_bytes() == (w / private).read_bytes()
    helpers.append(dict(original=str((w / private).relative_to(root)), path=str(p.relative_to(root)), sha256=sha(p)))
sources = [
    'warpower/_surface_bvh.py', 'tests/test_surface_bound_reuse.py',
    'tests/surface_nearest_reference.py', 'benchmarks/mesher.py',
    'examples/vtm_gallery.py', 'benchmarks/graph_capture.py',
]
manifest = dict(
    runtime=runtime, baseline=baseline,
    scope='Exact per-query reuse of rounded float64 child-box distances, preserving original traversal and triangle arithmetic. Earlier operation attribution and float32/tail prototypes use the recorded baseline and are not additional production changes. No comparable-quality CPU outperformance claim.',
    source_sha256={p: sha(root / p) for p in sources},
    shared_inputs=shared, shared_helpers=helpers,
    unexecuted_artifact_sources=['test_cached_surface_bounds_v1.py'],
    reference_binaries_sha256={n: sha(w / n) for n in ['check-delaunay', 'surface-distance', 'surface-reference']},
    files=files,
)
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for row in files:
    assert sha(root / row['path']) == row['sha256']
print(len(files), sum(r['bytes'] for r in files), 'public bytes', flush=True)
