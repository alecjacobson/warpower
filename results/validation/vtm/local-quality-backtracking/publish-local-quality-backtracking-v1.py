"""Publish completed local-retry evidence; preserve every attempted policy and failure."""
import hashlib
import json
import os
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = root / 'results/validation/vtm/local-quality-backtracking'
assert not (out / 'manifest.json').exists()


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return str(p.relative_to(root))


memory = json.loads((w / 'local-backtracking-memory-v2-commands.json').read_text())
assert len(memory) == 6 and all(r['exit_code'] == 0 and r['zero_memcheck_errors'] for r in memory)
native = json.loads((w / 'local-backtracking-native-commands.json').read_text())
assert len(native) == 3 and all(r['exit_code'] == 0 for r in native)
assert '1034 passed' in (w / 'local-backtracking-production-full-suite-v1.log').read_text()
assert '46 passed' in (w / 'local-backtracking-production-focused-v2.log').read_text()
final = json.loads((w / 'local-backtracking-final-independent-v2.json').read_text())
assert sum(v.get('valid', 1) != 0 for c in final['cases'].values() for v in c.values()) == 27
assert len(final['standalone']) == 5 and len(final['default_equivalence']) == 3

prefixes = ('local_backtracking', 'local-backtracking', 'test_local_backtracking',
            'run-local-backtracking', 'run-local-blocking', 'check-local-backtracking',
            'build-local-blocking', 'plot-local-backtracking', 'write-local-backtracking',
            'analyze-local-backtracking')
unexecuted = ['check-local-backtracking-final-v1.py', 'run-local-backtracking-native-v1.py']
files = [p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes)
         and p.suffix != '.md' and p.name not in unexecuted]
for directory in ['local-backtracking-full-v1', 'local-backtracking-pilots-v1',
                  'local-backtracking-pilots-v2', 'local-backtracking-pilots-v3',
                  'local-backtracking-broad-v1', 'local-backtracking-final']:
    files += [p for p in (w / directory).rglob('*') if p.is_file()
              and not (p.name == 'input.off' and p.parent.name.startswith('armadillo-'))]
files += [Path(__file__)]
files = sorted(files)
assert len(files) == len(set(files))

meta = dict(
    production_runtime='6aa4ba63a34910ba7e9148966dd8a9015a261d06',
    baseline_runtime='39f8f54fa5cc410b46cd9b057e617fc92f7e4c28',
    private_experiment_runtime='39f8f54fa5cc410b46cd9b057e617fc92f7e4c28',
    attribution_runtime='4334f459fe743d662e600844bf8bdfe01ae25bd3',
    attribution_runtime_note='Attribution ran before the docs-only PR203 merge. Library sources are identical to baseline39f8f54.',
    scope='Opt-in on-device local quality backtracking; complete-solve quality/time tradeoffs, unchanged defaults/standalone APIs, full tests, memory and native capture evidence. No CPU outperformance claim.',
    device='NVIDIA L40', warp='1.19.0', files=[], shared_inputs=[], shared_helpers=[], source_sha256={},
    reproduction=[
        'Use separate persistent checkouts of the pinned baseline and production commits. Install dependencies and build independent validators from benchmarks/cpu; executables are not included.',
        'Restore source helpers and required shared inputs to their original .work/vtm paths in an empty reproduction workspace. Do not prepopulate generated output paths: harnesses reject existing outputs. Published outputs can instead be restored into a separate validation workspace for the independent checkers.',
        'Substitute checkout roots in recorded absolute paths. The final harness expects a baseline worktree at .work/vtm/local-backtracking-baseline. Preserve the pinned commit and code/test files there.',
        'Use PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 WARP_CACHE_PATH=/opt/warpower-generated-cache, PYTHONPATH=. for production and PYTHONPATH=.:.work/vtm for private probes. Run GPU jobs sequentially; independent CPU validators follow clean timings.',
        'Final harness v1 stopped at the baseline seed7 failure. v2 resumes the recorded prefix without replacing outputs. All41 commands are retained. Native v2 follows all six completed memory-v2 jobs. v1 memory timeout and earlier failed tests/runner logs are retained.',
        'Some original baseline test log prerequisites are recorded gates, not experiment inputs. Re-run tests at the corresponding pinned runtime before the measurement harness. Private policy sources are experiments and must not be installed on top of the production helper.',
        'mesh.vtu.gz, boundary.off.gz, graph DOT and Nsight gzip archives are lossless. Restore uncompressed copies to disposable scratch when needed. SQLite exports are regenerable from the retained Nsight reports; exact export/analyze commands are in the audit script.'
    ],
    omitted_unexecuted_sources=unexecuted,
    limitations=[
        'Private pilots are exploratory instrumented variants, not production performance evidence. Direct blocking with ordinary rounding is the only adopted option; midpoint correction, halving factors, and old-star blocking remain unadopted.',
        'All tested policies and budget points are retained, including slower runs and worse minima/fit. Default local_quality_retries remains zero.',
        'Four final production results are explicitly invalid: coarse-hand and dense-hand seed7 before/enabled. They have error64 and zero public counts, and are excluded from mesh-speed results. Gallery does not export their invalid candidate meshes.',
        'Sampled surface/crease errors are not Hausdorff bounds or proof of constrained-feature conformity. All three feature-table crease RMS values worsen with this option.',
        'Original combined padded sanitizer run timed out at600s; no pass is inferred. Same seven tests passed in isolated5/1/1 groups in both padded and pool-disabled lifetime modes at900s limits, no kernel exclusions.',
        'Memory instrumentation uses the existing131072-byte diagnostic stack limit. Ordinary production benchmarks and native traces use normal runtime configuration.',
        'Native Nsight may omit graph-copy activity records; DOT copy direction is independently checked. First-launch instantiation/upload and a final external synchronization are present.',
        'The overall native CPU comparable-quality speedup goal remains unmet.'
    ])

for p in files:
    target = out / p.relative_to(w)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(p, target)
    assert p.stat().st_ino == target.stat().st_ino
    target.chmod(0o444)
    assert sha(target) == sha(p)
    meta['files'].append(dict(original=rel(p), path=rel(target), bytes=p.stat().st_size, sha256=sha(p)))

shared = [
    ('shared_helpers', 'representable_boundary_v1.py', 'representable-boundary-probe/representable_boundary_v1.py'),
    ('shared_helpers', 'test_representable_transactions_v2.py', 'representable-boundary-probe/test_representable_transactions_v2.py'),
    ('shared_helpers', 'check-surface-bound-reuse-final-v2.py', 'surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),
    ('shared_helpers', 'check-large-reuse-proof-pairs.py', 'large-fragment-reuse/check-large-reuse-proof-pairs.py'),
    ('shared_helpers', 'large-reuse-isolated-test.py', 'large-fragment-reuse/large-reuse-isolated-test.py'),
    ('shared_inputs', 'boundary-timing-commands.json', 'analytic-filters/boundary-timing-commands.json'),
    ('shared_inputs', 'representable-full-v1/dense-hand-candidate/mesh.npz', 'representable-boundary-probe/representable-full-v1/dense-hand-candidate/mesh.npz'),
]
for group, original, canonical in shared:
    p, target = w / original, root / 'results/validation/vtm' / canonical
    assert sha(p) == sha(target), (p, target)
    meta[group].append(dict(original=rel(p), path=rel(target), sha256=sha(target)))
canonical = root / 'results/examples/vtm/armadillo-uniform/input.off'
for p in [w / 'scanned-armadillo-input/armadillo.off', *sorted((w / 'local-backtracking-final').glob('armadillo*/input.off'))]:
    assert sha(p) == sha(canonical)
    meta['shared_inputs'].append(dict(original=rel(p), path=rel(canonical), sha256=sha(canonical)))
for folder in ['warpower', 'benchmarks', 'examples', 'tests']:
    for p in sorted((root / folder).rglob('*.py')):
        meta['source_sha256'][rel(p)] = sha(p)
meta['source_sha256']['tests/data/hand-fit256.npz'] = sha(root / 'tests/data/hand-fit256.npz')
meta['reference_binaries_sha256'] = {p: sha(w / p) for p in ['check-delaunay', 'surface-distance', 'surface-reference']}
(out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
(out / 'manifest.json').chmod(0o444)
print(json.dumps(dict(files=len(files), bytes=sum(p.stat().st_size for p in files), manifest=rel(out / 'manifest.json'))))
