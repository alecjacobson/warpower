"""Publish exact query reuse, retained attempts and pinned production evidence."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = root / 'results/validation/vtm/surface-query-cache'
assert not (out / 'manifest.json').exists()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p): return str(p.relative_to(root))
assert '1113 passed' in (w / 'surface-query-cache-full-suite-v1.log').read_text()
assert '239 passed' in (w / 'surface-query-cache-production-focused-v1.log').read_text()
for mode, count in [('retained',53),('full',106)]:
    r = json.loads((w / f'surface-query-cache-{mode}-production-check-v1.json').read_text())
    assert r['finished'] and len(r['cases']) == count
memory = json.loads((w / 'surface-query-cache-memory-v1-commands.json').read_text())
assert len(memory) == 6 and all(r['exit_code'] == 0 and r['zero_memcheck_errors'] for r in memory)
for kind in ['full','loop']:
    stem = w / f'surface-query-cache-{kind}-native-v1'
    r = json.loads(Path(str(stem)+'-command.json').read_text()); assert r['exit_code'] == 0
    a = json.loads(Path(str(stem)+'-audit.json').read_text()); assert a['gpu_kernel_activities'] > 0
    assert all(x not in a['dot_node_types'] for x in ['HOST','MEM_ALLOC','MEM_FREE'])
    assert all('DtoD' in x for x in a['dot_copy_kinds'])
    assert all('DTOD' in x for x in a['gpu_memcpy_kinds'])
    assert not any(part in name.lower() for name in a['cuda_api_calls'] for part in ['alloc','free','memcpy'])
    assert a['cuda_api_calls'].get('cuCtxSynchronize') == 1
    result = json.loads(stem.with_suffix('.json').read_text())
    assert result['valid'] == 1 and result['error'] == 0 and result['proof_removed'] > 0
    assert result['cache_storage']['total_array_bytes'] == 10325560
    if kind == 'loop':
        rows = result['metadata']
        assert [r[0] for r in rows] == [0,1,0,0] and [r[1] for r in rows] == [1,0,1,1]
        assert [r[2] for r in rows] == [1,0,0,1] and rows[1][4:] == [0,0,0,0]
final = json.loads((w / 'surface-query-cache-final-independent-v3.json').read_text())
assert final['complete'] and len(final['equivalence']) == 13 and len(final['standalone']) == 5
assert final['equivalence_failures']==['hand-seed1'] and final['accepted_material_variation']['case']=='hand-seed1'
assert sum(c['valid'] for phases in final['cases'].values() for c in phases.values()) == 24
commands = json.loads((w / 'surface-query-cache-final/commands.json').read_text())
assert len(commands) == 36 and all(r['quota_clean'] for r in commands)
runtime = next(r['source'] for r in commands if r['phase'] == 'after')
assert runtime == subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md']) == b''
prefixes = ('surface-query-cache','surface_query_cache','surface-query-reuse','surface_query_reuse',
            'surface_query_epoch','surface_query_observer','run-surface-query','check-surface-query',
            'profile-surface-query','generate-surface-query','proposed_test_surface_query','gate-surface-query')
files = [p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes)
         and not any(x in p.name for x in ['-storage','-archived-','-progress','-issue-','-pr-','-report-draft'])]
directories = [f'surface-query-cache-pilot-v{i}' for i in [1,2,3]]
directories += [f'surface-query-reuse-observer-v{i}' for i in [1,2,3]]
directories += ['surface-query-cache-final','proposed-query-cache-runtime-v1','proposed-query-cache-runtime-v2','surface-query-cache-repeat-before-v1','surface-query-cache-repeat-after-v1','surface-query-cache-paired-audit-hand1-v1']
inputs = [w / 'scanned-armadillo-input/armadillo.off']
for directory in directories:
    for p in (w / directory).rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts: continue
        if p.name == 'input.off' and p.parent.name.startswith('graded-armadillo'): inputs.append(p)
        else: files.append(p)
files += [Path(__file__).resolve(),w/'verify-surface-query-cache-publication-v1.py']
files = sorted(files); assert len(files) == len(set(files))
meta = dict(production_runtime=runtime,baseline_runtime='07aa4c78fc69fdeae468db457f455c953ca84c06',
            scope='Private full-VTM exact surface-query reuse with device generation. No native CPU outperformance claim.',
            device='NVIDIA L40',warp='1.19.0',files=[],shared_helpers=[],shared_inputs=[],source_sha256={},
            reproduction=[
                'Use separate persistent checkouts of the pinned baseline and production commits; restore helpers to original .work/vtm paths, replacing absolute roots as needed. Harnesses refuse to overwrite evidence paths.',
                'Build independent CGAL validators from benchmarks/cpu. Binaries are not included. They validate outputs outside resident solve timings.',
                'Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.; add .work/vtm for private helpers. Run one GPU job at a time, with no heavy CPU geometry validation during clean timings.',
                'Run observers and private cache prototypes at baseline 07aa4c7; production harnesses at the pinned production_runtime. The first observer has a Warp mutable-loop compile failure; the first loop-check script has a Python syntax error. Original sources/logs remain. Subsequent versions fix these separately.',
                'Observer v2 covers full and ordinary retained classifiers; v3 adds recovery selection. Both still execute original queries. Private cache v1 uses reset callbacks; v2 uses a device epoch. Only v2 device functions were promoted; the AST audit and production original-kernel tests bind this change.',
                'Observer comparisons use four preceding bounded-hull-box final outputs and its independent report. Those dependencies remain in the prior evidence manifest. Explicit own-frame exterior-variation audits retain tiny hand differences without claiming bitwise equivalence.',
                'Strict final checker v1 stops at hand-seed1 material/boundary connectivity variation. Checker v2 hash-binds its successful prefix and completes all24 valid meshes, preserving the rejected comparison. A separate rational local boundary/volume audit,16 independently validated baseline/candidate repeats, and a full-trajectory paired query audit establish baseline-existing variability. Checker v3 accepts that explicit assessment without changing the failed strict claim. The original combined gate driver returns nonzero for the retained strict CPU failure after all GPU audits pass.',
                'Final pinned controls cover 13 complete configurations and five independent APIs. Coarse-hand rejection is included as an error-path gate and excluded from valid-mesh speed comparisons.',
                'Nsight and DOT archives are lossless gzip; regenerate SQLite with the recorded export command. Memory checks use the diagnostic stack allowance; normal production timing/capture jobs do not.',
                'Run this publisher with stdout outside collected prefixes so it never hashes a still-open output.'
            ],limitations=[
                'About 42.2 MB additional cache arrays at 4k sites; seven per-consumer caches, plus an eight-byte surface generation. Hash-probe exhaustion reduces reuse and falls back to ordinary traversal.',
                'Surface changes require refit. Surface topology is fixed at setup. The device generation saturates and permanently disables reuse rather than wrapping.',
                'Distance/feature checks are sampled measures, not Hausdorff or exact constrained-edge certificates. Coarse hand remains invalid. Feature-complete comparable-quality wide native CPU outperformance remains open.',
                'Seven hand pairs have tiny coordinate differences (maximum1.1641532182693481e-10); hand7 changes exterior cells and hand1 changes six material cells/two boundary faces. Both hand1 alternatives occur in baseline and candidate repeats. Its mean radius ratio differs by3.681778328e-5; sampled fit and feature metrics match. No bitwise trajectory, identical material quality, equal statistical distributions or unique causal-kernel claim.',
                'Unchanged standalone controls do not attribute speed changes to this private cache. Instrumented/native-profile times are diagnostic, not clean performance results.'
            ])
for p in files:
    target = out / p.relative_to(w); target.parent.mkdir(parents=True,exist_ok=True)
    assert not p.is_symlink()
    if not target.exists(): os.link(p,target)
    assert p.stat().st_ino == target.stat().st_ino
    target.chmod(0o444)
    meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [
    ('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),
    ('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),
    ('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),
    ('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json'),
    ('bounded_hull_box_variation_v1.py','bounded-hull-box/bounded_hull_box_variation_v1.py')]:
    p=w/original; target=root/'results/validation/vtm'/canonical; assert sha(p)==sha(target)
    meta['shared_helpers'].append(dict(original=rel(p),path=rel(target),sha256=sha(target)))
canonical=root/'results/examples/vtm/armadillo-uniform/input.off'
for p in inputs:
    assert sha(p)==sha(canonical)
    meta['shared_inputs'].append(dict(original=rel(p),path=rel(canonical),sha256=sha(canonical)))
meta['prior_control_manifest']=dict(path='results/validation/vtm/bounded-hull-box/manifest.json',sha256=sha(root/'results/validation/vtm/bounded-hull-box/manifest.json'))
for folder in ['warpower','benchmarks','examples','tests']:
    for p in sorted((root/folder).rglob('*.py')): meta['source_sha256'][rel(p)]=sha(p)
for name in ['hand-fit256.npz','proven-exterior-original.npz','proven-exterior-continuation.npz']:
    p=root/'tests/data'/name;meta['source_sha256'][rel(p)]=sha(p)
meta['reference_binaries_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),manifest=rel(p))))
