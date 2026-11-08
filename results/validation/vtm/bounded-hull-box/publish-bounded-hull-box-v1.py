"""Publish certified maximum-corner bounds, retained attempts and all final gates."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/bounded-hull-box'
assert not (out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
assert '1103 passed' in (w/'bounded-hull-box-full-suite-v1.log').read_text()
memory=json.loads((w/'bounded-hull-box-memory-v1-commands.json').read_text())
assert len(memory)==8 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
for kind in ['full','loop']:
    stem=w/f'bounded-hull-box-{kind}-native-v1'
    r=json.loads(Path(str(stem)+'-command.json').read_text());assert r['exit_code']==0
    a=json.loads(Path(str(stem)+'-audit.json').read_text());assert a['gpu_kernel_activities']>0
    assert all(x not in a['dot_node_types'] for x in ['HOST','MEM_ALLOC','MEM_FREE'])
    assert all('DtoD' in x for x in a['dot_copy_kinds']) and all('DTOD' in x for x in a['gpu_memcpy_kinds'])
    assert not any(part in name.lower() for name in a['cuda_api_calls'] for part in ['alloc','free','memcpy'])
    assert a['cuda_api_calls'].get('cuCtxSynchronize')==1
    result=json.loads(stem.with_suffix('.json').read_text())
    assert result['valid']==1 and result['error']==0 and result['proof_removed']>0 and result['midpoint'][2]>0
    if kind=='loop':
        rows=result['metadata'];assert [r[0] for r in rows]==[0,1,0,0] and [r[1] for r in rows]==[1,0,1,1]
        assert [r[2] for r in rows]==[1,0,0,1] and rows[1][4:]==[0,0,0,0]
final=json.loads((w/'bounded-hull-box-final-independent-v2.json').read_text())
assert final['complete'] and len(final['equivalence'])==13 and len(final['standalone'])==5
assert sum(c['valid'] for phases in final['cases'].values() for c in phases.values())==24
assert list(final['exterior_variation'])==['hand-seed7']
assert final['strict_failure_sha256']==sha(w/'bounded-hull-box-final-independent-v1.json')
commands=json.loads((w/'bounded-hull-box-final/commands.json').read_text())
assert len(commands)==36 and all(r['quota_clean'] for r in commands)
for version in [1,2]:
    audits=sorted((w/f'hull-box-audit-v{version}').glob('*/decision-audit.json'));assert len(audits)==4
    for p in audits:
        for instance in json.loads(p.read_text())['instances']:assert instance['statistics'][2:4]==[0,0]
runtime=next(r['source'] for r in commands if r['phase']=='after')
assert runtime==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert subprocess.check_output(['git','diff',runtime,'--','warpower','tests','benchmarks','examples','README.md'])==b''
prefixes=('bounded-hull-box','bounded_hull_box','hull-box','hull_box','run-bounded-hull-box','check-bounded-hull-box','profile-bounded-hull-box','test_bounded_hull_box','test-bounded-hull-box','run-hull-box','check-hull-box','current-torus-profile-v1','profile-current-torus-v1','run-current-torus-profile-v1')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and p.suffix!='.md' and '-storage-' not in p.name]
directories=['bounded-hull-box-final','bounded-hull-box-pilot-v1','bounded-hull-box-pilot-v2','bounded-hull-box-pilot-v3','hull-box-audit-v1','hull-box-audit-v2','hull-box-observer-v1','hull-box-observer-v2','hull-box-filter-observer-v1']
shared_inputs=[w/'scanned-armadillo-input/armadillo.off']
for directory in directories:
    for p in (w/directory).rglob('*'):
        if not p.is_file():continue
        if p.name=='input.off' and p.parent.name.startswith('graded-armadillo'):shared_inputs.append(p)
        else:files.append(p)
files += [w/'bounded-hull-box-proof-v1.md',w/'bounded-hull-box-proof-v2.md',Path(__file__).resolve(),w/'verify-bounded-hull-box-publication-v1.py']
files=sorted(files);assert len(files)==len(set(files))
meta=dict(production_runtime=runtime,baseline_runtime='c74e492888311599efacd2ce682f545604b4bc1a',scope='Certified maximum-corner signs in private pure-Warp VTM hull support. Existing exact fallback and independent diagram/Delaunay defaults remain. No native CPU outperformance claim.',device='NVIDIA L40',warp='1.19.0',files=[],shared_helpers=[],shared_inputs=[],source_sha256={},reproduction=[
    'Use separate persistent checkouts of the pinned baseline and production commits. Build independent CGAL validators from benchmarks/cpu; binaries are not included.',
    'Restore published helpers to original .work/vtm paths in an empty reproduction workspace, substituting local roots for recorded absolute paths. Do not prepopulate result paths: harnesses refuse overwrite. Validate retained evidence separately.',
    'Final harness uses .work/vtm/bounded-hull-box-baseline. Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.; add .work/vtm for private instrumentation.',
    'GPU jobs run one at a time. No heavy independent CPU validation runs during clean performance timings. CPU geometry checks may overlap correctness/sanitizer jobs.',
    'Private prototypes and observation runs precede production promotion. The AST audit binds the second prototype helpers/kernel to production. The first prototype only skips certified inside boxes; the second also skips redundant interval arithmetic for certified not-contained boxes.',
    'Duplicate audits compare old/new complete face-support masks and errors on identical current inputs. Version 1 captures the old baseline kernel; version 2 imports the explicitly retained original. Instrumented trajectory work can differ across separate runs; per-call old/new comparisons remain paired.',
    'The first Armadillo observer uses incorrect generic settings and rejects initialization before hull calls. The supplemental run uses established feature angle120 and 1/1 quadrature. Both are retained.',
    'Strict final checker v1 stops on hand7 exterior connectivity variation after eight independent successful meshes. Preserve this failure. The separate variation audit tests own-frame validity, changed exterior volumes, material quality/fit, and cross-frame sphere failures. Checker v2 binds and reuses the successful prefix, explicitly records exterior variation, and finishes all remaining checks.',
    'Nsight and DOT archives are lossless gzip; regenerate SQLite using the recorded export command. Memory checks use the diagnostic stack allowance; production timings and native traces use normal configuration.',
    'Run this publisher with stdout outside collected prefixes, e.g. publication-bounded-hull-box-v1.log, so it never hashes a still-open output.'
],limitations=[
    'Coarse hand remains explicitly invalid in both versions and is excluded from valid-mesh speed comparisons.',
    'Six final hand pairs have small coordinate changes; maximum5.820766091346741e-11. Hand7 also has eight removed/eight added exterior cells, with both own-frame full hulls valid and material/boundary/work equal. No bitwise reproducibility or causal-kernel claim.',
    'Distance and feature measurements use samples, not Hausdorff or exact constrained-edge certificates. Feature-complete comparable-quality wide CPU outperformance remains open.',
    'Native profiler activity records may omit graph copies; DOT directions are independently audited. Unchanged standalone controls do not attribute speed changes to the private filter.'
])
for p in files:
    target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():os.link(p,target)
    assert not p.is_symlink() and target.stat().st_ino==p.stat().st_ino
    target.chmod(0o444);assert sha(p)==sha(target)
    meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json')]:
    p=w/original;target=root/'results/validation/vtm'/canonical;assert sha(p)==sha(target)
    meta['shared_helpers'].append(dict(original=rel(p),path=rel(target),sha256=sha(target)))
canonical=root/'results/examples/vtm/armadillo-uniform/input.off'
for p in shared_inputs:
    assert sha(p)==sha(canonical);meta['shared_inputs'].append(dict(original=rel(p),path=rel(canonical),sha256=sha(canonical)))
for folder in ['warpower','benchmarks','examples','tests']:
    for p in sorted((root/folder).rglob('*.py')):meta['source_sha256'][rel(p)]=sha(p)
for name in ['hand-fit256.npz','proven-exterior-original.npz','proven-exterior-continuation.npz']:
    p=root/'tests/data'/name;meta['source_sha256'][rel(p)]=sha(p)
meta['reference_binaries_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),manifest=rel(p))))
