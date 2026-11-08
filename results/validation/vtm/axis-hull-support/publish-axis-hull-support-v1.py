"""Publish pinned exact-coordinate hull-support evidence after all gates pass."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/axis-hull-support';assert not (out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
assert '1102 passed' in (w/'axis-hull-full-suite-v1.log').read_text()
memory=json.loads((w/'axis-hull-memory-v1-commands.json').read_text());assert len(memory)==8 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
for kind in ['full','loop']:
 r=json.loads((w/f'axis-hull-{kind}-native-v1-command.json').read_text());assert r['exit_code']==0
 a=json.loads((w/f'axis-hull-{kind}-native-v1-audit.json').read_text());assert a['gpu_kernel_activities']>0
 assert all(x not in a['dot_node_types'] for x in ['HOST','MEM_ALLOC','MEM_FREE'])
 assert all('DtoD' in x for x in a['dot_copy_kinds']) and all('DTOD' in x for x in a['gpu_memcpy_kinds'])
 assert not any(part in name.lower() for name in a['cuda_api_calls'] for part in ['alloc','free','memcpy'])
 assert a['cuda_api_calls'].get('cuCtxSynchronize')==1
final=json.loads((w/'axis-hull-final-independent-v1.json').read_text());assert len(final['equivalence'])==13 and len(final['standalone'])==5
commands=json.loads((w/'axis-hull-final/commands.json').read_text());assert len(commands)==36 and all(r['quota_clean'] for r in commands)
audits=[w/'axis-hull-audit-v1/hand7/decision-audit.json',*sorted((w/'axis-hull-audit-v2').glob('*/decision-audit.json'))]
assert len(audits)==4
for p in audits:
 a=json.loads(p.read_text())
 for instance in a['instances']:assert instance['statistics'][2:4]==[0,0],instance
runtime=next(r['source'] for r in commands if r['phase']=='after');assert runtime==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
prefixes=('axis-hull','axis_hull','run-axis-hull','check-axis-hull','profile-axis-hull','test_axis_hull','test-axis-hull')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and p.suffix!='.md' and '-storage-' not in p.name]
for directory in ['axis-hull-observer-v1','axis-hull-pilot-v1','axis-hull-audit-v1','axis-hull-audit-v2','axis-hull-final']:
 files += [p for p in (w/directory).rglob('*') if p.is_file() and not (p.name=='input.off' and p.parent.name.startswith('graded-armadillo'))]
files.extend([Path(__file__).resolve(),w/'verify-axis-hull-publication-v1.py']);files=sorted(files);assert len(files)==len(set(files))
meta=dict(production_runtime=runtime,baseline_runtime='92a8e2b3553e697a2ded817c62297c9eb9d23a34',scope='Exact coordinate-plane hull support from positive current cells and current original-site bounds, retaining the original filtered/exact fallback. Private pure-Warp VTM optimization; no native CPU outperformance claim.',device='NVIDIA L40',warp='1.19.0',files=[],shared_helpers=[],shared_inputs=[],source_sha256={},reproduction=[
 'Use separate persistent checkouts of the pinned baseline and production commits. Build independent CGAL validators from benchmarks/cpu; binaries are not included.',
 'Restore published helpers to their original .work/vtm paths in an empty reproduction workspace, substituting local roots for recorded absolute paths. Do not prepopulate result paths: the harnesses refuse overwrite. Validate retained evidence in a separate workspace.',
 'Final harness uses .work/vtm/axis-hull-baseline. Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.; add .work/vtm for private instrumentation.',
 'GPU jobs run one at a time. No heavy independent CPU validation runs during clean performance timings. CPU geometry validators may overlap correctness tests and sanitizer instrumentation.',
 'Observer and private pilot were run before runtime promotion. Private production-equivalence JSON binds the identical helper and kernel AST to the runtime. Duplicate audit v1 captures the old kernel before installing the candidate and must run on the baseline; audit v2 loads the explicitly retained original kernel and compares it against production.',
 'The observer counts exact-plane applicability; duplicate audits compare complete per-face support flags and error states on the same current inputs. Their timings are diagnostic, not speed evidence.',
 'Nsight/DOT archives are lossless gzip; regenerate SQLite from each retained native report using the recorded export command. Memory instrumentation uses the existing diagnostic stack allowance; production timings and native traces use normal configuration.',
 'Run this publisher with stdout named outside the collected prefixes, such as publication-axis-hull-support-v1.log, so it never hashes its own still-open output.'
],limitations=[
 'Coarse hand remains explicitly invalid in both versions and is excluded from valid mesh speed comparisons.',
 'All final canonical full/material/boundary connectivity and recorded work agree, but seven hand pairs have small coordinate variations; maximum 5.820766091346741e-11. The evidence does not establish bitwise reproducibility or a causal kernel.',
 'Independent distance/feature measurements use samples and are not Hausdorff certificates. Exact feature-edge conformity and wide comparable-quality native CPU outperformance remain open.',
 'Native profiler activity records may omit some graph copies; DOT directions are independently audited.'
])
for p in files:
 target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():os.link(p,target)
 assert target.stat().st_ino==p.stat().st_ino;target.chmod(0o444);assert sha(p)==sha(target)
 meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json')]:
 p=w/original;target=root/'results/validation/vtm'/canonical;assert sha(p)==sha(target);meta['shared_helpers'].append(dict(original=rel(p),path=rel(target),sha256=sha(target)))
canonical=root/'results/examples/vtm/armadillo-uniform/input.off'
for p in [w/'scanned-armadillo-input/armadillo.off',*sorted((w/'axis-hull-final').glob('graded-armadillo*/input.off'))]:
 assert sha(p)==sha(canonical);meta['shared_inputs'].append(dict(original=rel(p),path=rel(canonical),sha256=sha(canonical)))
for folder in ['warpower','benchmarks','examples','tests']:
 for p in sorted((root/folder).rglob('*.py')):meta['source_sha256'][rel(p)]=sha(p)
for name in ['hand-fit256.npz','proven-exterior-original.npz','proven-exterior-continuation.npz']:
 p=root/'tests/data'/name;meta['source_sha256'][rel(p)]=sha(p)
meta['reference_binaries_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),manifest=rel(p))))
