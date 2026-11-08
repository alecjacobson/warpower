"""Publish pinned containment-filter experiments and validation, including failed attempts."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/containment-filters';assert not (out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
assert '1100 passed' in (w/'containment-filter-full-suite-v1.log').read_text()
memory=json.loads((w/'containment-filter-memory-v1-commands.json').read_text());assert len(memory)==8 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
for kind in ['full','loop']:
 r=json.loads((w/f'containment-filter-{kind}-native-v1-command.json').read_text());assert r['exit_code']==0
 a=json.loads((w/f'containment-filter-{kind}-native-v1-audit.json').read_text());assert a['gpu_kernel_activities']>0
 assert all(x not in a['dot_node_types'] for x in ['HOST','MEM_ALLOC','MEM_FREE'])
 assert all('DtoD' in x for x in a['dot_copy_kinds']) and all('DTOD' in x for x in a['gpu_memcpy_kinds'])
final=json.loads((w/'containment-filter-final-independent-v1.json').read_text());assert len(final['equivalence'])==13 and len(final['standalone'])==5
commands=json.loads((w/'containment-filter-final/commands.json').read_text());assert len(commands)==36 and all(r['quota_clean'] for r in commands)
variation=json.loads((w/'containment-filter-pilot-variation-v1.json').read_text());assert variation['all_material_meshes_equivalent']
trajectory=json.loads((w/'containment-trajectory-v1/commands.json').read_text());assert len(trajectory)==4 and all(r['exit_code']==0 and r['decision-audit.json']['data']['statistics'][1]==0 for r in trajectory)
runtime=next(r['source'] for r in commands if r['phase']=='after');assert runtime==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
prefixes=('containment-filter','containment-trajectory','containment_trajectory','containment_exact_baseline','run-containment-','check-containment-','profile-containment-','test_containment_ray_filter_failed')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and p.suffix!='.md' and '-storage-' not in p.name]
for directory in ['containment-filter-pilot-v1','containment-filter-pilot-v2','containment-trajectory-v1','containment-filter-final']:
 files += [p for p in (w/directory).rglob('*') if p.is_file() and not (p.name=='input.off' and p.parent.name.startswith('graded-armadillo'))]
files.append(Path(__file__).resolve());files=sorted(files);assert len(files)==len(set(files))
meta=dict(production_runtime=runtime,baseline_runtime='96027fbb07625b144106c2c36d92870b3a7402db',scope='Certified binary64 containment filters with unchanged integer fallback and recovery policy. Pure-Warp device-only optimization; no native CPU outperformance claim.',device='NVIDIA L40',warp='1.19.0',files=[],shared_helpers=[],shared_inputs=[],source_sha256={},reproduction=[
 'Use separate persistent checkouts of the pinned baseline and production commits. Build independent CGAL validators from benchmarks/cpu; binaries are not included.',
 'Restore published helpers to their original .work/vtm paths in an empty reproduction workspace, substituting local roots for recorded absolute paths. Do not prepopulate result paths: the harnesses refuse overwrite. Validate retained evidence in a separate workspace.',
 'Final harness uses .work/vtm/containment-baseline. Full suite must pass at the pinned candidate before final and memory gates. Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.; add .work/vtm for trajectory instrumentation.',
 'GPU jobs run one at a time. No heavy independent CPU validation runs during clean performance timings. CPU geometry validators may overlap correctness tests and sanitizer instrumentation.',
 'Trajectory instrumentation replaces only the containment-add kernel at recording time. It computes legacy and candidate decisions for every eligible recovery cell, then uses the candidate result. Recorded timings of these instrumented solves are not performance evidence. Legacy module matches the pinned baseline with import-only rewrites.',
 'The first focused test harness used unsupported indexed tuple stores. Its failed source/log and corrected passing run are retained. Pilot v1 reused an after-directory because an export-loop variable shadowed the scenario name; it was stopped, and the original first candidate geometry was overwritten. Its timing is excluded. Pilot v2 uses fresh unique directories and overwrite checks.',
 'Pilot checker v2 stopped on exterior full-hull variation. V3 validates all ten outputs and preserves the differences; the separate audit binds exact changed exterior volumes, per-material-cell qualities, fit and work to the independent checks.',
 'Nsight/DOT archives are lossless gzip; regenerate SQLite from each retained native report using the recorded export command. Memory instrumentation uses the existing diagnostic stack allowance; production timings and native traces use normal configuration.',
 'Run this publisher with stdout named outside the collected prefixes, such as publication-containment-filter-v1.log, so it never hashes its own still-open output.'
],limitations=[
 'Coarse hand remains explicitly invalid in both versions and is excluded from valid mesh speed comparisons.',
 'Four exterior full-hull cells differ in one pilot pair after near-zero coordinate changes; all material cells, boundary, work and measured quality agree. The audit does not claim bitwise reproducibility or a traced causal kernel.',
 'Independent distance/feature measurements use samples and are not Hausdorff certificates. Exact feature-edge conformity and wide native CPU outperformance remain open.',
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
for p in [w/'scanned-armadillo-input/armadillo.off',*sorted((w/'containment-filter-final').glob('graded-armadillo*/input.off'))]:
 assert sha(p)==sha(canonical);meta['shared_inputs'].append(dict(original=rel(p),path=rel(canonical),sha256=sha(canonical)))
for folder in ['warpower','benchmarks','examples','tests']:
 for p in sorted((root/folder).rglob('*.py')):meta['source_sha256'][rel(p)]=sha(p)
for name in ['hand-fit256.npz','proven-exterior-original.npz','proven-exterior-continuation.npz']:
 p=root/'tests/data'/name;meta['source_sha256'][rel(p)]=sha(p)
meta['reference_binaries_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),manifest=rel(p))))
