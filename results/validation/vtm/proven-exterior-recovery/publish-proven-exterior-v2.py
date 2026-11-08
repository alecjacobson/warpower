"""Publish exact recovery evidence, including failed variants and negative cases."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/proven-exterior-recovery'
assert not (out/'manifest.json').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):return str(p.relative_to(root))
assert '1084 passed' in (w/'proven-exterior-full-suite-v2.log').read_text()
memory=json.loads((w/'proven-exterior-memory-v1-commands.json').read_text());assert len(memory)==8 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
for kind in ['full','loop']:
 r=json.loads((w/f'proven-exterior-{kind}-native-v1-command.json').read_text());assert r['exit_code']==0
 a=json.loads((w/f'proven-exterior-{kind}-native-v1-audit.json').read_text());assert a['gpu_kernel_activities']>0
 assert all(x not in a['dot_node_types'] for x in ['HOST','MEM_ALLOC','MEM_FREE'])
 assert all('DtoD' in x for x in a['dot_copy_kinds'])
 assert all('DTOD' in x for x in a['gpu_memcpy_kinds'])
final=json.loads((w/'proven-exterior-final-independent-v2.json').read_text());assert len(final['healthy_equivalence'])==9 and len(final['standalone'])==5
commands=json.loads((w/'proven-exterior-final/commands.json').read_text());assert len(commands)==37
physical=json.loads((w/'proven-exterior-physical-cells-v1.json').read_text());assert physical['all_snapshot_hashes_match_independent_reports']
variation=json.loads((w/'proven-exterior-baseline-variation-v3.json').read_text());assert variation['audited_existing_baseline_variation'] and variation['other_healthy_pairs_canonical_unchanged']
assert [k for k,v in final['healthy_equivalence'].items() if not v['comparison']['canonical_unchanged']]==['hand-seed1']
runtime=next(r['source'] for r in commands if r['phase']=='after');assert runtime==subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
prefixes=('proven_exterior','proven-exterior','proven-recovery','proven-filter','centroid-ray-filter','hand-seed7','hand-seed2026','hand-full-proven','hand-proven-exterior','hand_seed_recovery','hand-seed-recovery','continue-hand-failure','export-hand-','check-hand-','prove-hand-','check-proven-','run-proven-','run-hand-seed-recovery','full_proven_recovery','run-full-proven-','check-full-proven-','profile-full-proven-','run-profile-proven-','check-centroid-ray-','build-proven-','test_proven_exterior','test_proven_recovery','profile-proven-exterior')
files=[p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and (p.suffix!='.md' or p.name=='proven-exterior-filter-proof-v2.md') and '-storage-' not in p.name and 'open-issues' not in p.name]
for directory in ['hand-full-proven-recovery-v1','hand-full-proven-recovery-v2','hand-seed-recovery-midpoint-v1','proven-recovery-broad-v1','proven-exterior-final']:
 files += [p for p in (w/directory).rglob('*') if p.is_file() and not (p.name=='input.off' and p.parent.name.startswith('graded-armadillo'))]
files.extend([Path(__file__).resolve(),w/'publish-proven-exterior-v1.py']);files=sorted(files);assert len(files)==len(set(files))
meta=dict(production_runtime=runtime,baseline_runtime='a34910d64a27dbb8e0fb03f0813e76e00e799744',private_experiment_runtime='a34910d64a27dbb8e0fb03f0813e76e00e799744',scope='Pure-Warp captured final boundary recovery using midpoint-aware fitting and exact whole-exterior proofs, with transactional guards and unchanged standalone construction APIs. Robustness milestone, not CPU outperformance.',device='NVIDIA L40',warp='1.19.0',files=[],shared_helpers=[],shared_inputs=[],source_sha256={},reproduction=[
 'Use separate persistent checkouts of the pinned production and baseline commits. Build independent CGAL validators from benchmarks/cpu; binaries are not included.',
 'Restore published helper sources and required shared inputs to their original .work/vtm paths in an empty reproduction workspace. Do not prepopulate output paths: harnesses refuse to overwrite. For validating retained outputs, restore them into a separate validation workspace.',
 'Substitute checkout roots in absolute paths. Final harness expects baseline at .work/vtm/proven-exterior-baseline. Keep its pinned commit. Final and memory harnesses require the recorded full-suite pass or a new full pass at that revision.',
 'Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. for production; add .work/vtm to PYTHONPATH for private probes. GPU jobs are sequential. Independent CPU geometry checks follow clean timings.',
 'Native Nsight reports and DOT archives are lossless gzip. SQLite is regenerable from the retained report using the exact recorded export command. Private pilot VTU/OFF files may remain uncompressed; final and broad exports use gzip with hashes of original bytes.',
 'Private monkeypatch prototypes must not be installed on top of the production implementation. Their sources and results explain the investigation; use the pinned private runtime.',
 'The first failed full-suite run used direct new attribute access in SelectiveUpdate. The retained failed-compatibility patch reproduces that old access from the corrected production source. Its traceback source excerpts can show the corrected file because edits occurred after the old function was imported.',
 'The first publication manifest accidentally captured its still-open stdout log before the final111-byte message was written. That initial manifest is retained for audit. This second pass runs with a log name outside the collected prefixes and hashes the now-closed earlier log. No scientific source or result changed.',
 'Original full-suite v1 has1083 cases; the passing v2 adds one proof-only point-contact parameter. New test fixtures are exact field subsets of original/continuation snapshots, with separate hashes and field provenance.'
],limitations=[
 'Original hand seed2/seed7 and coarse-hand failures remain in baseline records. Invalid outputs are excluded from mesh-speed comparisons. Coarse hand remains invalid in production.',
 'Recovery proves removed open cell interiors exterior under valid-domain prerequisites. It preserves referenced vertices but does not establish exact constrained-feature edges or solve arbitrary input self-intersections.',
 'Independent surface and feature distances are sampled measurements, not Hausdorff bounds. Native CPU comparable-quality wide speedup remains unmet.',
 'Eight healthy pairs preserve canonical connectivity. Hand seed1 differs in8full/6material cells and2boundary triangles after coordinate differences around1e-18; the same alternate triangulations occur between identical baseline runs. Production matches the earlier baseline. Common-cell quality, reported percentiles, fit and work agree, but the six changed cell qualities and the full sorted quality arrays differ. See issue208 and the v3 baseline-variation audit.',
 'Final checker v1 stopped at the canonical mismatch. v2 retains all comparisons and validates all21 valid meshes; the separate variation audit closes that comparison. Audit v1 incorrectly expected only two changed coordinates in earlier baseline pairs (there are four); v2 incorrectly required identical sorted qualities across changed cells; v3 compares common-cell qualities and records changed-cell qualities explicitly.'
 'Private integer prototype timings, one-step continuations, and offline selection proposals are not production speed claims. Retained unsuccessful midpoint-only and oracle-harness attempts are described in proven-exterior-attempts-v1.json.',
 'Memory instrumentation uses the existing131072-byte diagnostic CUDA stack allowance. Clean benchmarks and native traces use normal runtime configuration.',
 'Native Nsight may omit graph-copy activity records; DOT directions are checked independently. First-launch instantiation/upload and one final external synchronization are present.'
])
for p in files:
 target=out/p.relative_to(w);target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():os.link(p,target)
 assert target.stat().st_ino==p.stat().st_ino;target.chmod(0o444);assert sha(p)==sha(target)
 meta['files'].append(dict(original=rel(p),path=rel(target),bytes=p.stat().st_size,sha256=sha(p)))
for original,canonical in [('representable_boundary_v1.py','representable-boundary-probe/representable_boundary_v1.py'),('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py'),('large-reuse-isolated-test.py','large-fragment-reuse/large-reuse-isolated-test.py'),('boundary-timing-commands.json','analytic-filters/boundary-timing-commands.json')]:
 p=w/original;target=root/'results/validation/vtm'/canonical;assert sha(p)==sha(target)
 meta['shared_helpers'].append(dict(original=rel(p),path=rel(target),sha256=sha(target)))
canonical=root/'results/examples/vtm/armadillo-uniform/input.off'
for p in [w/'scanned-armadillo-input/armadillo.off',*sorted((w/'proven-recovery-broad-v1').glob('graded-armadillo*/input.off')),*sorted((w/'proven-exterior-final').glob('graded-armadillo*/input.off'))]:
 assert sha(p)==sha(canonical);meta['shared_inputs'].append(dict(original=rel(p),path=rel(canonical),sha256=sha(canonical)))
for folder in ['warpower','benchmarks','examples','tests']:
 for p in sorted((root/folder).rglob('*.py')):meta['source_sha256'][rel(p)]=sha(p)
for name in ['hand-fit256.npz','proven-exterior-original.npz','proven-exterior-continuation.npz']:
 p=root/'tests/data'/name;meta['source_sha256'][rel(p)]=sha(p)
meta['reference_binaries_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference']}
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\n');p.chmod(0o444)
print(json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files),manifest=rel(p))))
