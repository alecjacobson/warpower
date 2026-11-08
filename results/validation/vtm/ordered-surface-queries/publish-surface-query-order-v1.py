"""Publish ordered surface-query evidence, including failed and rejected private trials."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
out = root / 'results/validation/vtm/ordered-surface-queries'
assert not out.exists()
subprocess.run([str(root/'.venv/bin/python'), str(w/'verify-surface-query-order-v1.py')], check=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

s = json.loads((w/'surface-query-order-production-source-v1.json').read_text())
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == s['runtime']
prefixes = ('surface-query-order-', 'surface_query_order_', 'run-surface-query-order-', 'profile-surface-query-order-', 'test-surface-query-order-', 'test_surface_query_order_', 'check-surface-query-order-', 'audit-surface-query-order-', 'build-surface-query-order-')
files = [p for p in w.iterdir() if p.is_file() and p.name.startswith(prefixes) and not any(x in p.name for x in ['-pr-', '-merge-', '-publication-', '-issue', '-adoption-issue', '-premerge-'])]
folders = ['surface-query-order-profile-v1', 'surface-query-order-pilot-v1', 'surface-query-order-pilot-v2', 'surface-query-order-final-v1', 'surface-query-order-production-loop-v1', 'surface-query-order-loop-native-v1-frames', 'surface-query-order-memory-v1-loop-padded-frames', 'surface-query-order-memory-v1-loop-lifetime-frames']
for folder in folders:
    files += [p for p in (w/folder).rglob('*') if p.is_file()]
files += [Path(__file__).resolve(), w/'verify-surface-query-order-v1.py', w/'write-surface-query-order-doc-v1.py']
meta = dict(scope=__doc__, runtime=s['runtime'], baseline=s['baseline'], runtime_source_sha256=s['source_sha256'], files=[], shared_references=[], native_validators={n:sha(w/n) for n in ['check-delaunay', 'surface-distance', 'surface-reference']}, limitations=['Measured GPU revision improvements do not establish the comparable-quality native CPU speedup goal.', 'Nine final pairs have bitwise equal points; tiny hand/cavity coordinate differences remain explicit. All final full/material/oriented connectivity and roles agree.', 'Original CUDA900 capture failure, static warm/small regressions and 16K-gate 4K full-solve regression are retained.', 'Ordering adds workspace and graph nodes; setup is excluded from replay timing and is disclosed separately.', 'Finite recovery, sampled fitting distances and prior combined-process instrumentation limits remain.'], reproduction=['Pinned source baseline/runtime, NVIDIA L40 and Warp 1.19.0; single GPU process and no heavy CPU validation during clean timing.', 'Restore artifact original paths plus referenced prior manifests. Drivers refuse overwrites. Adjust absolute workspace paths when relocating.', 'Use .venv/bin/python and PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm; native validators are built from benchmarks/cpu.', 'Raw Nsight reports and graph DOT are losslessly gzip-compressed with raw/compressed hashes. Memory and native diagnostic times are not performance samples.', 'Stanford/libigl Armadillo source provenance and redistribution terms remain in the referenced input publication. No asset relicensing.'])
prior = root/'results/validation/vtm/tangential-topology-recovery'
refs = [prior/n for n in ['manifest.json', 'run-tangential-production-gallery-v1.py', 'large-reuse-isolated-test.py', 'test_tangential_production_loop_v2.py', 'check-tangential-production-memory-v3.py', 'tangential-production-loop-v2-independent.json', 'tangential-production-final-v1-independent.json', 'tangential-production-final-v1/commands.json', 'tangential-production-loop-v2/result.json', 'tangential-production-loop-v2/frame0.npz', 'tangential-production-loop-v2/frame1.npz', 'tangential-production-loop-v2/frame4.npz', 'tangential-production-final-v1/torus1000-after.npz']]
refs += [root/'results/validation/vtm'/n for n in ['recovery-query-storage/manifest.json', 'recovery-query-storage/quality-query-storage-final-v1/commands.json', 'surface-bound-reuse/check-surface-bound-reuse-final-v2.py', 'large-fragment-reuse/check-large-reuse-proof-pairs.py', 'larger-active-prefixes/larger-prefix-final-v1/commands.json', 'exterior-proof-hierarchy/manifest.json', 'exterior-proof-hierarchy/scanned-armadillo-input/armadillo.off', 'exterior-proof-hierarchy/scanned-armadillo-input/provenance.json']]
for p in refs:
    meta['shared_references'].append(dict(path=str(p.relative_to(root)), sha256=sha(p)))
for p in sorted(set(files)):
    q = out/p.relative_to(w)
    q.parent.mkdir(parents=True, exist_ok=True)
    assert not q.exists()
    row = dict(original=str(p.relative_to(root)), path=str(q.relative_to(root)), sha256=sha(p), bytes=p.stat().st_size)
    os.link(p, q)
    q.chmod(0o444)
    assert sha(q) == row['sha256']
    meta['files'].append(row)
p = out/'manifest.json'
p.write_text(json.dumps(meta, indent=2)+'\n')
p.chmod(0o444)
print(dict(files=len(meta['files']), bytes=sum(r['bytes'] for r in meta['files']), manifest_sha256=sha(p)))
