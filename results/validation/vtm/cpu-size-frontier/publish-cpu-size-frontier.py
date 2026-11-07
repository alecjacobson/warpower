"""Publish all native size pilots, including failures and quota-contaminated samples."""
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

root = Path.cwd()
w = root / '.work/vtm'
out = root / 'results/validation/vtm/cpu-size-frontier'
baseline = '336d25d2c575f980419fe6f5479852f8219976d6'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == baseline
assert not subprocess.check_output(['git', 'diff', '--', 'warpower', 'tests', 'benchmarks', 'examples', 'README.md'], text=True)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
names = ['cpu-frontier-torus-019-default-parallel-verified', 'cpu-frontier-torus-016-default-parallel', 'cpu-frontier-torus-0163-default-sequential', 'cpu-frontier-torus-0163-default-parallel']
build = json.loads((root/'results/validation/vtm/cpu-mesh3-build.json').read_text())
assert sha(root/'benchmarks/cpu/mesh3.cpp') == build['source_sha256']
summaries = {}
for name in names:
    d = json.loads((w/name/'result.json').read_text())
    assert len(d['runs']) == 3
    assert d['binary_sha256'] in [b['binary_sha256'] for b in build['builds']]
    assert d['surface_sha256'] == sha(root/'results/validation/vtm/cpu-mesh3-torus-input.off')
    for i, r in enumerate(d['runs']):
        raw = gzip.decompress((w/name/f'mesh-{i}.txt.gz').read_bytes())
        assert hashlib.sha256(raw).hexdigest() == r['mesh_sha256']
        q = r['independent_quality']
        assert q['referenced_vertices'] == r['vertices']
        assert q['components'] == 1 and q['boundary_euler'] == 0
        assert q['radius_ratio_percentiles']['0'] > 0
        assert r['odt_return_code'] == 3
        assert r['cpu_throttled'] == any(x['throttled_periods'] or x['throttled_usec'] for x in r['cpu_quota_delta'].values())
    clean = [r['complete_seconds'] for r in d['runs'] if not r['cpu_throttled']]
    assert len(clean) == d['unthrottled_samples']
    assert float(np.median(clean)) == d['median_unthrottled_complete_seconds']
    summaries[name] = {k:d[k] for k in ['parameters','median_complete_seconds','unthrottled_samples','median_unthrottled_complete_seconds']}

assert not out.exists()
out.mkdir(parents=True)
files=[]
def copy(p, relative):
    q = out/relative
    q.parent.mkdir(parents=True, exist_ok=True)
    os.link(p,q)
    files.append(dict(original=str(p.relative_to(root)),path=str(q.relative_to(root)),bytes=q.stat().st_size,sha256=sha(q)))
for name in names:
    copy(w/(name+'.log'), name+'.log')
    for p in sorted((w/name).iterdir()):
        if p.is_file():
            copy(p,p.relative_to(w))
for name in ['run-cpu-frontier.py','publish-cpu-size-frontier.py','cpu-frontier-lossless-mesh-archives.json','cpu-frontier-torus-019-default-parallel.log']:
    copy(w/name,name)
gpu = root/'results/validation/vtm/prepared-point-queries'
refs = [root/'results/validation/vtm/cpu-mesh3-build.json',root/'results/validation/vtm/cpu-mesh3-torus-input.off',gpu/'prepared-query-final/torus4096-after.json',gpu/'prepared-query-final/torus4096-after.npz',gpu/'prepared-query-final-independent.json']
enum = Path('/home/horde/references/CGAL-6.2.1/include/CGAL/Mesh_optimization_return_code.h')
assert 'CONVERGENCE_REACHED' in enum.read_text()
manifest = dict(baseline=baseline,gpu_runtime='6a82244d67bf261a5295c2225e6208a84a90abea',scope='Four size/reference batches; all twelve samples retained. Counts and quality differ. This is not CPU outperformance or an equal-quality speedup comparison. No runtime change.',native_source_sha256=sha(root/'benchmarks/cpu/mesh3.cpp'),native_enum=dict(path=str(enum),sha256=sha(enum),return_code_3='CONVERGENCE_REACHED'),shared_references=[dict(path=str(p.relative_to(root)),sha256=sha(p)) for p in refs],summaries=summaries,files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in files:
    assert sha(root/r['path']) == r['sha256']
print('Verified',len(files),'files;',sum(r['bytes'] for r in files),'bytes; all twelve native meshes verified')
