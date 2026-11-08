"""Publish current-runtime schedule and native size controls; no runtime change."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess

root=Path.cwd()
w=root/'.work/vtm'
out=root/'results/validation/vtm/fixed-site-current'
assert not (out/'manifest.json').exists()
source='b44ab9e7332551a7a8a1c9bf9697f7bf8d7c109c'
assert not subprocess.check_output(['git','diff',source,'--','warpower','tests','benchmarks','examples','README.md'])
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(p):
    return str(p.relative_to(root))
for part,count in [('pilot',6),('larger',4)]:
    commands=json.loads((w/f'fixed-site-current-{part}-v1/commands.json').read_text())
    assert len(commands)==count
    assert all(r['exit_code']==0 and r['valid']==1 and r['error']==0 and r['quota_clean'] for r in commands)
    checks=json.loads((w/f'fixed-site-current-{part}-v1-independent.json').read_text())
    assert checks['complete'] and len(checks['cases'])==count
    for r in checks['cases'].values():
        q=r['independent']
        assert q['full_hull']['valid'] and q['all_boundary_vertex_links_single_cycles'] and q['boundary_self_intersection_pairs']==0
for cpu_part,count in [('cpu',4),('cpu-refined',2)]:
    cpu_root=w/f'fixed-site-current-{cpu_part}-v1'
    cpu_commands=json.loads((cpu_root/'commands.json').read_text())
    assert len(cpu_commands)==count and all(r['exit_code']==0 for r in cpu_commands)
    for row in cpu_commands:
        result=json.loads((cpu_root/row['name']/'result.json').read_text())
        assert len(result['runs'])==3
        for i,r in enumerate(result['runs']):
            q=r['independent_quality']
            assert q['components']==1 and q['boundary_euler']==0 and q['radius_ratio_percentiles']['0']>0
            packed=cpu_root/row['name']/f'mesh-{i}.txt.gz'
            assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==r['mesh_sha256']
names=['run-fixed-site-current-pilot-v1.py','run-fixed-site-current-larger-v1.py','run-fixed-site-current-cpu-v1.py','run-fixed-site-current-native-v1.py','check-fixed-site-current-pilot-v1.py','check-fixed-site-current-larger-v1.py','plot-fixed-site-current-v1.py','fixed-site-current-plot-source-v1.json','fixed-site-current-plot-v1.log']
names += [f'fixed-site-current-{part}-v1{suffix}' for part in ['pilot','larger'] for suffix in ['.log','-independent.json','-independent.log']]
names += ['fixed-site-current-cpu-v1.log']
names += ['run-fixed-site-current-cpu-refined-v1.py','fixed-site-current-cpu-refined-v1.log']
names += ['write-fixed-site-current-doc-v1.py']
names += [f'fixed-site-current-frontier-v1.{suffix}' for suffix in ['png','pdf','svg']]
files=[w/name for name in names]+[Path(__file__).resolve()]
for part in ['pilot','larger','cpu','cpu-refined']:
    files += [p for p in (w/f'fixed-site-current-{part}-v1').rglob('*') if p.is_file()]
files=sorted(files)
assert len(files)==len(set(files))
meta=dict(scope=__doc__,runtime=source,device='NVIDIA L40',warp='1.19.0',files=[],source_sha256={},shared_helpers=[],limitations=['Single torus surface and seed; existing public schedules only. No default change or universally superior schedule.','GPU resident timing excludes allocation/compilation/upload; native time includes domain construction, meshing and ODT. Setup/first execution reported separately.','Native criteria and counts differ. All three CPU samples, including quota-throttled samples if present, are retained. No quality-matched CPU speedup claim.','GPU full-hull/boundary checks are stronger than the retained native material quality checks; sampled RMS is not a Hausdorff bound.'],reproduction=['Use a separate checkout of the recorded runtime, restore original .work/vtm paths and shared helpers, replacing the workspace root as needed. Harnesses refuse overwriting outputs.','Run one GPU process at a time without concurrent heavy CPU work during clean timing. CPU quality checking follows each complete timing batch.','Use PYTHONDONTWRITEBYTECODE=1 WARP_CACHE_PATH=/opt/warpower-generated-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=.:.work/vtm. Build CGAL references using benchmarks/cpu instructions and the linked build record.'])
for p in files:
    q=out/p.relative_to(w)
    q.parent.mkdir(parents=True,exist_ok=True)
    assert not p.is_symlink() and p.is_file()
    if not q.exists():
        os.link(p,q)
    q.chmod(0o444)
    assert p.stat().st_ino==q.stat().st_ino
    meta['files'].append(dict(original=rel(p),path=rel(q),bytes=p.stat().st_size,sha256=sha(p)))
for name,canonical in [('check-surface-bound-reuse-final-v2.py','surface-bound-reuse/check-surface-bound-reuse-final-v2.py'),('check-large-reuse-proof-pairs.py','large-fragment-reuse/check-large-reuse-proof-pairs.py')]:
    p=w/name
    q=root/'results/validation/vtm'/canonical
    assert sha(p)==sha(q)
    meta['shared_helpers'].append(dict(original=rel(p),path=rel(q),sha256=sha(q)))
source_files=[p for p in (root/'warpower').glob('*.py')]
source_files += [root/name for name in ['benchmarks/mesher.py','benchmarks/cpu_mesher.py','benchmarks/mesh_quality.py','benchmarks/surface_distance.py','benchmarks/run.py','benchmarks/selective.py','tests/test_domain.py','tests/test_mesher.py','tests/test_extraction.py','benchmarks/cpu/mesh3.cpp']]
meta['source_sha256']={rel(p):sha(p) for p in sorted(source_files)}
meta['native_binary_sha256']={name:sha(w/name) for name in ['check-delaunay','surface-distance','surface-reference','cpu-mesh3-target-sequential','cpu-mesh3-target-parallel']}
meta['shared_native_input_and_build']={rel(root/'results/validation/vtm'/name):sha(root/'results/validation/vtm'/name) for name in ['cpu-mesh3-build.json','cpu-mesh3-torus-input.off']}
p=out/'manifest.json'
p.write_text(json.dumps(meta,indent=2)+'\n')
p.chmod(0o444)
for r in meta['files']:
    assert sha(root/r['path'])==sha(root/r['original'])==r['sha256']
for part in ['cpu','cpu-refined']:
    for r in json.loads((w/f'fixed-site-current-{part}-v1/lossless-archives.json').read_text()):
        p=Path(r['path'])
        assert sha(p)==r['sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==r['raw_sha256']
result=dict(files=len(files),manifest_sha256=sha(out/'manifest.json'),all_artifact_source_archive_hashes_verified=True,runtime_unchanged=source)
(w/'verified-fixed-site-current-publication-v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
