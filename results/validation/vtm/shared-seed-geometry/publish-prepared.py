"""Publish #157 only after integrated geometry, timing, memory and capture gates."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/shared-seed-geometry'
read=lambda name:json.loads((w/name).read_text())
summary=read('prepared-independent.json')
assert len(summary['cases'])==9 and len(summary['standalone'])==5
commands=read('prepared-final-commands.json')
assert len(commands)==27 and all(r['exit_code']==0 for r in commands),len(commands)
assert '643 passed' in (w/'prepared-full-suite.log').read_text()
for name in ['memcheck','lifetime']:
    s=(w/f'prepared-{name}.log').read_text()
    assert 'ERROR SUMMARY: 0 errors' in s and 'passed' in s
assert read('prepared-full-native-audit.json')['gpu_kernel_activities']>0
assert read('prepared-isolation.json')['original_seed_kernels_ast_equal']
assert not subprocess.check_output(['git','diff','68e2386','--','warpower'],text=True)
assert not out.exists()
out.mkdir(parents=True);files=[]


def copy(source, relative=None, compress=False):
    source=Path(source)
    if source.exists():data=source.read_bytes()
    else:
        packed=source.with_name(source.name+'.gz');assert packed.exists(),source
        data=gzip.decompress(packed.read_bytes())
    dest=out/(relative or source.name)
    if compress:dest=dest.with_name(dest.name+'.gz')
    assert not dest.exists(),dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(gzip.compress(data,mtime=0) if compress else data)
    decoded=gzip.decompress(dest.read_bytes()) if compress else dest.read_bytes()
    digest=hashlib.sha256(data).hexdigest();assert hashlib.sha256(decoded).hexdigest()==digest
    files.append(dict(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),uncompressed_sha256=digest))


cases=['torus1000','torus4096','graded-torus','thin-hand','cavity','disconnected','counterbored-plate','armadillo-uniform','armadillo-graded']
for phase,prefix in [('before','seed-pairs-control'),('after','prepared-final'),('prototype','seed-pairs-prepared')]:
    for case in cases:
        stem=w/f'{prefix}-{case}';folder=Path('timings')/phase
        if case.startswith('torus'):
            for suffix in ['.json','.npz','.log']:copy(stem.with_suffix(suffix),folder/(case+suffix))
        else:
            copy(stem.with_suffix('.log'),folder/(case+'.log'))
            for name in ['result.json','mesh.npz']:
                copy(stem/name,folder/case/name)
            if phase!='prototype':
                for name in ['mesh.vtu','boundary.off']:copy(stem/name,folder/case/name,compress=True)
                p=stem/'boundary-self-intersections.txt'
                if p.exists():copy(p,folder/case/p.name)
for phase,prefix in [('before','prepared-before'),('after','prepared-final')]:
    for case in ['delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
        for suffix in ['.json','.log']:copy(w/f'{prefix}-{case}{suffix}',Path('timings')/phase/(case+suffix))
for name in ['run-prepared-final.py','check-prepared-results.py','analyze-prepared-native.py','publish-prepared.py','prepared-final-commands.json','prepared-final-driver.log','prepared-targeted-tests.log','prepared-full-suite.log','prepared-memcheck.log','prepared-lifetime.log','prepared-mesher-loop.json','prepared-mesher-loop.log','prepared-independent.json','prepared-independent.log','prepared-original-torus-comparison.json','prepared-isolation.json','prepared-full-native-kernel-summary.json','seed-pairs-commands.json','seed-pairs-driver.log','seed-pairs-comparison.json','seed-pairs-comparison.log','run-prepared-pairs.py','compare-prepared-pairs.py','seed-pairs-text-compression.json']:
    copy(w/name)
for suffix in ['.json','.log','-export.log','-audit.json','-audit.log','.dot','.sqlite']:
    p=w/f'prepared-full-native{suffix}';copy(p,Path('native')/p.name,compress=suffix in ['.dot','.sqlite'])
for name in ['prepared-native-analysis.log','prepared-native-analysis-initial.log','analyze-prepared-native-initial.py']:
    copy(w/name,Path('native')/name)

for name in ['seed_filter_prepared.py','seed_filter_split.py','seed_filter_split_polled.py','seed_filter_staged.py','run-seed-prepared.py','run-seed-split.py','run-seed-staged.py','test-seed-prepared-existing.py','test-seed-staged-existing.py','test_seed_staged_bounds.py','profile-seed-split.py']:
    copy(w/name,Path('experiments')/name)
copy(w/'private-seed-comparisons.json',Path('experiments')/'private-seed-comparisons.json')
for prefix in ['seed-split','seed-prepared','seed-staged']:
    for p in sorted(w.glob(prefix+'-*')):
        if p.is_file() and p.suffix in ['.json','.log','.npz','.txt']:
            copy(p,Path('experiments')/p.name)
for suffix in ['.dot','.sqlite']:
    p=w/f'seed-split-native{suffix}';copy(p,Path('experiments')/p.name,compress=True)

for name in ['profile-reuse-inputs.py','profile-reuse-inputs-initial.py','run-reuse-input-profiles.py']:
    copy(w/name,Path('reuse-inputs')/name)
for pattern in ['reuse-input-*','reuse-inputs-*']:
    for p in sorted(w.glob(pattern)):
        if p.is_file() and p.suffix in ['.json','.log','.npz']:
            copy(p,Path('reuse-inputs')/p.name)
        elif p.is_dir():
            for name in ['result.json','mesh.npz']:copy(p/name,Path('reuse-inputs')/p.name/name)

source_files=['warpower/_facet_seeds.py','warpower/_prepared_seeds.py','warpower/_reuse_delaunay.py','warpower/_delaunay_certificate.py','warpower/_sphere_bounds.py','tests/test_facet_seeds.py','benchmarks/mesher.py','examples/vtm_gallery.py','benchmarks/mesh_quality.py','benchmarks/surface_distance.py','benchmarks/cpu/check_delaunay.cpp','benchmarks/cpu/surface_intersections.cpp','benchmarks/cpu/surface_distance.cpp']
manifest=dict(runtime=subprocess.check_output(['git','rev-parse','68e2386'],text=True).strip(),baseline='a84e274425fcc4f0a85e00bbc1e2b61648888894',experimental_runtime='a84e274425fcc4f0a85e00bbc1e2b61648888894',production_scope=read('prepared-isolation.json')['changed'],source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files},native_reference_binary_sha256={p:hashlib.sha256((w/p).read_bytes()).hexdigest() for p in ['check-delaunay','surface-reference','surface-distance']},goal_status='CPU wide outperformance unmet; default standalone kernels and device storage preserved',files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in files:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
print('published',len(files),'files',sum(r['bytes'] for r in files),'bytes',flush=True)
