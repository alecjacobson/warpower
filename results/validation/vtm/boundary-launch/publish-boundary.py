"""Publish #154 evidence only after the complete-source gates finish."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/boundary-launch'
read=lambda p:json.loads((w/p).read_text())
summary=read('boundary-independent.json')
assert len(summary['cases'])==7 and len(summary['extra_cases'])==2 and len(summary['standalone'])==5
commands=read('boundary-timing-commands.json')
assert len(commands)==28 and all(x['exit_code']==0 for x in commands)
audits=read('boundary-audit-commands.json')
assert len(audits)==8 and all(x['exit_code']==0 for x in audits)
assert '633 passed' in (w/'boundary-full-suite.log').read_text()
for name in ('memcheck','lifetime'):
    text=(w/f'boundary-{name}.log').read_text()
    assert '11 passed' in text and 'ERROR SUMMARY: 0 errors' in text
assert read('boundary-full-native-audit.json')['gpu_kernel_activities']>0
assert not subprocess.check_output(['git','diff','3e33911','--','warpower'],text=True)
scope=subprocess.check_output(['git','diff','--name-only','1624d62','--','warpower'],text=True).splitlines()
assert scope==['warpower/_boundary.py'],scope
out.mkdir(parents=True,exist_ok=True);files=[]

def copy(source,relative=None,compress=False):
    source=Path(source);dest=out/(relative or source.name)
    if compress:dest=dest.with_name(dest.name+'.gz')
    dest.parent.mkdir(parents=True,exist_ok=True)
    original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if compress:
        with source.open('rb') as inp,dest.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as enc:
            shutil.copyfileobj(inp,enc)
        with gzip.open(dest,'rb') as f:assert hashlib.sha256(f.read()).hexdigest()==original_hash
    else:shutil.copyfile(source,dest)
    files.append(dict(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),uncompressed_sha256=original_hash))

for phase in ('before','after'):
    for case in ('torus1000','torus4096','delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000'):
        stem=w/f'boundary-{phase}-{case}'
        for suffix in ('.json','.npz','.log'):
            p=stem.with_suffix(suffix)
            if p.exists():copy(p,Path('timings')/phase/(case+suffix))
    for case in ('graded-torus','thin-hand','cavity','disconnected','counterbored-plate','armadillo-uniform','armadillo-graded'):
        stem=w/f'boundary-{phase}-{case}';copy(stem.with_suffix('.log'),Path('timings')/phase/(case+'.log'))
        for name in ('result.json','mesh.npz','mesh.vtu','boundary.off','boundary-self-intersections.txt'):
            p=stem/name
            if p.exists():copy(p,Path('timings')/phase/case/name,compress=p.suffix in ('.vtu','.off'))

for pattern in ('boundary-*-commands.json','boundary-*-driver.log'):
    for p in sorted(w.glob(pattern)):copy(p,Path('commands')/p.name)
for name in ('publish-boundary.py','run-boundary-timings.py','run-boundary-audits.py','check-boundary-results.py','retry-boundary-timings.py',
             'boundary-initial-paired-comparison.json','boundary-initial-quota-rejection.log',
             'boundary-independent.json','boundary-independent.log','boundary-original-torus-comparison.json',
             'boundary-full-suite.log','boundary-memcheck.log','boundary-lifetime.log','boundary-mesher-loop.json',
             'boundary-mesher-loop.log','boundary-full-native-kernel-summary.json'):
    copy(w/name)
for name in ('boundary_reduction_kernel.py','boundary-reduction-toy.py','boundary-reduction-toy.log','boundary-reduction-toy.json',
             'boundary-reduction-probe.py','boundary-reduction-probe.log','boundary-reduction-probe.json','boundary-reduction-tests128.log',
             'boundary-variant-run.py','run-boundary-pilot.py','boundary-pilot-comparison.json'):
    copy(w/name,Path('experiments')/name)
for name in ('boundary-pilot-before-torus.json','boundary-pilot-scalar32-torus.json','boundary-pilot-before-plate/result.json','boundary-pilot-scalar32-plate/result.json'):
    copy(w/name,Path('experiments')/name)
for suffix in ('.json','.log','-export.log','-audit.json','-audit.log','.dot','.sqlite'):
    p=w/f'boundary-full-native{suffix}';copy(p,Path('native')/p.name,compress=suffix in ('.dot','.sqlite'))
for p in sorted((w/'boundary-initial-throttled').rglob('*')):
    if p.is_file():copy(p,Path('initial-throttled')/p.relative_to(w/'boundary-initial-throttled'),compress=p.suffix in ('.vtu','.off'))
source_files=['warpower/_boundary.py','warpower/_site_bvh.py','benchmarks/mesher.py','examples/vtm_gallery.py','tests/test_boundary.py']
manifest=dict(runtime=subprocess.check_output(['git','rev-parse','3e33911'],text=True).strip(),baseline='1624d62c89d56ad5f3e801c67fd724bbe450a7f4',
              production_scope=scope,source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files},
              goal_status='CPU wide outperformance unmet; standalone source paths preserved',files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in files:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
print('published',len(files),'files',sum(r['bytes'] for r in files),'bytes',flush=True)
