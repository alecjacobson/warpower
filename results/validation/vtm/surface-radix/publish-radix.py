"""Publish final, checked surface radix evidence. Run only after all gates finish."""
import gzip,hashlib,json,shutil,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/surface-radix'
def read(name):return json.loads((w/name).read_text())
summary=read('radix-independent.json')
assert len(summary['cases'])==7 and len(summary['extra_cases'])==5 and len(summary['standalone'])==5
commands=read('radix-audit-commands.json')
assert len(commands)==16 and all(x['exit_code']==0 for x in commands),commands
assert '633 passed' in (w/'radix-full-suite.log').read_text()
for name in ['memcheck-surface','domain-lifetime']+[f'memcheck-mesher-{case}' for case in ['cube','torus','cavity','disconnected','producer','outer-loop']]:
 text=(w/f'radix-{name}.log').read_text();assert 'ERROR SUMMARY: 0 errors' in text,name
for name in ['full-native','domain-native']:
 audit=read(f'radix-{name}-audit.json');assert audit['gpu_kernel_activities']>0
assert not subprocess.check_output(['git','diff','2c95331','--','warpower'],text=True)
assert not subprocess.check_output(['git','diff','e08ecb1','--','benchmarks/domain.py','tests/test_surface_radix.py','tests/surface_traversal_reference.py'],text=True)
scope=subprocess.check_output(['git','diff','--name-only','18b18f5','--','warpower'],text=True).splitlines()
assert set(scope)=={'warpower/_surface_bvh.py','warpower/_triangle_contact.py','warpower/_sampling.py'},scope
out.mkdir(parents=True,exist_ok=True)
files=[]
def copy(src,relative=None,compress=False):
 src=Path(src);dest=out/(relative or src.name);dest.parent.mkdir(parents=True,exist_ok=True)
 if compress:
  with src.open('rb') as inp,dest.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as enc:shutil.copyfileobj(inp,enc)
  with gzip.open(dest,'rb') as inp:assert hashlib.sha256(inp.read()).hexdigest()==hashlib.sha256(src.read_bytes()).hexdigest()
 else:shutil.copyfile(src,dest)
 files.append(dict(path=str(dest.relative_to(root)),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
# Keep complete before/after numerical results, their raw samples, and paired mesh snapshots.
for phase in ['before','after']:
 for case in ['torus1000','torus4096','domain','delaunay1000','delaunay100000','delaunay-grid','delaunay-fitted-hand','diagram100000']:
  stem=w/f'radix-{phase}-{case}'
  for suffix in ['.json','.log','.npz']:
   p=stem.with_suffix(suffix)
   if p.exists():copy(p,Path('timings')/phase/(case+suffix))
 for case in ['graded-torus','thin-hand','cavity','disconnected','counterbored-plate','armadillo-uniform','armadillo-graded']:
  stem=w/f'radix-{phase}-{case}';copy(stem.with_suffix('.log'),Path('timings')/phase/(case+'.log'))
  for name in ['result.json','mesh.npz','mesh.vtu','boundary.off','boundary-self-intersections.txt']:
   p=stem/name
   if p.exists():copy(p,Path('timings')/phase/case/name)
for case in ['thin-hand','cavity','disconnected']:
 stem=w/f'radix-after-coarse-{case}';copy(stem.with_suffix('.log'),Path('coarse')/(case+'.log'))
 for name in ['result.json','mesh.npz','mesh.vtu','boundary.off','boundary-self-intersections.txt']:
  p=stem/name
  if p.exists():copy(p,Path('coarse')/case/name)
for pattern in ['radix-*-commands.json','radix-*-driver.log']:
 for p in sorted(w.glob(pattern)):copy(p,Path('commands')/p.name)
for name in ['publish-radix.py','run-radix-pilot.py','run-radix-timings.py','run-radix-scan-controls.py','check-radix-results.py','run-radix-audits.py','resume-radix-audits.py','run-radix-baseline-memory-control.py','radix-memcheck-isolated.py','radix-domain-lifetime.py','radix-independent.json','radix-original-torus-comparison.json','radix-arithmetic-scope.json','radix-independent.log','radix-full-native-kernel-summary.json','radix-domain-focused-tests.log','radix-full-suite.log','radix-memcheck-surface.log','radix-domain-lifetime.log','radix-domain-lifetime.json','radix-mesher-loop.log','radix-mesher-loop.json','radix-after-warm-setup.json','radix-after-warm-setup.log']:
 copy(w/name)
for case in ['cube','torus','cavity','disconnected','producer','outer-loop']:
 copy(w/f'radix-memcheck-mesher-{case}.log')
for name in ['radix-baseline-mixed-memory.log','radix-baseline-mixed-memory-command.json']:
 copy(w/name)
for name in ['full-native','domain-native']:
 for suffix in ['.json','.log','-export.log','-audit.json','-audit.log']:
  copy(w/f'radix-{name}{suffix}',Path('native')/(name+suffix))
 for suffix in ['.dot','.sqlite']:
  copy(w/f'radix-{name}{suffix}',Path('native')/(name+suffix+'.gz'),compress=True)
# Retain the rejected integration and benchmark-harness failures separately.
for name in ['radix-focused-tests.log','radix-focused-fixed-tests.log','radix-focused-fixed2-tests.log','radix-focused-distance-bits.log','radix-ranges-tests.log','radix-closest-debug.json','radix-domain-uninitialized-assert.log','radix-memcheck-mesher-interrupted.log','radix-memcheck-mesher-stack.log']:
 copy(w/name,Path('initial-failures')/name)
source_files=scope+['benchmarks/domain.py','tests/test_surface_radix.py','tests/surface_traversal_reference.py']
manifest=dict(runtime=subprocess.check_output(['git','rev-parse','2c95331'],text=True).strip(),benchmark=subprocess.check_output(['git','rev-parse','e08ecb1'],text=True).strip(),source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_files},baseline='18b18f566ca1bd0c9a4cf5e96caf623ab009f07c',production_scope=scope,goal_status='CPU wide outperformance unmet; standalone paths preserved',memory_scope='89 surface/sampling cases and 6 complete mesher cases pass in isolated processes; original combined mesher run interrupted after 2 passed / 1834.97s at device synchronization, no sanitizer errors, retained as INCOMPLETE in issue155; baseline control recorded separately.',files=files)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in files:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
print('published',len(files),'files',sum(x['bytes'] for x in files),'bytes',flush=True)
