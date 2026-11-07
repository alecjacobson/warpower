"""Preserve the unsuccessful full-classification split experiment for #159."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/extraction-split-probe'
assert not out.exists();out.mkdir(parents=True)
assert '40 passed in 19.47s' in (w/'extraction-split-tests.log').read_text()
for directory in ['extraction-split-pilot','extraction-split-4096']:
 p=w/directory
 c=json.loads((p/'comparison.json').read_text())
 assert all(c['arrays_equal'].values()) and c['work_equal'] and c['all_initial_and_timed_quotas_clean']
 shutil.copytree(p,out/directory)
 shutil.copy2(w/(directory+'-driver.log'),out/(directory+'-driver.log'))
for name in ['extraction_split.py','run-extraction-split.py','test-extraction-split-existing.py','run-extraction-split-pilot.py','run-extraction-split-4096.py','extraction-split-tests.log','extraction-split-properties.json','extraction-split-properties.log','publish-extraction-split.py']:
 shutil.copy2(w/name,out/name)
checks=[]
for count,part in [(1000,'extraction-split-pilot'),(4096,'extraction-split-4096')]:
 old=root/f'results/validation/vtm/shared-seed-geometry/timings/after/torus{count}.npz'
 new=out/part/'control.npz';a=np.load(old);b=np.load(new)
 matched={}
 for k in a.files:
  x,y=a[k],b[k]
  if k in ['tetrahedra','full_tetrahedra']:
   x,y=[np.sort(v,axis=1) for v in [x,y]];x,y=[v[np.lexsort(v.T[::-1])] for v in [x,y]]
  elif k=='boundary':
   x,y=[np.asarray([np.roll(r,-int(np.argmin(r))) for r in v]) for v in [x,y]];x,y=[v[np.lexsort(v.T[::-1])] for v in [x,y]]
  matched[k]=bool(np.array_equal(x,y))
 assert all(matched.values()),(count,matched)
 checks.append(dict(sites=count,existing_validated_mesh=str(old.relative_to(root)),existing_validated_mesh_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),pilot_control_sha256=hashlib.sha256(new.read_bytes()).hexdigest(),matched=matched))
(out/'existing-validation-comparison.json').write_text(json.dumps(checks,indent=2)+'\n')
source_files=['warpower/_extraction.py','warpower/_selective.py','warpower/_wide_center.py','warpower/_surface_bvh.py','tests/test_extraction.py','tests/test_wide_center.py','tests/test_selective.py','benchmarks/mesher.py']
assert not subprocess.check_output(['git','diff','9228a3e','--','warpower','benchmarks','tests','README.md'],text=True)
manifest=dict(runtime='9228a3e7b3804b0cf1b191976477e35c25dc47ef',scope='Private launch interception splits only full classify; no production change, no adoption. Selective classify_retained remains original.',source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files},files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for r in manifest['files']:assert hashlib.sha256((root/r['path']).read_bytes()).hexdigest()==r['sha256']
print('verified',len(manifest['files']),'artifacts',sum(r['bytes'] for r in manifest['files']),'bytes; inherited geometry comparison and source isolation pass')
