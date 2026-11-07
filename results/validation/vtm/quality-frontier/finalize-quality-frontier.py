import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/quality-frontier'
for name in ['quality-frontier-publication.log','quality-frontier-plot.log','write-quality-frontier-doc.py','finalize-quality-frontier.py']:
 shutil.copy2(w/name,out/name)
source_files=['benchmarks/mesher.py','benchmarks/mesh_quality.py','benchmarks/surface_distance.py','benchmarks/cpu/check_delaunay.cpp','benchmarks/cpu/surface_intersections.cpp','benchmarks/cpu/surface_distance.cpp','tests/test_domain.py','tests/test_mesher.py','tests/test_extraction.py']
source_files += [str(p.relative_to(root)) for p in sorted((root/'warpower').glob('*.py'))]
assert not subprocess.check_output(['git','diff','9228a3e','--','warpower','benchmarks','tests','README.md'],text=True)
manifest=dict(runtime='9228a3e7b3804b0cf1b191976477e35c25dc47ef',scope='Existing-API schedule comparison, no runtime, test, benchmark or README changes. No CPU speedup claim.',source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_files},native_reference_binary_sha256={p:hashlib.sha256((w/p).read_bytes()).hexdigest() for p in ['check-delaunay','surface-reference','surface-distance']},files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
for row in manifest['files']:assert hashlib.sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
for path in ['docs/quality-frontier.md','docs/mesher.md','docs/refinement.md','docs/cpu-mesher-baseline.md']:
 p=root/path
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' in target or target.startswith('#'):continue
  assert (p.parent/target.split('#')[0]).exists(),(path,target)
print('verified',len(manifest['files']),'artifacts',sum(r['bytes'] for r in manifest['files']),'bytes; source isolation and edited-document links pass')
