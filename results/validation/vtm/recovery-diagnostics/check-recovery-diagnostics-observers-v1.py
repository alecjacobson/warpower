import json
from pathlib import Path
w=Path('.work/vtm')
helper=(w/'check-current-selection-final.py').read_text().split("out=w/'current-selection-final';commands=")[0]
exec(compile(helper,'current-selection-validation-helper','exec'))
rows={}
for p in sorted((w/'recovery-diagnostics-observers-v1').glob('*.json')):
 if p.name=='commands.json':continue
 d=json.loads(p.read_text())
 for r in d['replays']:
  mesh=r['changed_mesh']
  if mesh:
   rows[Path(mesh).name]=validate(Path(mesh),None)
   (w/'recovery-diagnostics-observers-v1-independent.json').write_text(json.dumps(rows,indent=2)+'\n');print(Path(mesh).name,rows[Path(mesh).name]['full_hull']['valid'],flush=True)
assert len(rows)==3
