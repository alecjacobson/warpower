"""Close the first publisher's stdout snapshot before final manifest verification."""
import hashlib,json,os
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/shared-wall'
manifest=out/'manifest.json';original=manifest.read_bytes();d=json.loads(original)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bad=[r for r in d['files'] if sha(root/r['path'])!=r['sha256']]
assert len(bad)==1 and bad[0]['path']==str((out/'shared-wall-publication.log').relative_to(root))
assert bad[0]['bytes']==0
initial=w/'shared-wall-publication-initial-manifest.json';assert not initial.exists();initial.write_bytes(original)
row=bad[0];p=root/row['path'];row.update(bytes=p.stat().st_size,sha256=sha(p))
report=dict(scope='The initial publisher included its still-open redirected stdout log, then appended its completion message after hashing. Only that 13-byte log disagreed. The initial manifest is retained as historical evidence; the final manifest records the closed log and verifies every artifact. No meshing, timing or validation result changed.',initial_manifest_sha256=hashlib.sha256(original).hexdigest(),corrected_path=row['path'],corrected_sha256=row['sha256'])
record=w/'shared-wall-publication-repair.json';assert not record.exists();record.write_text(json.dumps(report,indent=2)+'\n')
for p in [Path(__file__).resolve(),initial,record]:
 dest=out/p.relative_to(w);assert not dest.exists();os.link(p,dest)
 d['files'].append(dict(original=str(p.relative_to(root)),path=str(dest.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
manifest.write_text(json.dumps(d,indent=2)+'\n')
for r in d['files']:assert sha(root/r['path'])==r['sha256'],r['path']
print('Verified',len(d['files']),'artifacts;',sum(r['bytes'] for r in d['files']),'bytes')
