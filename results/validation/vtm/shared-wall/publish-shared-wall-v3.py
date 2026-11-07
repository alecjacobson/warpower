"""Keep full production mesh pairs public; retain pilot raw meshes privately."""
import hashlib,json,os
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';out=root/'results/validation/vtm/shared-wall';p=out/'manifest.json';d=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
keep=[];private=[]
for r in d['files']:
 public=root/r['path'];original=root/r['original'];assert sha(public)==r['sha256']
 if any('/'+folder+'/' in r['original'] for folder in ['local-dewall-vtm-v3','shared-wall-vtm-v1','shared-wall-broad-v1']) and public.suffix in ['.npz','.off','.gz']:
  assert original.is_file() and sha(original)==r['sha256'] and original.stat().st_nlink>=2
  private.append(dict(path=r['original'],bytes=r['bytes'],sha256=r['sha256'],former_staging_path=r['path']))
  public.unlink()  # Only the uncommitted publication alias; exact private evidence remains.
 else:keep.append(r)
assert private
report=dict(scope='Publication retains all complete production before/after mesh pairs, native traces, source, pilot timing/independent reports and failures. Earlier pilot raw meshes remain unchanged in the persistent private workspace; their hashes are recorded below. Only their uncommitted public hardlink aliases were removed. No previous published artifact or unique source/result was deleted.',raw_files=len(private),raw_bytes=sum(r['bytes'] for r in private))
record=w/'shared-wall-publication-selection.json';assert not record.exists();record.write_text(json.dumps(report,indent=2)+'\n')
d['files']=keep;d['retained_private_pilot_meshes']=private;d['publication_scope']=report['scope']
for src in [Path(__file__).resolve(),record]:
 dest=out/src.relative_to(w);assert not dest.exists();os.link(src,dest)
 d['files'].append(dict(original=str(src.relative_to(root)),path=str(dest.relative_to(root)),bytes=src.stat().st_size,sha256=sha(src)))
p.write_text(json.dumps(d,indent=2)+'\n')
for r in d['files']:assert sha(root/r['path'])==r['sha256']
for r in private:assert sha(root/r['path'])==r['sha256']
doc=root/'docs/shared-wall-reconstruction.md';text=doc.read_text();old='records source hashes, commands, full meshes, pilots and native traces.';assert old in text
text=text.replace(old,'records source hashes, commands, complete production mesh pairs, pilot summaries\nand native traces. Earlier pilot raw meshes remain private; their hashes are\nlisted in `retained_private_pilot_meshes`. The initial publication manifest\nincluded a still-open stdout log; its corrected closed snapshot and the original\nmanifest are retained. The final manifest is authoritative for published paths.')
doc.write_text(text)
print('Verified',len(d['files']),'public artifacts;',sum(r['bytes'] for r in d['files']),'bytes;',len(private),'private pilot raw meshes retained')
