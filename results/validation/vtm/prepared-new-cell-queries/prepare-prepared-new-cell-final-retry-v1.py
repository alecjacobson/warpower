"""Retain the throttled batch; resume with three prior clean pairs and new output paths."""
from pathlib import Path
import gzip,hashlib,json
w=Path('.work/vtm');old=w/'prepared-new-cell-final-v1';rows=json.loads((old/'commands.json').read_text());assert len(rows)==8 and all(r['quota_clean'] for r in rows[:7]) and not rows[7]['quota_clean'];assert rows[7]['exit_code']==0
records=[]
for name in ['mesh.vtu','boundary.off']:
 p=old/'graded-torus16384-after'/name;data=p.read_bytes();q=p.with_suffix(p.suffix+'.gz');assert not q.exists();q.write_bytes(gzip.compress(data,mtime=0));assert gzip.decompress(q.read_bytes())==data;records.append(dict(original=str(p),path=str(q.absolute()),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),raw_sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)));p.unlink()
p=w/'prepared-new-cell-final-v1-throttled-archives.json';assert not p.exists();p.write_text(json.dumps(records,indent=2)+'\n')
s=(w/'run-prepared-new-cell-final-v1.py').read_text().replace("out=w/'prepared-new-cell-final-v1'", "out=w/'prepared-new-cell-final-v2'")
oldline='out.mkdir();rows=[];archives=[]'
new='''out.mkdir();rows=[];archives=[]
prior=json.loads((w/'prepared-new-cell-final-v1/commands.json').read_text())
assert len(prior)==8 and all(r['quota_clean'] for r in prior[:7]) and not prior[7]['quota_clean']
retained_names={'torus16384-fixed32','torus8192-progressive','reserved16384-target4096'}
aliases=[]
for ref in prior[:6]:
    assert ref['name'] in retained_names and ref['exit_code']==0 and ref['quota_clean']
    for suffix in ['.json','.npz','.log']:
        src=w/'prepared-new-cell-final-v1'/(ref['name']+'-'+ref['phase']+suffix);dst=out/src.name
        assert not dst.exists();os.link(src,dst)
        digest=hashlib.sha256(src.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==digest
        aliases.append(dict(original=str(src),alias=str(dst),sha256=digest))
    rows.append(dict(ref,reused_clean_pair=True,primary_result_alias=str(out/Path(ref['result']).name)))
(out/'retained-clean-aliases.json').write_text(json.dumps(aliases,indent=2)+'\\n')
(out/'commands.json').write_text(json.dumps(rows,indent=2)+'\\n')'''
assert oldline in s;s=s.replace(oldline,new)
s=s.replace('def run(name,phase,cmd,cwd):\n', 'def run(name,phase,cmd,cwd):\n    if name in retained_names:return\n')
p=w/'run-prepared-new-cell-final-v2.py';assert not p.exists();p.write_text(s)
