"""Prepare release-bundle publication; raw evidence stays on persistent storage outside Git."""
import ast,hashlib,json
from pathlib import Path
w=Path('.work/vtm');changes=[]
def derive(old,new,transform):
 p=w/old;q=w/new;assert not q.exists();s=transform(p.read_text());ast.parse(s);q.write_text(s)
 changes.append(dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),target=str(q),target_sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
def publisher(s):
 s=s.replace('import hashlib,json,os,subprocess','import gzip,hashlib,json,os,subprocess,tarfile')
 s=s.replace("out=root/'results/validation/vtm/selective-step-history'", "out=root/'results/local/selective-step-history'")
 s=s.replace("manifest=out/'manifest.json'", "manifest=out/'manifest.json'") # original uses p below
 needle="p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\\n');p.chmod(0o444)"
 assert needle in s
 insert='''# This archive is reproducible packaging: every unique source/evidence file
# stays under persistent /home/horde. Only the disposable archive copy is cached.
tag='evidence-selective-step-history-'+s['runtime'][:7]
archive_name='selective-step-history-evidence.tar.gz'
cache=Path('/opt/warpower-generated-cache/release-packaging')/tag
cache.mkdir(parents=True,exist_ok=True);archive=cache/archive_name;assert not archive.exists()
with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w|',format=tarfile.PAX_FORMAT,dereference=False) as tar:
    for row in meta['files']:
        for name in [row['original'],row['path']]:
            path=root/name;info=tar.gettarinfo(str(path),arcname=name)
            info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o444
            if info.isfile():
                with path.open('rb') as f:tar.addfile(info,f)
            else:
                assert info.islnk() and not Path(info.linkname).is_absolute()
                tar.addfile(info)
meta['release_bundle']=dict(tag=tag,name=archive_name,sha256=sha(archive),bytes=archive.stat().st_size,
    url='https://github.com/alecjacobson/warpower/releases/download/'+tag+'/'+archive_name,
    packaging_cache=str(archive),unique_evidence_remains_persistent=True)
meta['storage_policy']='Only this manifest, verification result and compact summary are committed. Raw files are downloadable release artifacts, materialized under ignored results/local and .work paths.'
p=out/'manifest.json';p.write_text(json.dumps(meta,indent=2)+'\\n');p.chmod(0o444)
public=root/'results/validation/vtm/selective-step-history';assert not public.exists();public.mkdir(parents=True)
os.link(p,public/'manifest.json')
'''
 return s.replace(needle,insert)
derive('publish-selective-step-history-v1.py','publish-selective-step-history-v2.py',publisher)
def verifier(s):
 s=s.replace('import ast,gzip,hashlib,json,re,subprocess','import ast,gzip,hashlib,json,os,re,subprocess,tarfile')
 s=s.replace("out=root/'results/validation/vtm/selective-step-history'", "out=root/'results/local/selective-step-history'")
 needle="for r in manifest['shared_references']:assert sha(root/r['path'])==r['sha256']"
 assert needle in s
 s=s.replace(needle,needle+'''
bundle=manifest['release_bundle'];archive=Path(bundle['packaging_cache'])
assert sha(archive)==bundle['sha256'] and archive.stat().st_size==bundle['bytes']
expected={name:row['sha256'] for row in manifest['files'] for name in [row['original'],row['path']]}
seen=set()
with tarfile.open(archive,'r:gz') as tar:
    for member in tar:
        assert member.name in expected and member.name not in seen
        assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
        if member.islnk():
            assert member.linkname in seen and expected[member.linkname]==expected[member.name]
        else:
            assert member.isfile()
            f=tar.extractfile(member);digest=hashlib.sha256()
            for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
            assert digest.hexdigest()==expected[member.name]
        seen.add(member.name)
assert seen==set(expected)
''')
 s=s.replace("if path==out/'premerge-integrity.json':continue", "if path==root/'results/validation/vtm/selective-step-history/premerge-integrity.json':continue")
 needle="p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\\n');p.chmod(0o444);print(r)"
 assert needle in s
 return s.replace(needle,"r.update(release_bundle_sha256=bundle['sha256'],release_bundle_members=len(seen),raw_artifacts_in_git=False)\n"+needle+"\nos.link(p,root/'results/validation/vtm/selective-step-history/premerge-integrity.json')\n")
derive('verify-selective-step-history-v1.py','verify-selective-step-history-v2.py',verifier)
def doc(s):
 s=s.replace('Reproduce using the pinned revisions and the commands/environment in the','Raw evidence is distributed as a release bundle, keeping binary snapshots and\ntraces out of ordinary Git. The manifest records its download URL and SHA-256.\nThe bundle restores the original `.work/vtm` files and ignored `results/local`\naliases; extract it from the repository root after verifying the checksum.\n\nReproduce using the pinned revisions and the commands/environment in the')
 return s
derive('publish-selective-step-history-doc-v1.py','publish-selective-step-history-doc-v2.py',doc)
p=w/'selective-step-history-release-preparation-v1.json';assert not p.exists();p.write_text(json.dumps(dict(scope=__doc__,files=changes,raw_publication_v1_not_executed=True),indent=2)+'\n')
print('Prepared release-only raw publication; no archive created or uploaded.')
