"""Verify published source/evidence hashes, archive round trips and local doc links."""
import gzip,hashlib,json,re,subprocess
from pathlib import Path
root=Path.cwd();out=root/'results/validation/vtm/hierarchical-seed-queries';p=out/'manifest.json';d=json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for row in d['files']:
 a,b=root/row['path'],root/row['original'];assert sha(a)==sha(b)==row['sha256'];assert a.stat().st_ino==b.stat().st_ino;assert a.stat().st_mode&0o222==0
for name,value in d['source_sha256'].items():assert sha(root/name)==value
for row in d['shared_helpers']+d['shared_prior_evidence']:assert sha(root/row['path'])==row['sha256']
assert not subprocess.check_output(['git','diff',d['runtime'],'--','warpower','tests','benchmarks','examples','README.md'])
links=[]
for name in ['hierarchical-seed-queries.md','architecture.md','delaunay-reuse.md','vtm.md','extended-fragment-reuse.md']:
 doc=root/'docs'/name
 for match in re.findall(r'\[[^\]]*\]\(([^)]+)\)',doc.read_text()):
  if '://' in match or match.startswith('#'):continue
  target=(doc.parent/match.split('#')[0]).resolve();rel=str(target.relative_to(root))
  if not target.exists():subprocess.run(['git','cat-file','-e','HEAD:'+rel],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  links.append(dict(document=str(doc.relative_to(root)),target=rel))
# The original uncompressed native report is retained losslessly in gzip.
for label in ['profile','full-native','loop-native']:
 stem=out/f'large-stackless-{label}-v1';r=json.loads(stem.with_name(stem.name+'-kernels.json').read_text());packed=stem.with_suffix('.nsys-rep.gz')
 assert sha(packed)==r['trace_gz_sha256'];assert hashlib.sha256(gzip.decompress(packed.read_bytes())).hexdigest()==r['trace_raw_sha256']
 dot=stem.with_suffix('.dot.gz');audit=json.loads(stem.with_name(stem.name+'-audit.json').read_text());assert hashlib.sha256(gzip.decompress(dot.read_bytes())).hexdigest()==audit['dot_sha256']
checks=json.loads((out/'large-stackless-final-v1-independent.json').read_text());assert checks['complete']
q=json.loads((out/'large-stackless-qualification-v1.json').read_text());assert len(q)==3 and all(x['exit_code']==0 for x in q)
result=dict(manifest_sha256=sha(p),artifact_count=len(d['files']),source_count=len(d['source_sha256']),shared_references=len(d['shared_helpers'])+len(d['shared_prior_evidence']),all_hashes_inodes_readonly_archive_and_source_checks_pass=True,doc_links=links,runtime=d['runtime'],head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
dest=out/'premerge-integrity.json';assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');dest.chmod(0o444)
print({k:v for k,v in result.items() if k!='doc_links'});print('Verified local doc links:',len(links))
