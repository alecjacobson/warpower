"""Read-only verification of published evidence, current sources and local report links."""
import hashlib,json,re,subprocess
from pathlib import Path
root=Path.cwd();out=root/'results/validation/vtm/cached-exact-seed-queries';manifest=out/'manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads(manifest.read_text());assert m['runtime']=='17c63614d124ff7874b930bb9b95c568c9331cec'
assert len(m['files'])==345
for r in m['files']:
 p,q=root/r['path'],root/r['original'];assert sha(p)==sha(q)==r['sha256'];assert p.stat().st_ino==q.stat().st_ino;assert p.stat().st_mode&0o222==0
for name,digest in m['source_sha256'].items():assert sha(root/name)==digest,name
refs=[*m['shared_helpers'],*m['shared_prior_evidence'],*m['input_fixtures']]
for r in refs:assert sha(root/r['path'])==r['sha256'],r['path']
assert subprocess.check_output(['git','diff',m['runtime'],'--','warpower','tests','benchmarks','examples','README.md'])==b''
links=[]
for name in ['architecture.md','delaunay-reuse.md','hierarchical-seed-queries.md','vtm.md','cached-exact-seed-queries.md']:
 p=root/'docs'/name;s=p.read_text();targets=re.findall(r'\[[^\]]*\]\(([^)]+)\)',s)+re.findall(r'^\[[^\]]+\]:\s+(\S+)',s,re.M)
 for target in targets:
  target=target.strip('<>')
  if target.startswith(('https://','http://','#','mailto:')):continue
  target=target.split('#',1)[0].split('?',1)[0];q=(p.parent/target).resolve();rel=str(q.relative_to(root))
  if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+rel],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  links.append(dict(document=str(p.relative_to(root)),target=rel))
for kind in ['full','loop']:
 a=json.loads((out/f'hierarchical-cache-{kind}-native-v1-audit.json').read_text());assert set(a['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'};assert set(a['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert all(not any(w in k.lower() for w in ['alloc','free','memcpy']) for k in a['cuda_api_calls'])
 if kind=='loop':assert a['cuda_api_calls']['cudaGraphInstantiateWithFlags_v11040']==a['cuda_api_calls']['cudaGraphUpload_v10000']==1
 assert a['cuda_api_calls']['cudaGraphLaunch_v10000']==a['cuda_api_calls']['cuCtxSynchronize']==1
mem=json.loads((out/'hierarchical-cache-memory-v1-commands.json').read_text());assert len(mem)==28 and all(x['exit_code']==0 and x['zero_memcheck_errors'] for x in mem)
r=dict(runtime=m['runtime'],manifest_sha256=sha(manifest),artifacts=len(m['files']),source_files=len(m['source_sha256']),shared_references=len(refs),local_links=len(links),all_hashes_and_aliases_match=True,native_audits_pass=True,memory_jobs=28,memory_wall_seconds=sum(x['wall_seconds'] for x in mem),readme_unchanged=True,verification_source_sha256=sha(Path(__file__)),checked_links=links)
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');p.chmod(0o444)
print({k:v for k,v in r.items() if k!='checked_links'})
