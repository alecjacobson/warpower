"""Read-only source/evidence verification, plus a new immutable premerge record."""
import argparse,ast,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--expected-files',type=int,required=True);args=p.parse_args();root=Path.cwd();out=root/'results/validation/vtm/prepared-new-cell-queries';manifest=out/'manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((out/n).read_text())
m=json.loads(manifest.read_text());assert m['runtime']=='3b68a6043860209d3c507951b0e51309ed8bb8c5' and m['baseline']=='a4f7215fbf99249d11f765a4902c586daa606829';assert len(m['files'])==args.expected_files and len({r['path'] for r in m['files']})==args.expected_files
for r in m['files']:
 p,q=root/r['path'],root/r['original'];assert sha(p)==sha(q)==r['sha256'];assert p.stat().st_ino==q.stat().st_ino;assert p.stat().st_mode&0o222==0
for name,digest in m['source_sha256'].items():
 assert sha(root/name)==digest,name
 assert hashlib.sha256(subprocess.check_output(['git','show',m['runtime']+':'+name])).hexdigest()==digest,name
refs=[*m['shared_helpers'],*m['shared_prior_evidence'],*m['input_fixtures']]
for r in refs:assert sha(root/r['path'])==r['sha256'],r['path']
for name,digest in m['native_binary_sha256'].items():assert sha(root/'.work/vtm'/name)==digest
assert subprocess.check_output(['git','diff',m['runtime'],'--','warpower','tests','benchmarks','examples','README.md'])==b''
old=ast.parse((out/'prepared_new_cell_v1.py').read_text());new=ast.parse((root/'warpower/_unproved_queries.py').read_text())
for name in ['prepare_unproved','check_prepared_unproved']:
 a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name);b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name);assert ast.dump(a)==ast.dump(b)
def options(tree):
 return [ast.literal_eval(n.value.args[0]) for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='set_module_options']
assert options(old)==options(new)==[dict(enable_backward=False,fast_math=False,fuse_fp=False)]
links=[]
for name in ['architecture.md','delaunay-reuse.md','cached-exact-seed-queries.md','vtm.md','prepared-new-cell-queries.md']:
 p=root/'docs'/name;s=p.read_text();targets=re.findall(r'\[[^\]]*\]\(([^)]+)\)',s)+re.findall(r'^\[[^\]]+\]:\s+(\S+)',s,re.M)
 for target in targets:
  target=target.strip('<>')
  if target.startswith(('https://','http://','#','mailto:')):continue
  target=target.split('#',1)[0].split('?',1)[0];q=(p.parent/target).resolve();rel=str(q.relative_to(root))
  if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+rel],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  links.append(dict(document=str(p.relative_to(root)),target=rel))
for kind in ['full','loop']:
 a=read(f'prepared-new-cell-{kind}-native-v1-audit.json');assert set(a['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'};assert set(a['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert all(not any(w in k.lower() for w in ['alloc','free','memcpy']) for k in a['cuda_api_calls'])
 if kind=='loop':assert a['cuda_api_calls']['cudaGraphInstantiateWithFlags_v11040']==a['cuda_api_calls']['cudaGraphUpload_v10000']==1
 assert a['cuda_api_calls']['cudaGraphLaunch_v10000']==a['cuda_api_calls']['cuCtxSynchronize']==1
loop=read('prepared-new-cell-loop-native-v1.json');assert loop['stable_pointers'];assert [r[0] for r in loop['metadata']]==[0,0,1,1,0];assert [r[1] for r in loop['metadata']]==[1,1,0,0,1];assert all(not any(r[2:]) for r in loop['metadata'][2:4]);assert read('prepared-new-cell-native-loop-independent-v1.json')['complete']
mem=read('prepared-new-cell-memory-v1-commands.json');assert len(mem)==28 and all(x['exit_code']==0 and x['zero_memcheck_errors'] for x in mem)
primary=read('prepared-new-cell-final-v2/commands.json');assert len(primary)==26 and all(r['exit_code']==0 and r['quota_clean'] for r in primary);assert sum(r.get('reused_clean_pair',False) for r in primary)==6
prior=read('prepared-new-cell-final-v1/commands.json');assert len(prior)==8 and not prior[-1]['quota_clean'];assert read('prepared-new-cell-throttled-pair-independent-v1.json')['complete']
final=read('prepared-new-cell-final-v2-independent.json');assert final['complete'] and len(final['cases'])==7 and len(final['standalone'])==5;assert read('prepared-new-cell-grid-v1-independent.json')['complete']
hand=read('prepared-new-cell-hand-trace-v1/nonregression-conclusion-v1.json');assert hand['complete'] and hand['modified_modules_not_loaded'] and hand['loaded_module_sources_identical'] and hand['kernel_signature_identical'];assert hand['loaded_project_modules']==40 and hand['kernels_per_replay']==143
assert read('prepared-new-cell-hand-repeat-v1/comparison.json')['complete'];assert read('prepared-new-cell-hand-import-audit-v1/comparison.json')['complete']
r=dict(runtime=m['runtime'],manifest_sha256=sha(manifest),artifacts=len(m['files']),source_files=len(m['source_sha256']),shared_references=len(refs),local_links=len(links),all_hashes_and_aliases_match=True,measured_kernel_ASTs_and_compiler_options_equal=True,native_audits_pass=True,memory_jobs=28,memory_wall_seconds=sum(x['wall_seconds'] for x in mem),primary_timing_jobs=26,retained_clean_jobs=6,throttled_pair_retained_and_independently_valid=True,standalone_hand_all_samples_retained_and_path_independent=True,readme_unchanged=True,verification_source_sha256=sha(Path(__file__)),checked_links=links)
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');p.chmod(0o444);print({k:v for k,v in r.items() if k!='checked_links'})
