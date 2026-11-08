"""Verify immutable selective-history source, geometry, native and memory qualification."""
import ast,gzip,hashlib,json,re,subprocess
from pathlib import Path
root=Path.cwd();out=root/'results/validation/vtm/selective-step-history';w=root/'.work/vtm'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda n:json.loads((out/n).read_text())
manifest=read('manifest.json');source=read('selective-step-history-production-source-v1.json')
assert manifest['runtime']==source['runtime'] and manifest['baseline']==source['baseline']
for r in manifest['files']:
 a,b=root/r['original'],root/r['path'];assert sha(a)==sha(b)==r['sha256'] and a.stat().st_ino==b.stat().st_ino
 assert not b.stat().st_mode&0o222
 if b.suffix=='.gz':gzip.decompress(b.read_bytes())
for r in manifest['shared_references']:assert sha(root/r['path'])==r['sha256']
assert len(source['source_sha256'])==269 and source['unchanged_files']==265
assert source['changed_files']==['tests/test_selective_step_history.py','warpower/_mesher.py','warpower/_selective_step_history.py','warpower/_vtm.py']
for p,h in source['source_sha256'].items():assert sha(root/p)==h
assert not subprocess.check_output(['git','diff',source['runtime'],'--','warpower','tests','benchmarks','examples','README.md','pyproject.toml','LICENSE'])
node=lambda text,name:next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
body=lambda n:ast.dump(ast.Module(body=n.body,type_ignores=[]))
class Canonical(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in ['original_start_attempt','original_finish_attempt']:n.id=n.id.replace('original_','_',1)
  return n
old=subprocess.check_output(['git','show',source['baseline']+':warpower/_vtm.py'],text=True)
for name in ['start_attempt','finish_attempt']:
 assert body(node(old,name))==body(node((root/'warpower/_vtm.py').read_text(),'_'+name))
 assert body(Canonical().visit(node((out/'selective_step_history_v2.py').read_text(),name)))==body(node((root/'warpower/_selective_step_history.py').read_text(),name))
# Every executed source binding is checked against live immutable helpers,
# a pinned baseline/runtime blob, or the exactly recovered intermediate source.
checked=set()
def bindings(value):
 if isinstance(value,dict):
  hashes=value.get('source_sha256',{})
  if isinstance(hashes,dict):
   for name,h in hashes.items():
    if (name,h) in checked:continue
    p=Path(name);p=p if p.is_absolute() else root/p
    if p.exists() and sha(p)==h:checked.add((name,h));continue
    rel=p.relative_to(root)
    recovered=out/'selective-step-history-integration-v1-source'/rel
    if recovered.exists() and sha(recovered)==h:checked.add((name,h));continue
    assert str(rel).startswith(('warpower/','tests/','benchmarks/','examples/')),(name,h)
    blob=subprocess.check_output(['git','show',source['baseline']+':'+str(rel)])
    assert hashlib.sha256(blob).hexdigest()==h,(name,h)
    checked.add((name,h))
  for v in value.values():bindings(v)
 elif isinstance(value,list):
  for v in value:bindings(v)
for r in manifest['files']:
 p=root/r['path']
 if p.suffix=='.json':bindings(json.loads(p.read_text()))
for version,cpu,cuda in [(1,13,17),(2,17,21)]:
 for kind,count in [('cpu',cpu),('cuda',cuda)]:
  r=read(f'selective-step-history-{kind}-contracts-v{version}.json');assert r['exit_code']==0 and r['passed']==count
assert source['integrated_contracts']['passed']==60
tests=read('selective-step-history-production-tests-v1.json');assert len(tests)==2 and all(r['exit_code']==0 and r['passed'] for r in tests)
assert '1407 passed' in (out/'selective-step-history-production-full-suite-v1.log').read_text()
meshes=0;fresh=0
for folder,report,count in [('selective-step-history-pilot-v1','selective-step-history-pilot-v1-independent-v2.json',8),('selective-step-history-pilot-v2','selective-step-history-pilot-v2-independent-v3.json',8),('selective-step-history-gallery-v2','selective-step-history-gallery-v2-independent-v2.json',14),('selective-step-history-final-v1','selective-step-history-final-v1-independent.json',32)]:
 commands=read(folder+'/commands.json');assert len(commands)==count and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
 proof=read(report);assert proof['complete']
 for name,row in proof['cases'].items():
  for phase in ['before','after']:
   p=out/folder/(name+'-'+phase);p=p/'mesh.npz' if p.is_dir() else p.with_suffix('.npz')
   v=row['independent'][phase];assert sha(p)==v['mesh_sha256'] and v['full_hull']['valid'] and v['all_boundary_vertex_links_single_cycles'] and v['boundary_self_intersection_pairs']==0
   if p.name=='mesh.npz':assert v['vtu_round_trip_exact'] and v['boundary_off_round_trip_exact']
   if report=='selective-step-history-final-v1-independent.json':meshes+=1;fresh+=int(v.get('fresh_validation',False))
  assert row['timing']['additional_pool_bytes']==(32 if folder.endswith('pilot-v1') else 44)
assert meshes==22
proof=read('selective-step-history-final-v1-independent.json');assert len(proof['standalone'])==5
for r in proof['standalone'].values():assert all(r['unchanged'].values()) and r['relative_change']<.01
assert read('selective-step-history-pilot-v1-independent-v2.json')['cases']['armadillo-uniform8192-recovery']['timing']['reduction_fraction']<-.04
memory=read('selective-step-history-memory-v1-commands.json');assert len(memory)==4
for r in memory:
 assert r['exit_code']==0 and r['zero_memcheck_errors']
 log=(out/f"selective-step-history-memory-v1-{r['name']}-{r['mode']}.log").read_text()
 assert 'ERROR SUMMARY: 0 errors' in log and f"{r['expected_tests']} passed" in log
 assert r['expected_tests']==(37 if r['name']=='kernels' else 1)
qualified=read('selective-step-history-qualified-v1.json');assert len(qualified)==6 and all(r['exit_code']==0 for r in qualified)
native={}
for name in ['torus','uniform','loop']:
 r=read(f'selective-step-history-{name}-native-v1-audit.json');assert r['gpu_kernel_activities']>0
 assert set(r['dot_node_types'])<={'KERNEL','MEMCPY','MEMSET','CONDITIONAL','EMPTY','EVENT_RECORD','WAIT_EVENT'}
 assert all('DEVICE to DEVICE' in k for k in r['dot_copy_kinds']) and all('DTOD' in k for k in r['gpu_memcpy_kinds'])
 assert not any(any(x in k.lower() for x in ['memcpy','memalloc','memfree','hostfunc','callback']) for k in r['cuda_api_calls'])
 assert r['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
 assert sum(r['cuda_api_calls'].get(k,0) for k in ['cuCtxSynchronize','cuStreamSynchronize'])==1
 native[name]=r['gpu_kernel_activities']
for name in ['ordinary-loop','native-loop','padded-loop','lifetime-loop']:
 r=read('selective-step-history-'+name+'-comparison-v1.json');assert r['complete'] and r['rows_equal'] and r['stable_pointers']
 assert all(v['equal'] for f in r['frames'].values() for v in f['fields'].values())
assert read('selective-step-history-native-full-v1-comparison.json')['complete']
r=read('selective-step-history-native-torus-v1-comparison.json');assert r['complete'] and r['scheduler_work_equal'] and r['history_statistics'][0]>0 and r['history_statistics'][1]>0
features=read('selective-step-history-feature-fit-v1.json');assert features['complete']
for r in features['cases'].values():assert r['exact_corners_equal'] and r['chain_diagnostics_equal'] and abs(r['rms_difference'])<1e-12
p=root/'docs/selective-step-history.md'
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
 if target.startswith(('http://','https://','#')):continue
 path=(p.parent/target.split('#',1)[0]).resolve()
 if path==out/'premerge-integrity.json':continue
 if not path.exists():subprocess.run(['git','cat-file','-e','HEAD:'+str(path.relative_to(root))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
r=dict(scope=__doc__,artifacts=len(manifest['files']),manifest_sha256=sha(out/'manifest.json'),verifier_sha256=sha(Path(__file__)),source_files=269,unchanged_files=265,source_bindings=len(checked),production_tests=1407,final_meshes=meshes,fresh_final_validations=fresh,clean_final_jobs=32,memory_jobs=4,native_kernel_activities=native,complete=True)
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');p.chmod(0o444);print(r)
