"""Verify published recovery evidence, immutable aliases and production qualification."""
import argparse,ast,gzip,hashlib,json,re,subprocess
from pathlib import Path
root=Path.cwd();out=root/'results/validation/vtm/tangential-topology-recovery'
p=argparse.ArgumentParser();p.add_argument('--expected-files',type=int,required=True);args=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((out/n).read_text())
m=read('manifest.json');assert len(m['files'])==args.expected_files
for r in m['files']:
 p,q=root/r['path'],root/r['original']
 assert sha(p)==sha(q)==r['sha256'] and p.stat().st_ino==q.stat().st_ino and p.stat().st_mode&0o222==0
for r in m['shared_references']:assert sha(root/r['path'])==r['sha256']
for n,h in m['runtime_source_sha256'].items():
 assert sha(root/n)==h
 assert hashlib.sha256(subprocess.check_output(['git','show',m['runtime']+':'+n])).hexdigest()==h
assert not subprocess.check_output(['git','diff',m['runtime'],'--','warpower','tests','benchmarks','examples','README.md','pyproject.toml'])
s=read('tangential-production-source-audit-v1.json');assert len(s['unchanged_files'])==255 and len(s['changed_files'])==6
assert s['readme_unchanged'] and s['independent_diagram_and_delaunay_sources_unchanged']
r=read('tangential-production-final-v1-independent.json');assert r['complete'] and len(r['cases'])==9 and len(r['standalone'])==5
for name,c in r['cases'].items():
 assert c['after_valid']==1 and c['independent_after']['full_hull']['valid'] and c['independent_after']['boundary_self_intersection_pairs']==0
 if c['before_valid']:
  assert all(c['work_equal'].values()) and all(c['input_arrays_equal'].values())
  assert c['independent_before']['full_hull']['valid'] and c['independent_before']['boundary_self_intersection_pairs']==0
  d=c['comparison']
 else:
  assert c['baseline_failed_zero_output'] and not c['successful_mesh_speedup_claim']
  assert all(c['private_prototype_work_equal'].values());d=c['private_prototype_comparison']
 assert all(d[k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
 assert d['relative_coordinate_difference']<1e-9
 if 'maximum_radius_ratio_difference' in d:assert d['maximum_radius_ratio_difference']<1e-7 and abs(d['minimum_radius_ratio_before']-d['minimum_radius_ratio_after'])<1e-12
for c in r['standalone'].values():assert all(c['unchanged'].values())
assert sum(c['before_valid']==0 for c in r['cases'].values())==2
commands=read('tangential-production-final-v1/commands.json');assert len(commands)==28 and all(d['quota_clean'] and d['exit_code'] in [0,1] for d in commands)
assert '1326 passed' in (out/'tangential-production-full-suite-v1.log').read_text()
checks=read('tangential-production-memory-v3-commands.json');assert len(checks)==10 and all(d['exit_code']==0 and d['zero_memcheck_errors'] for d in checks)
assert read('tangential-production-memory-v3-comparison.json')['complete']
assert {(d['name'],d['mode']) for d in checks}=={(name,mode) for name in ['kernels','policy-healthy','policy-disabled','policy-implied','loop'] for mode in ['padded','lifetime']}
isolation=read('tangential-production-policy-isolation-v1.json')
original=root/isolation['source'];generated=out/'test_tangential_policy_isolated_v1.py'
assert sha(original)==isolation['source_sha256']==m['runtime_source_sha256'][isolation['source']]
assert sha(generated)==isolation['generated_sha256']
policy=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='test_recovery_policy_scopes_storage_and_healthy_solve_skips_search')
generated_functions={n.name:n for n in ast.parse(generated.read_text()).body if isinstance(n,ast.FunctionDef)}
indices=[]
for part in isolation['groups'].values():
 body=generated_functions[part['function']].body
 expected=[policy.body[0]]+[policy.body[i] for i in part['original_statement_indices']]
 assert [ast.dump(n) for n in body]==[ast.dump(n) for n in expected]
 indices+=part['original_statement_indices']
assert sorted(indices)==list(range(1,len(policy.body)))
for name in ['tangential-production-loop-v2-independent.json','tangential-production-loop-native-v1-independent.json']:
 d=read(name);assert d['complete'] and len(d['frames'])==2
 assert all(d['recovery'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
 assert d['exhaustion']['full_hull']['valid'] and d['exhaustion']['points_restored_bitwise'] and d['exhaustion']['full_cells_restored'] and d['exhaustion']['public_zero_counts']
 for c in d['frames'].values():assert c['full_hull']['valid'] and c['boundary_self_intersection_pairs']==0
for stem in ['tangential-production-uniform-native-v1','tangential-production-loop-native-v1']:
 a=read(stem+'-audit.json');calls=a['cuda_api_calls']
 assert set(a['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'} and set(a['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert calls['cudaGraphLaunch_v10000']==1 and sum(v for k,v in calls.items() if 'Synchronize' in k)==1
 assert not any(any(t in k.lower() for t in ['alloc','free','memcpy']) for k in calls)
 k=read(stem+'-kernels.json');p=out/(stem+'.nsys-rep.gz')
 assert sha(p)==k['trace_gz_sha256'] and hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==k['trace_raw_sha256']
 assert hashlib.sha256(gzip.decompress((out/(stem+'.dot.gz')).read_bytes())).hexdigest()==a['dot_sha256']
nf=read('tangential-production-native-full-v1-independent.json');assert nf['complete'] and nf['independent']['full_hull']['valid'] and nf['independent']['boundary_self_intersection_pairs']==0
for name in ['tangential-production-loop-v2/result.json','tangential-production-loop-native-v1-frames/result.json','tangential-production-memory-v3-loop-padded-frames/result.json','tangential-production-memory-v3-loop-lifetime-frames/result.json']:
 for path,digest in read(name)['production_source_sha256'].items():assert m['runtime_source_sha256'][path]==digest
refs=read('tangential-production-reference-v1.json');assert refs['complete'] and len(refs['frames'])==4
for f in refs['frames']:assert f['retained_original_references'] and f['all_roles_equal'] and f['relative_displacement']<=.2
links=[]
for name in ['tangential-topology-recovery.md','mesher.md','architecture.md','exterior-proof-hierarchy.md','vtm.md','examples.md']:
 p=root/'docs'/name
 for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  if target.startswith(('http://','https://','#')):continue
  q=(p.parent/target.split('#',1)[0]).resolve();rel=str(q.relative_to(root))
  if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+rel],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  links.append(dict(document=name,target=rel))
result=dict(manifest_sha256=sha(out/'manifest.json'),artifact_hashes_aliases_readonly_verified=args.expected_files,runtime_files_verified=len(m['runtime_source_sha256']),standalone_paths_unchanged=True,readme_unchanged=True,validated_meshes=16,new_valid_cases=2,native_loop_validated_frames=2,exhaustion_full_hull_and_restore_valid=True,memory_checks=10,full_tests=1326,links=links,verification_source_sha256=sha(Path(__file__)))
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');p.chmod(0o444);print({k:v for k,v in result.items() if k!='links'})
