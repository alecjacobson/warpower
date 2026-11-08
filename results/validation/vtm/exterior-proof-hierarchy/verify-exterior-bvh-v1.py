"""Verify artifact integrity, runtime scope and all proof-acceleration gates."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
root=Path.cwd();out=root/'results/validation/vtm/exterior-proof-hierarchy'
p=argparse.ArgumentParser();p.add_argument('--expected-files',type=int,required=True);args=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((out/n).read_text())
m=read('manifest.json');assert len(m['files'])==args.expected_files
for r in m['files']:
 p,q=root/r['path'],root/r['original']
 assert sha(p)==sha(q)==r['sha256'] and p.stat().st_ino==q.stat().st_ino and p.stat().st_mode&0o222==0
for r in m['shared_references']:assert sha(root/r['path'])==r['sha256']
for name,digest in m['runtime_source_sha256'].items():
 assert sha(root/name)==digest
 assert hashlib.sha256(subprocess.check_output(['git','show',m['runtime']+':'+name])).hexdigest()==digest
assert not subprocess.check_output(['git','diff',m['runtime'],'--','warpower','tests','benchmarks','examples','README.md'])
s=read('exterior-bvh-source-comparison-v1.json')
assert set(s['changed'])=={'warpower/_proven_exterior.py'} and set(s['added'])=={'warpower/_exterior_proof_bvh.py','tests/test_exterior_proof_bvh.py'}
assert s['unchanged_count']==255 and s['readme_unchanged']
r=read('exterior-bvh-final-v3-independent.json');assert r['complete'] and len(r['cases'])==9 and len(r['standalone'])==5
for name,c in r['cases'].items():
 assert c['before_valid']==c['after_valid']==1 and all(c['work_equal'].values())
 assert all(c['comparison'][k] for k in ['roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
 assert all(c['input_arrays_equal'].values())
 d=c['coordinate_details'];assert c['comparison']['points_bitwise_equal']==(d['changed_coordinates']==0)
 assert d['relative_coordinate_difference']<1e-9 and d['maximum_radius_ratio_difference']<1e-7
 assert d['minimum_radius_ratio_before']==d['minimum_radius_ratio_after']
 if name not in ['thin-hand-seed7','cavity','disconnected']:assert d['changed_coordinates']==0
 for k in ['independent_after','independent_before']:
  assert c[k]['full_hull']['valid'] and c[k]['boundary_self_intersection_pairs']==0
for c in r['standalone'].values():assert all(c['unchanged'].values())
assert len(read('exterior-bvh-final-v2/commands.json'))==28
assert '1317 passed' in (out/'exterior-bvh-full-suite-v1.log').read_text()
checks=read('exterior-bvh-memory-v1-commands.json');assert len(checks)==8 and all(x['exit_code']==0 and x['zero_memcheck_errors'] for x in checks)
loop=read('exterior-bvh-native-loop-independent-v1.json');assert loop['complete'] and len(loop['frames'])==3
assert all(loop['recovery_comparison'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
for c in loop['frames'].values():assert c['full_hull']['valid'] and c['boundary_self_intersection_pairs']==0
for stem in ['exterior-bvh-uniform-native-v1','exterior-bvh-loop-native-v1']:
 a=read(stem+'-audit.json');assert set(a['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'}
 assert set(a['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert a['cuda_api_calls']['cudaGraphLaunch_v10000']==a['cuda_api_calls']['cuCtxSynchronize']==1
 assert not any(any(t in k.lower() for t in ['alloc','free','memcpy']) for k in a['cuda_api_calls'])
links=[]
for name in ['exterior-proof-hierarchy.md','proven-exterior-recovery.md','architecture.md']:
 p=root/'docs'/name
 for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
  if target.startswith(('http://','https://','#')):continue
  q=(p.parent/target.split('#',1)[0]).resolve();rel=str(q.relative_to(root))
  if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+rel],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  links.append(dict(document=name,target=rel))
result=dict(manifest_sha256=sha(out/'manifest.json'),artifact_hashes_aliases_readonly_verified=args.expected_files,runtime_files_verified=len(m['runtime_source_sha256']),standalone_paths_unchanged=True,readme_unchanged=True,validated_pairs=9,native_loop_validated_frames=3,memory_checks=8,full_tests=1317,links=links,verification_source_sha256=sha(Path(__file__)))
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');p.chmod(0o444);print({k:v for k,v in result.items() if k!='links'})
