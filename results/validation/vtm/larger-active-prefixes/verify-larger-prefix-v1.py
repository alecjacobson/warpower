"""Verify committed code, immutable evidence aliases, and completed qualification gates."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--expected-files',required=True,type=int);a=p.parse_args();root=Path.cwd();out=root/'results/validation/vtm/larger-active-prefixes'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((out/n).read_text())
m=read('manifest.json');assert m['runtime']=='670bd3b1a801e5397f55837d3caf685c20286656' and m['baseline']=='df74493c97633324147cb39ef4c41a7d34c7b90a'
assert len(m['files'])==a.expected_files and len({r['path'] for r in m['files']})==a.expected_files
for r in m['files']:
 p,q=root/r['path'],root/r['original'];assert sha(p)==sha(q)==r['sha256'];assert p.stat().st_ino==q.stat().st_ino and p.stat().st_mode&0o222==0
assert len(m['source_sha256'])==148
for name,digest in m['source_sha256'].items():
 assert sha(root/name)==digest
 assert hashlib.sha256(subprocess.check_output(['git','show',m['runtime']+':'+name])).hexdigest()==digest
refs=[*m['shared_helpers'],*m['shared_prior_evidence'],*m['input_fixtures']]
for r in refs:assert sha(root/r['path'])==r['sha256']
for name,digest in m['native_binary_sha256'].items():assert sha(root/'.work/vtm'/name)==digest
assert subprocess.check_output(['git','diff',m['runtime'],'--','warpower','tests','benchmarks','examples','README.md'])==b''
changed=subprocess.check_output(['git','diff','--name-only',m['baseline'],m['runtime']],text=True).splitlines()
assert set(changed)=={'warpower/_active_delaunay.py','warpower/_vtm.py','tests/test_larger_prefixes.py'}
audit=read('larger-prefix-production-source-audit-v2.json')
assert all(audit[k] for k in ['all_device_kernel_ASTs_unchanged','all_other_methods_unchanged','production_constructor_equals_measured_prototype_after_explicit_policy','vtm_change_is_only_setup_level_selection','readme_unchanged'])
assert len(audit['unchanged_files'])==141
assert read('larger-prefix-full-suite-v1.json')['tests']==1292
assert '47 passed' in (out/'larger-prefix-focused-v1.log').read_text()
primary=read('larger-prefix-final-v1/commands.json');assert len(primary)==34 and all(r['quota_clean'] and r['exit_code'] in [0,1] for r in primary)
report=read('larger-prefix-final-v1-independent.json');assert report['complete'] and len(report['cases'])==11 and len(report['standalone'])==5
normal=0;rejected=0
for name,r in report['cases'].items():
 assert all(r['work_equal'].values())
 if r['before_valid']==r['after_valid']==1:
  normal+=1;assert all(r['comparison'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
  assert r['independent_before']['full_hull']['valid'] and r['independent_after']['full_hull']['valid']
 else:
  rejected+=1;assert 'armadillo' in name and r['before_valid']==r['after_valid']==0 and all(r['failed_state_equal'].values())
assert normal==9 and rejected==2
assert read('larger-prefix-grid-v1-independent.json')['complete']
assert read('larger-prefix-pilot-v1-independent.json')['complete']
assert read('larger-prefix-current-reservation-comparison-v1.json')['complete']
mem=read('larger-prefix-memory-v1-commands.json');assert len(mem)==28 and all(r['exit_code']==0 and r['zero_memcheck_errors'] and r['expected_tests']==1 for r in mem)
qual=read('larger-prefix-qualification-v1.json');assert len(qual)==3 and all(r['exit_code']==0 for r in qual)
for kind in ['full','loop']:
 r=read(f'larger-prefix-{kind}-native-v1-audit.json');assert set(r['dot_node_types'])<={'KERNEL','MEMSET','MEMCPY','CONDITIONAL'} and set(r['dot_copy_kinds'])=={'DtoD (DEVICE to DEVICE)'}
 assert all(not any(k in name.lower() for k in ['alloc','free','memcpy']) for name in r['cuda_api_calls'])
 assert r['cuda_api_calls']['cudaGraphLaunch_v10000']==r['cuda_api_calls']['cuCtxSynchronize']==1
loop=read('larger-prefix-loop-native-v1.json');assert loop['stable_pointers'];assert [r[0] for r in loop['metadata']]==[0,0,1,1,0];assert [r[1] for r in loop['metadata']]==[1,1,0,0,1]
assert [r[2] for r in loop['metadata']]==[4096,8192,0,0,4096];assert all(not any(loop['metadata'][i][2:]) and not any(loop['volumes'][i]) for i in [2,3])
checks=read('larger-prefix-native-loop-independent-v1.json');assert checks['complete'] and all(checks['recovery_comparison'][k] for k in ['points_bitwise_equal','roles_equal','full_tets_equal','material_tets_equal','oriented_boundary_equal'])
links=[]
for name in ['architecture.md','vtm.md','active-reconstruction.md','extended-fragment-reuse.md','prepared-new-cell-queries.md','larger-active-prefixes.md']:
 p=root/'docs'/name;s=p.read_text();targets=re.findall(r'\[[^\]]*\]\(([^)]+)\)',s)+re.findall(r'^\[[^\]]+\]:\s+(\S+)',s,re.M)
 for target in targets:
  target=target.strip('<>')
  if target.startswith(('http://','https://','#','mailto:')):continue
  target=target.split('#',1)[0].split('?',1)[0];q=(p.parent/target).resolve();rel=str(q.relative_to(root))
  if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+rel],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
  links.append(dict(document=str(p.relative_to(root)),target=rel))
r=dict(runtime=m['runtime'],manifest_sha256=sha(out/'manifest.json'),artifacts=len(m['files']),source_files=148,shared_references=len(refs),local_links=len(links),all_hashes_aliases_match=True,measured_setup_equivalence_and_unchanged_kernels_verified=True,primary_timing_jobs=34,independently_valid_normal_meshes=18,retained_armadillo_rejection_pairs=2,full_suite_tests=1292,focused_tests=47,memory_jobs=28,native_audits_pass=True,loop_frames_independently_valid=3,exact_recovery=True,readme_unchanged=True,verification_source_sha256=sha(Path(__file__)),checked_links=links)
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(r,indent=2)+'\n');p.chmod(0o444);print({k:v for k,v in r.items() if k!='checked_links'})
