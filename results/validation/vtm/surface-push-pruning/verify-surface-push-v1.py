"""Verify immutable publication, exact source change, full qualification and frame bindings."""
import ast
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

root=Path.cwd(); out=root/'results/validation/vtm/surface-push-pruning'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name): return json.loads((out/name).read_text())
manifest=read('manifest.json'); source=read('surface-push-production-source-v1.json')
def source_bindings(record):
    revision=record.get('runtime',record.get('baseline',record.get('source',manifest['baseline'])))
    for name,digest in record.get('source_sha256',{}).items():
        path=Path(name); path=path if path.is_absolute() else root/path
        if path.exists() and sha(path)==digest: continue
        relative=str(path.relative_to(root)); assert relative.startswith(('warpower/','tests/','benchmarks/','examples/')),(name,revision)
        content=subprocess.check_output(['git','show',revision+':'+relative])
        assert hashlib.sha256(content).hexdigest()==digest,(name,revision)
assert manifest['runtime']==source['runtime'] and manifest['baseline']==source['baseline']
for row in manifest['files']:
    a,b=root/row['original'],root/row['path']
    assert sha(a)==sha(b)==row['sha256'] and a.stat().st_ino==b.stat().st_ino
    assert not b.stat().st_mode&0o222
    if b.suffix=='.gz': gzip.decompress(b.read_bytes())
for row in manifest['shared_references']: assert sha(root/row['path'])==row['sha256']
assert set(source['changed'])=={'warpower/_surface_bvh.py'} and source['README_unchanged'] and source['all_public_standalone_implementations_unchanged']
assert source['source_files']==267 and len(source['unchanged'])==266
for p,h in source['source_sha256'].items(): assert sha(root/p)==h
assert not subprocess.check_output(['git','diff',source['runtime'],'--','warpower','tests','benchmarks','examples','README.md','pyproject.toml','LICENSE'])
def function(text,name): return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
a=function((root/'warpower/_surface_bvh.py').read_text(),'closest_point')
b=function((out/'surface_push_pruning_v1.py').read_text(),'pruned_closest_point'); b.name='closest_point'
assert ast.dump(a)==ast.dump(b)
for row in read('surface-push-qualification-preparation-v1.json')['files']:
    assert sha(Path(row['source']))==row['source_sha256']
    assert sha(Path(row['target']))==row['target_sha256']
for label,count in [('surface-push-pruning-contracts-v1',145),('surface-push-pruning-cache-contracts-v1',29)]:
    r=read(label+'.json'); assert r['complete'] and r['passed'] and r['exit_code']==0
    source_bindings(r); assert f'{count} passed' in (out/(label+'.log')).read_text()
tests=read('surface-push-production-tests-v1.json'); assert len(tests)==2 and all(r['passed'] and r['exit_code']==0 for r in tests)
assert '1387 passed' in (out/'surface-push-production-full-suite-v1.log').read_text()
static=read('surface-push-pruning-static-summary-v1.json'); assert static['complete'] and static['rows']==64
for name,c in static['cases'].items():
    reports=[read('surface-push-pruning-static-v1/'+name+'-'+phase+'.json') for phase in ['before','after']]
    for r in reports:
        assert r['complete'] and len(r['rows'])==8 and all(row['quota_clean'] for row in r['rows'])
        source_bindings(r); assert sha(Path(r['input']))==r['input_sha256']
    for a,b in zip(reports[0]['rows'],reports[1]['rows']):
        assert a['name']==b['name']
        if a['name']=='quadrature':
            assert a['output_sha256']['maximum']==b['output_sha256']['maximum'] and a['values']['error']==b['values']['error']==0
            assert abs(a['values']['integral']-b['values']['integral'])<=2e-12*abs(a['values']['integral'])
        else: assert a['output_sha256']==b['output_sha256']
pilot=read('surface-push-pruning-pilot-v1-independent.json'); assert pilot['complete'] and pilot['strict_trajectory_parity'] and pilot['exact_geometry_transfers']==8
commands=read('surface-push-final-v1/commands.json'); assert len(commands)==32 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
proof=read('surface-push-final-v1-independent.json'); assert proof['complete'] and len(proof['cases'])==11 and len(proof['standalone'])==5
review=read('surface-push-variation-review-v1.json'); assert review['complete'] and review['proof_sha256']==sha(out/'surface-push-final-v1-independent.json')
assert review['source_sha256']==sha(out/'review-surface-push-variation-v1.py')
for name,c in proof['cases'].items():
    assert all(c['work_equal'].values()) and c['timing']['additional_pool_bytes']==0
    for phase in ['before','after']:
        p=out/'surface-push-final-v1'/(name+'-'+phase); p=p/'mesh.npz' if p.is_dir() else p.with_suffix('.npz')
        v=c['independent'][phase]; assert sha(p)==v['mesh_sha256'] and v['full_hull']['valid'] and v['boundary_self_intersection_pairs']==0 and v['all_boundary_vertex_links_single_cycles']
        if p.name=='mesh.npz': assert v['vtu_round_trip_exact'] and v['boundary_off_round_trip_exact']
for c in proof['standalone'].values(): assert all(c['unchanged'].values()) and c['relative_change']<.01
memory=read('surface-push-memory-v1-commands.json'); assert len(memory)==4 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
for r in memory:
    text=(out/f"surface-push-memory-v1-{r['name']}-{r['mode']}.log").read_text()
    assert 'ERROR SUMMARY: 0 errors' in text and f"{r['expected_tests']} passed" in text
    assert r['expected_tests']==(116 if r['name']=='kernels' else 1)
qualified=read('surface-push-qualified-v1.json'); assert len(qualified)==5 and all(r['exit_code']==0 for r in qualified)
native={}
for prefix in ['uniform','loop']:
    r=read(f'surface-push-{prefix}-native-v1-audit.json'); assert r['gpu_kernel_activities']>0
    assert set(r['dot_node_types'])<={'KERNEL','MEMCPY','MEMSET','CONDITIONAL','EMPTY','EVENT_RECORD','WAIT_EVENT'}
    assert all('DEVICE to DEVICE' in k for k in r['dot_copy_kinds']) and all('DTOD' in k for k in r['gpu_memcpy_kinds'])
    assert not any(any(x in k.lower() for x in ['memcpy','memalloc','memfree','hostfunc','callback']) for k in r['cuda_api_calls'])
    assert r['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
    assert sum(r['cuda_api_calls'].get(k,0) for k in ['cuCtxSynchronize','cuStreamSynchronize'])==1
    native[prefix]=r
for name in ['ordinary-loop','native-loop','padded-loop','lifetime-loop']:
    r=read('surface-push-'+name+'-comparison-v1.json'); assert r['complete'] and r['rows_equal'] and r['stable_pointers']
    assert all(v['equal'] for f in r['frames'].values() for v in f['fields'].values())
assert read('surface-push-native-full-v1-comparison.json')['complete']
source_bindings(read('surface-push-uniform-native-v1.json'))
cost=read('surface-push-native-cost-review-v1.json'); assert cost['complete'] and len(cost['rows'])==5
assert cost['source_sha256']==sha(out/'review-surface-push-native-cost-v1.py')
for path,digest in cost['inputs'].items(): assert sha(Path(path))==digest
assert all(row['before']['calls']==row['after']['calls'] for row in cost['rows'])
for name in ['surface-push-pruning.md','surface-bound-reuse.md']:
    p=root/'docs'/name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('http://','https://','#')): continue
        q=(p.parent/target.split('#',1)[0]).resolve()
        if not q.exists(): subprocess.run(['git','cat-file','-e','HEAD:'+str(q.relative_to(root))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
result=dict(scope=__doc__,artifacts=len(manifest['files']),manifest_sha256=sha(out/'manifest.json'),production_tests=1387,private_tests=174,clean_static_rows=64,clean_complete_and_standalone_jobs=32,independent_meshes=22,exact_geometry_transfers=proof['exact_geometry_transfers'],strict_trajectory_parity=proof['strict_trajectory_parity'],memory_jobs=4,native_kernel_activities={k:v['gpu_kernel_activities'] for k,v in native.items()},source_files=267,runtime_changes=list(source['changed']),verifier_sha256=sha(Path(__file__)))
p=out/'premerge-integrity.json'; assert not p.exists(); p.write_text(json.dumps(result,indent=2)+'\n'); p.chmod(0o444); print(result)
