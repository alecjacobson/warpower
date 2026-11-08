"""Verify publication bytes, source scope, qualification results and graph/frame bindings."""
import ast
import copy
import gzip
import hashlib
import json
import re
import subprocess
from pathlib import Path

root=Path.cwd();out=root/'results/validation/vtm/surface-sample-quadrature'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):return json.loads((out/name).read_text())
manifest=read('manifest.json');source=read('surface-sample-production-source-v2.json')

def source_bindings(record):
    revision=record.get('runtime',record.get('baseline',manifest['baseline']))
    for name,digest in record.get('source_sha256',{}).items():
        path=Path(name);path=path if path.is_absolute() else root/path
        if path.exists() and sha(path)==digest:continue
        relative=str(path.relative_to(root))
        assert relative.startswith(('warpower/','tests/','benchmarks/','examples/')),(name,revision)
        content=subprocess.check_output(['git','show',revision+':'+relative])
        assert hashlib.sha256(content).hexdigest()==digest,(name,revision)

assert manifest['runtime']==source['runtime'] and manifest['baseline']==source['baseline']
for row in manifest['files']:
    a,b=root/row['original'],root/row['path']
    assert sha(a)==sha(b)==row['sha256'] and a.stat().st_ino==b.stat().st_ino
    assert not b.stat().st_mode&0o222
    if b.suffix=='.gz':gzip.decompress(b.read_bytes())
for row in manifest['shared_references']:assert sha(root/row['path'])==row['sha256']
for p,h in source['source_sha256'].items():assert sha(root/p)==h
assert set(source['changed_files'])=={'warpower/_extraction.py','tests/surface_quadrature_reference.py','tests/test_surface_quadrature.py'}
assert source['production_kernel_matches_private'] and source['reference_matches_original'] and source['readme_unchanged']
assert not subprocess.check_output(['git','diff',source['runtime'],'--','warpower','tests','benchmarks','examples','README.md','pyproject.toml'])
tests=read('surface-sample-production-tests-v2.json');assert len(tests)==2 and all(r['passed'] and r['exit_code']==0 for r in tests)
assert '1387 passed' in (out/'surface-sample-production-full-suite-v2.log').read_text()
contract=read('surface-sample-parallel-contracts-v1.json');assert contract['complete'] and contract['passed']
source_bindings(contract)
source_bindings(read('surface-sample-production-source-v1.json'))
source_bindings(source)
assert '99 passed' in (out/'surface-sample-parallel-contracts-v1.log').read_text()
static=[]
for name in ['torus1000','torus16384','armadillo-uniform8192','armadillo-graded8192']:
    r=read('surface-sample-static-v1/'+name+'.json');assert r['complete'] and len(r['rows'])==4
    source_bindings(r)
    assert sha(Path(r['input']))==r['input_sha256']
    assert all(v['quota_clean'] and v['error']==0 and v['maximum_bitwise_equal'] and abs(v['relative_integral_difference'])<2e-12 for v in r['rows'])
    static.extend(r['rows'])
commands=read('surface-sample-final-v1/commands.json');assert len(commands)==32 and all(r['exit_code']==0 and r['quota_clean'] for r in commands)
proof=read('surface-sample-final-v1-independent.json');assert proof['complete'] and len(proof['cases'])==11 and len(proof['standalone'])==5
for name,c in proof['cases'].items():
    assert all(c['work_equal'].values())
    for phase in ['before','after']:
        p=out/'surface-sample-final-v1'/(name+'-'+phase);p=p/'mesh.npz' if p.is_dir() else p.with_suffix('.npz')
        v=c['independent'][phase];assert sha(p)==v['mesh_sha256'] and v['full_hull']['valid'] and v['boundary_self_intersection_pairs']==0 and v['all_boundary_vertex_links_single_cycles']
        if p.name=='mesh.npz':assert v['vtu_round_trip_exact'] and v['boundary_off_round_trip_exact']
for name,c in proof['standalone'].items():assert all(c['unchanged'].values()) and c['relative_change']<.01
memory=read('surface-sample-memory-v1-commands.json');assert len(memory)==4 and all(r['exit_code']==0 and r['zero_memcheck_errors'] for r in memory)
for r in memory:
    text=(out/f"surface-sample-memory-v1-{r['name']}-{r['mode']}.log").read_text()
    assert 'ERROR SUMMARY: 0 errors' in text and f"{r['expected_tests']} passed" in text
qualified=read('surface-sample-qualified-v1.json');assert len(qualified)==5 and all(r['exit_code']==0 for r in qualified)
native={}
for prefix in ['uniform','loop']:
    r=read(f'surface-sample-{prefix}-native-v1-audit.json');assert r['gpu_kernel_activities']>0
    assert set(r['dot_node_types'])<={'KERNEL','MEMCPY','MEMSET','CONDITIONAL','EMPTY','EVENT_RECORD','WAIT_EVENT'}
    assert all('DEVICE to DEVICE' in k for k in r['dot_copy_kinds'])
    assert all('DTOD' in k for k in r['gpu_memcpy_kinds'])
    assert not any(any(x in k.lower() for x in ['memcpy','memalloc','memfree','hostfunc','callback']) for k in r['cuda_api_calls'])
    assert r['cuda_api_calls'].get('cudaGraphLaunch_v10000')==1
    assert sum(r['cuda_api_calls'].get(k,0) for k in ['cuCtxSynchronize','cuStreamSynchronize'])==1
    native[prefix]=r
for name in ['ordinary-loop','native-loop','padded-loop','lifetime-loop']:
    r=read('surface-sample-'+name+'-comparison-v1.json');assert r['complete'] and r['rows_equal'] and r['stable_pointers']
    assert all(v['equal'] for f in r['frames'].values() for v in f['fields'].values())
assert read('surface-sample-native-full-v1-comparison.json')['complete']
source_bindings(read('surface-sample-uniform-native-v1.json'))
# Preserve rejected/interrupted harness attempts instead of counting them as passes.
assert read('surface-sample-pilot-v1/commands.json')[-1]['exit_code']==-15
assert read('surface-sample-production-tests-v1.json')[0]['exit_code']==-15
assert 'AttributeError' in (out/'surface-sample-pilot-v2-independent.log').read_text()
assert 'NameError' in (out/'surface-sample-pilot-v2-independent-v2.log').read_text()
for name in ['surface-sample-quadrature.md','ordered-surface-queries.md']:
    p=root/'docs'/name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('http://','https://','#')):continue
        q=(p.parent/target.split('#',1)[0]).resolve()
        if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+str(q.relative_to(root))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
result=dict(scope=__doc__,artifacts=len(manifest['files']),manifest_sha256=sha(out/'manifest.json'),production_tests=1387,initial_private_tests=99,clean_static_rows=len(static),clean_complete_and_standalone_jobs=32,independent_meshes=22,exact_geometry_transfers=proof['exact_geometry_transfers'],strict_trajectory_parity=proof['strict_trajectory_parity'],memory_jobs=4,native_kernel_activities={k:v['gpu_kernel_activities'] for k,v in native.items()},source_files=len(source['source_sha256']),runtime_changes=source['changed_files'],original_failures_retained=True,verifier_sha256=sha(Path(__file__)))
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');p.chmod(0o444);print(result)
