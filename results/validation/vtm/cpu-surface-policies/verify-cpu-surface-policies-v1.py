"""Verify native controls, rejected records, geometry reviews and publication bindings."""
import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,default=Path('.work/vtm'));p.add_argument('--output',type=Path);args=p.parse_args()
root=Path.cwd();out=args.evidence.resolve();w=root/'.work/vtm'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((out/n).read_text())
proof=read('cpu-surface-policies-production-v1/result.json')
assert proof['complete'] and proof['private_runner_ast_equal'] and proof['native_source_byte_equal']
assert proof['standalone_and_mesher_sources_unchanged'] and proof['readme_unchanged']
assert proof['historical_mesh_preserved'] and proof['nonempty_output_rejected_without_changes']
for n,h in proof['production_source_sha256'].items():
    assert sha(root/n)==h
    assert hashlib.sha256(subprocess.check_output(['git','show',proof['runtime']+':'+n])).hexdigest()==h
assert not subprocess.check_output(['git','diff',proof['baseline'],'--','warpower','tests','examples','README.md','pyproject.toml'])
assert ast.dump(ast.parse((root/'benchmarks/cpu_mesher.py').read_text()),include_attributes=False)==ast.dump(ast.parse((out/'cpu_mesher_feature_policy_v3.py').read_text()),include_attributes=False)
assert (root/'benchmarks/cpu/mesh3.cpp').read_bytes()==(out/'cpu-mesh3-feature-policy-v2.cpp').read_bytes()
for version in [1,2]:
    b=read(f'cpu-feature-policy-build-v{version}.json')
    assert b['complete'] and b['historical_binaries_unchanged'] and len(b['builds'])==2
    assert b['source_sha256']==sha(out/f'cpu-mesh3-feature-policy-v{version}.cpp')
    for r in b['builds']:
        assert r['exit_code']==0 and sha(w/f'cpu-mesh3-feature-policy-v{version}-{r["mode"]}')==r['binary_sha256']
assert read('cpu-feature-policy-contracts-v1-failure.json')['exit_code']==1
for version in [2,3]:
    c=read(f'cpu-feature-policy-contracts-v{version}/result.json')
    assert c['complete'] and c['historical_default_mesh_bitwise_preserved'] and c['explicit_policy_changes_featured_cube_mesh']
    assert len(c['native'])==(5 if version==2 else 7)
    for name,r in c['native'].items():
        assert r['boundary_self_intersections']==0 and r['exit_code']==0
        assert sha(out/f'cpu-feature-policy-contracts-v{version}'/(name+'.txt'))==r['mesh_sha256']
assert read('cpu-distance-policy-contracts-v1/result.json')['complete']
failure=read('cpu-scan-topology-failure-v1.json')
assert failure['all_same_mesh'] and not failure['valid_material_manifold'] and len(failure['meshes'])==3
for row in failure['meshes']:
    assert sha(out/Path(row['path']).relative_to('.work/vtm'))==row['sha256']
    assert row['nonmanifold_boundary_edges']==1 and row['witnesses'][0]['incident_faces']==4
pilot=read('cpu-scan-feature-pilot-v2-review.json');assert pilot['complete']
assert [(v['accepted_samples'],v['accepted_unthrottled_samples']) for v in pilot['cases'].values()]==[(0,0),(0,0),(2,0)]
controls=read('cpu-scan-controls-v1/report.json');review=read('cpu-scan-controls-v1-review.json')
assert controls['complete'] and review['complete'] and len(controls['cases'])==7
assert [n for n,c in review['cases'].items() if c['accepted_samples']]==['odt0','tight-fit','loose-fit','loose-strict-odt']
for folder,groups in [('cpu-scan-feature-pilot-v2',pilot['cases']),('cpu-scan-controls-v1',review['cases'])]:
    for name,c in groups.items():
        path=out/folder/name/'result.json';a=json.loads(path.read_text());assert a['complete']
        assert sha(path)==c['result_sha256']
        for i,(r,v) in enumerate(zip(a['runs'],c['records'])):
            assert sha(out/folder/name/f'mesh-{i}.txt')==r['mesh_sha256']==v['mesh_sha256']
            assert r['quality_valid']==v['initial_quality_valid'] and r['cpu_throttled']==v['cpu_throttled']
            assert r['cpu_quota_delta']
            if v['accepted']:
                assert not v['bad_vertex_links'] and v['boundary_self_intersection_pairs']==0 and v['components']==1 and v['boundary_euler']==2
            if folder=='cpu-scan-controls-v1':assert not r['cpu_throttled']
        if not a['successful_samples']:
            assert a['median_complete_seconds'] is None and a['median_unthrottled_complete_seconds'] is None
        if folder=='cpu-scan-controls-v1':assert sha(path)==controls['cases'][name]['result_sha256']
for c in proof['commands']:
    if 'result_sha256' in c:assert sha(out/'cpu-surface-policies-production-v1'/c['name']/'result.json')==c['result_sha256']
result=dict(scope=__doc__,runtime=proof['runtime'],native_contracts=7,public_cli_checks=len(proof['commands']),scan_controls=7,accepted_scan_controls=4,all_scan_control_quota_windows_clean=True,pilot_valid_unthrottled_samples=0,original_failures_retained=True,standalone_and_mesher_sources_unchanged=True,readme_unchanged=True)
if (out/'manifest.json').exists():
    m=read('manifest.json')
    for row in m['files']:
        a,b=root/row['original'],root/row['path']
        assert sha(a)==sha(b)==row['sha256'] and a.stat().st_ino==b.stat().st_ino and not b.stat().st_mode&0o222
    for row in m['shared_references']:assert sha(root/row['path'])==row['sha256']
    result.update(artifacts=len(m['files']),manifest_sha256=sha(out/'manifest.json'))
    for name in ['docs/cpu-surface-policies.md','docs/cpu-mesher-baseline.md','benchmarks/cpu/README.md']:
        p=root/name
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http://','https://','#')):continue
            q=(p.parent/target.split('#',1)[0]).resolve()
            if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+str(q.relative_to(root))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
result['verifier_sha256']=sha(Path(__file__))
if args.output:
    assert not args.output.exists();args.output.write_text(json.dumps(result,indent=2)+'\n');args.output.chmod(0o444)
print(result)
