"""Bind public CPU benchmark controls to native contracts and exercise the public CLI."""
import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root=Path.cwd();w=root/'.work/vtm';out=w/'cpu-surface-policies-production-v1'
assert not out.exists()
baseline='b17e50a100a380432d91f2c49b78e9581577aef3'
runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','diff',baseline,'--','warpower','tests','examples','README.md','pyproject.toml'])
assert set(subprocess.check_output(['git','diff',baseline,'--name-only'],text=True).splitlines())=={'benchmarks/cpu/README.md','benchmarks/cpu/mesh3.cpp','benchmarks/cpu_mesher.py'}
assert ast.dump(ast.parse((root/'benchmarks/cpu_mesher.py').read_text()),include_attributes=False)==ast.dump(ast.parse((w/'cpu_mesher_feature_policy_v3.py').read_text()),include_attributes=False)
assert (root/'benchmarks/cpu/mesh3.cpp').read_bytes()==(w/'cpu-mesh3-feature-policy-v2.cpp').read_bytes()
contracts=w/'cpu-feature-policy-contracts-v3/result.json'
assert json.loads(contracts.read_text())['complete']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
build=w/'cpu-feature-policy-build-v2.json';assert json.loads(build.read_text())['source_sha256']==sha(root/'benchmarks/cpu/mesh3.cpp')
out.mkdir()
report=dict(scope=__doc__,baseline=baseline,runtime=runtime,production_source_sha256={n:sha(root/n) for n in ['benchmarks/cpu_mesher.py','benchmarks/cpu/mesh3.cpp','benchmarks/cpu/README.md']},private_runner_ast_equal=True,native_source_byte_equal=True,native_contracts_sha256=sha(contracts),native_build_sha256=sha(build),standalone_and_mesher_sources_unchanged=True,readme_unchanged=True,commands=[],complete=False,limits='Functional CLI checks only; these times are not benchmark samples. Independent runtime geometry sources are unchanged, so no new GPU timing study is attributed to this benchmark-only change.')
surface=w/'cpu-feature-policy-contracts-v3/cube.off'
base=[str(root/'.venv/bin/python'),'-m','benchmarks.cpu_mesher','--surface',str(surface),'--size','.7','--facet-distance','.05','--odt-iterations','0','--repeats','1']
for name,binary,options,code in [('legacy','cpu-mesh3-target-sequential',[],0),('manifold','cpu-mesh3-feature-policy-v2-sequential',['--feature-angle','120','--topology','manifold','--distance-binary',str(w/'surface-distance')],0),('failed-native','cpu-mesh3-target-sequential',['--topology','manifold'],1),('invalid-angle','cpu-mesh3-feature-policy-v2-sequential',['--feature-angle','0'],2)]:
    cmd=base+['--binary',str(w/binary),'--output',str(out/name)]+options
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    (out/(name+'.stdout')).write_text(r.stdout);(out/(name+'.stderr')).write_text(r.stderr)
    assert r.returncode==code,(name,r.stderr)
    row=dict(name=name,command=cmd,exit_code=r.returncode)
    if code!=2:
        p=out/name/'result.json';a=json.loads(p.read_text());assert a['complete'] and a['valid']==(code==0)
        assert a['runs'][0]['cpu_quota_delta'];row['result_sha256']=sha(p)
        if code:
            assert a['successful_samples']==0 and a['median_complete_seconds'] is None and a['median_unthrottled_complete_seconds'] is None
    else:assert not (out/name).exists()
    report['commands'].append(row)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
old=json.loads((w/'cpu-feature-policy-contracts-v3/legacy-default/result.json').read_text())
current=json.loads((out/'legacy/result.json').read_text())
assert old['runs'][0]['mesh_sha256']==current['runs'][0]['mesh_sha256']
paths={str(p):sha(p) for p in (out/'legacy').rglob('*') if p.is_file()}
cmd=report['commands'][0]['command']
r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
assert r.returncode==2 and 'must be an empty directory' in r.stderr
assert {str(p):sha(p) for p in (out/'legacy').rglob('*') if p.is_file()}==paths
report.update(complete=True,historical_mesh_preserved=True,nonempty_output_rejected_without_changes=True)
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('Public native/runner controls, historical defaults, rejection records and evidence overwrite protection pass; GPU/standalone sources unchanged.')
