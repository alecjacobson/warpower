"""Verify the retained negative register-cap probe and its unchanged production baseline."""
import ast
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

root=Path.cwd();out=root/'results/validation/vtm/surface-register-limits'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((out/n).read_text())
m=read('manifest.json');s=read('surface-register-limits-source-v1.json');r=read('surface-register-limits-summary-v1.json')
assert m['baseline']==s['baseline'] and m['production_source_unchanged'] and not m['adopted']
assert r['complete'] and not r['adopted'] and not r['complete_solve_measured']
assert not subprocess.check_output(['git','diff',m['baseline'],'--','warpower','tests','benchmarks','examples','README.md','pyproject.toml'])
assert sha(root/'warpower/_ordered_surface_queries.py')==s['original_source_sha256']
for row in m['files']:
    a,b=root/row['original'],root/row['path'];assert sha(a)==sha(b)==row['sha256'];assert a.stat().st_ino==b.stat().st_ino and not b.stat().st_mode&0o222
for row in m['shared_references']:assert sha(root/row['path'])==row['sha256']
functions={n.name:n for n in ast.parse((root/'warpower/_ordered_surface_queries.py').read_text()).body if isinstance(n,ast.FunctionDef)}
for limit in [96,128,192]:
    row=s['variants'][str(limit)];p=out/Path(row['path']).name;assert sha(p)==row['sha256']
    for f in [n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)]:
        assert ast.unparse(f.decorator_list[0])==f'wp.kernel(cuda_max_registers={limit})'
        a,b=copy.deepcopy(f),copy.deepcopy(functions[f.name]);a.decorator_list=[];b.decorator_list=[];assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    p=out/'surface-register-limits-v1'/f'contracts-r{limit}.json';c=json.loads(p.read_text());assert c['passed'] and c['complete'] and c['exit_code']==0
    assert sha(p)==r['contracts'][str(limit)]['report_sha256'];assert '19 passed' in p.with_suffix('.log').read_text()
    for n,h in c['source_sha256'].items():assert sha(Path(n))==h
    assert all(v['register_count']==limit for v in c['properties'].values())
commands=read('surface-register-limits-v1/commands.json');assert len(commands)==4 and [v['exit_code'] for v in commands]==[0,0,0,1]
assert "KeyError: ('after', 'cold')" in (out/'surface-register-limits-v1/torus1000.log').read_text()
assert len(read('surface-register-limits-v1/torus1000.json')['rows'])==1
commands=read('surface-register-limits-v2/commands.json');assert len(commands)==7 and all(v['exit_code']==0 for v in commands)
for name,c in r['cases'].items():
    p=out/'surface-register-limits-v2'/(name+'.json');a=json.loads(p.read_text());assert sha(p)==c['report_sha256']
    assert a['complete'] and len(a['rows'])==24 and all(v['quota_clean'] for v in a['rows'])
    assert sha(Path(a['input']))==a['input_sha256']
    for n,h in a['source_sha256'].items():assert sha(Path(n))==h
    for mode in ['full','selection','retained']:
        for temperature in ['cold','warm']:
            rows=[v for v in a['rows'] if v['mode']==mode and v['cache']==temperature]
            assert [v['phase'] for v in rows]==['before','r96','r128','r192']
            assert all(v['output_sha256']==rows[0]['output_sha256'] and v['exact_outputs_match'] for v in rows[1:])
for name in ['surface-register-limits.md','ordered-surface-queries.md']:
    p=root/'docs'/name
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('http://','https://','#')):continue
        q=(p.parent/target.split('#',1)[0]).resolve()
        if not q.exists():subprocess.run(['git','cat-file','-e','HEAD:'+str(q.relative_to(root))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
result=dict(scope=__doc__,artifacts=len(m['files']),manifest_sha256=sha(out/'manifest.json'),contract_case_executions=57,clean_static_rows=96,all_candidate_outputs_bitwise_equal=True,production_unchanged=True,adopted=False,complete_solve_measured=False,original_harness_failure_retained=True,verifier_sha256=sha(Path(__file__)))
p=out/'premerge-integrity.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');p.chmod(0o444)
print(result)
