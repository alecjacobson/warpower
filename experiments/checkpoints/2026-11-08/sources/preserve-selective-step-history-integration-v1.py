"""Recover exact first-integration sources by reversing later edits and checking recorded hashes."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm');out=w/'selective-step-history-integration-v1-source';assert not out.exists()
record=json.loads((w/'selective-step-history-integrated-contracts-v1.json').read_text());changes={}
for name,h in record['source_sha256'].items():
 p=Path(name);s=p.read_text()
 if hashlib.sha256(p.read_bytes()).hexdigest()==h:continue
 if name=='warpower/_vtm.py':
  s=s.replace('        _selective_step_history=False,\n','')
  s=s.replace('''            if (
                _selective_step_history
                and selective_enabled is not None
                and local_quality_retries == 0
            ):''','''            if selective_enabled is not None and local_quality_retries == 0:''')
 elif name=='warpower/_mesher.py':s=s.replace('                _selective_step_history=True,\n','')
 elif name=='warpower/_selective_step_history.py':s=s.replace('The complete mesher opts in for selective cores without local vertex rollback.','Only device-controlled selective cores without local vertex rollback opt in.')
 elif name=='tests/test_selective_step_history.py':s=s[:s.index('\n\n@pytest.mark.parametrize("quality_rounds,local_retries"')].rstrip()+'\n'
 else:raise AssertionError(name)
 assert hashlib.sha256(s.encode()).hexdigest()==h,name
 changes[name]=s
out.mkdir()
for name,s in changes.items():
 p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
(w/'selective-step-history-integration-v1-source.json').write_text(json.dumps(dict(scope=__doc__,recovered={p:hashlib.sha256(s.encode()).hexdigest() for p,s in changes.items()},matches_executed_contracts_record=True),indent=2)+'\n')
print('Preserved',len(changes),'exact first-integration source versions.')
