"""Bind production policy to the validated prototype and original scalar functions."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';base=w/'selective-step-history-baseline'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
node=lambda p,n:next(x for x in ast.parse(p.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==n)
def body(n):return ast.dump(ast.Module(body=n.body,type_ignores=[]))
class Canonical(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in ['original_start_attempt','original_finish_attempt']:n.id=n.id.replace('original_','_',1)
  return n
for name in ['start_attempt','finish_attempt']:
 assert body(node(base/'warpower/_vtm.py',name))==body(node(root/'warpower/_vtm.py','_'+name))
 assert body(Canonical().visit(node(w/'selective_step_history_v2.py',name)))==body(node(root/'warpower/_selective_step_history.py',name))
for name in ['directions','propose','candidate_ready']:
 assert body(node(base/'warpower/_vtm.py',name))==body(node(root/'warpower/_vtm.py',name))
files=subprocess.check_output(['git','ls-files','--','warpower','tests','benchmarks','examples','pyproject.toml','README.md','LICENSE'],text=True).splitlines()
for p in ['warpower/_selective_step_history.py','tests/test_selective_step_history.py']:
 if p not in files:files.append(p)
files=sorted(files);changed=[p for p in files if not (base/p).exists() or sha(root/p)!=sha(base/p)]
assert changed==['tests/test_selective_step_history.py','warpower/_mesher.py','warpower/_selective_step_history.py','warpower/_vtm.py'],changed
log=w/'selective-step-history-integrated-contracts-v2.log';assert '60 passed' in log.read_text()
report=dict(scope=__doc__,baseline=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip(),source_sha256={p:sha(root/p) for p in files},changed_files=changed,unchanged_files=len(files)-len(changed),private_policy_sha256=sha(w/'selective_step_history_v2.py'),scalar_functions_ast_identical=True,proposal_directions_acceptance_ast_identical=True,private_policy_ast_identical=True,integrated_contracts=dict(passed=60,exit_code=0,seconds=60.57,log_sha256=sha(log)),complete_mesher_only=True)
p=w/'selective-step-history-production-source-v1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
print('Audited',len(files),'files,',len(changed),'changed; scalar bodies and policy AST identical.')
