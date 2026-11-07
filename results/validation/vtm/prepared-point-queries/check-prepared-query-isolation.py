"""Check production scope and equivalence to the independently tested prototype."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

w=Path('.work/vtm');path=Path('warpower/_prepared_seeds.py')
old=subprocess.check_output(['git','show','6469ab6:warpower/_prepared_seeds.py'],text=True)
new=path.read_text()
def nodes(s):
    return {n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
a,b=nodes(old),nodes(new)
assert set(b)-set(a)=={'point_value'}
unchanged={k:dump(a[k])==dump(b[k]) for k in a if k!='check'}
assert all(unchanged.values())
changed=subprocess.check_output(['git','diff','--name-only','6469ab6','--','warpower','benchmarks','examples','README.md'],text=True).splitlines()
assert changed==['warpower/_prepared_seeds.py']
reference=nodes(Path('tests/test_prepared_query_masks.py').read_text())['reference_check']
a['check'].name='reference_check'
assert dump(a['check'])==dump(reference)
point=b['point_value']
assert isinstance(point.body[0],ast.Expr) and isinstance(point.body[0].value,ast.Constant)
point.body.pop(0)
assert dump(point)==dump(nodes((w/'prepared_point_queries.py').read_text())['point_value'])
prototype=nodes((w/'prepared_strict_queries.py').read_text())['check_strict']
prototype.name='check'
assert prototype.args.args[-1].arg=='stop_rejected'
prototype.args.args.pop()
loop=next(n for n in prototype.body if isinstance(n,ast.For))
stops=[n for n in loop.body if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and isinstance(n.test.values[0],ast.Name) and n.test.values[0].id=='stop_rejected']
assert len(stops)==1
loop.body.remove(stops[0])
assert dump(prototype)==dump(b['check'])
result=dict(baseline='6469ab6fdb7c774baec8d52b4b3064e02315a728',changed_existing_runtime_files=changed,unchanged_preparation_allocation_and_run=unchanged,reference_kernel_matches_previous_AST=True,production_point_expression_matches_independently_tested_prototype_AST=True,production_checker_matches_exhaustive_prototype_AST=True,new_function='point_value',changed_function='check',source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(w/'prepared-query-source-isolation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
