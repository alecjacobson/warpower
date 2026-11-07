"""Prepare isolated production files; do not mutate active pilot runtime."""
from pathlib import Path
w=Path('.work/vtm');stage=w/'float32-staged';stage.mkdir(exist_ok=True)
s=(w/'bounded_float32_filters.py').read_text().replace('Private bounded-range binary32 strict filters, #180.','Bounded-range binary32 strict filters for private VTM consumers.').replace('from warpower import _analytic_predicates as old','from . import _analytic_predicates as old').replace('from warpower import _filtered_predicates as fp','from . import _filtered_predicates as fp').replace('prove_float32_filter_bounds.py','benchmarks/bounded_predicate_proof.py')
(stage/'_bounded_predicates.py').write_text(s)
s=(w/'flushed_float32_filters.py').read_text().replace('from bounded_float32_filters import','from warpower._bounded_predicates import')
(stage/'_flushed_bounded_predicates.py').write_text(s)
s=(w/'test_float32_filter_bounds.py').read_text().replace('import bounded_float32_filters as normal','from warpower import _bounded_predicates as normal').replace('import flushed_float32_filters as flushed','from tests import _flushed_bounded_predicates as flushed')
(stage/'test_bounded_predicates.py').write_text(s)
s=(w/'prove_float32_filter_bounds.py').read_text().replace('import ast\n','import ast\nimport argparse\nimport copy\n').replace('root = Path.cwd()','root = Path(__file__).resolve().parents[1]')
insert='''
# Couple the algebraic proof to the executable binary32 expression tree. Only
# the range gate, final exact promotions and separately bounded threshold floor
# differ from the reference grouping analyzed below.
bounded_source = root / 'warpower/_bounded_predicates.py'
bounded_module = ast.parse(bounded_source.read_text())
def function_in(module, name):
    return next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name==name)
class RemovePromotions(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='wp' and node.func.attr=='float64':
            assert len(node.args)==1 and not node.keywords
            return node.args[0]
        return node
def normalized(module,name,binary32):
    f=copy.deepcopy(function_in(module,name))
    if binary32:
        gate=f.body.pop(0)
        args=[a.arg for a in f.args.args]
        expected=ast.parse('if not ('+' and '.join('bounded('+a+')' for a in args)+'):\\n    return unavailable()').body[0]
        assert ast.dump(gate)==ast.dump(expected)
        for node in ast.walk(f):
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Attribute) and node.targets[0].attr=='error':
                value=node.value
                assert isinstance(value,ast.Call) and ast.unparse(value.func)=='wp.float64'
                term=value.args[0]
                assert isinstance(term,ast.BinOp) and isinstance(term.op,ast.Add) and ast.unparse(term.right)=='ABS_FLOOR'
                node.value=term.left
    f=RemovePromotions().visit(f)
    return ast.dump(f)
reference_module=ast.parse(source.read_text())
for name in ['orientation_filter','sphere_filter']:
    assert normalized(reference_module,name,False)==normalized(bounded_module,name,True),name
expected_range=ast.parse('return '+' and '.join('wp.isfinite(p['+str(i)+'])' for i in range(3))+' and '+' and '.join('wp.abs(p['+str(i)+']) <= 16.0' for i in range(3))).body[0]
assert ast.dump(function_in(bounded_module,'bounded').body[0])==ast.dump(expected_range)
constants={node.targets[0].id:node.value for node in bounded_module.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
for name, exponent in [('ORIENT_BOUND',-19),('SPHERE_BOUND',-17),('ABS_FLOOR',-64)]:
    expected=ast.parse('wp.constant(wp.float32(2.0**'+str(exponent)+'))',mode='eval').body
    assert ast.dump(constants[name])==ast.dump(expected),name

'''
s=s.replace('nodes = []\n','nodes = []\n'+insert)
s=s.replace("result = dict(source_sha256=", "result = dict(bounded_source_sha256=hashlib.sha256(bounded_source.read_bytes()).hexdigest(), formula_range_constants_verified=True, source_sha256=")
s=s.replace("out = root / '.work/vtm/float32-filter-bound-proof.json'\nout.write_text(json.dumps(result, indent=2) + '\\n')", "parser=argparse.ArgumentParser()\nparser.add_argument('--output',type=Path)\nargs=parser.parse_args()\nif args.output:\n    args.output.write_text(json.dumps(result,indent=2)+'\\n')")
(stage/'bounded_predicate_proof.py').write_text(s)
(stage/'test_bounded_predicate_proof.py').write_text('''"""Exact expression-tree bounds must match the current executable filters."""
import json
from pathlib import Path
import subprocess
import sys

def test_bounded_predicate_proof(tmp_path):
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'bound.json'
    subprocess.run([sys.executable,str(root/'benchmarks/bounded_predicate_proof.py'),'--output',str(output)],cwd=root,check=True)
    proof=json.loads(output.read_text())
    assert proof['formula_range_constants_verified']
    for row in proof['filters'].values():
        assert row['coefficient_inequality'] and row['floor_inequality']
        assert row['permanent_matches_absolute_monomials']
''')
print(stage)
