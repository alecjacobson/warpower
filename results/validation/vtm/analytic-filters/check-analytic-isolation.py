"""Verify the adopted scope and arithmetic against the validated private probe."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();base='fb795ac1b83f4bbf0c5759fb6f800ff41e1c403e'
def old(path):return subprocess.check_output(['git','show',base+':'+path],text=True)
def functions(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
path='warpower/_proved_completion.py';before=functions(old(path));after=functions((root/path).read_text())
assert before.keys()==after.keys()
changed=[k for k in before if before[k]!=after[k]];assert changed==['search_unproved_faces']
normalized=(root/path).read_text().replace('from . import _analytic_predicates as af\n','').replace('af.orient3d(', 'fp.orient3d(').replace('af.symbolic_sphere(', 'symbolic_sphere(')
assert normalized==old(path)
for directory in ['warpower','benchmarks','examples','tests']:
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',base,'--',directory],text=True).splitlines()
    for p in paths:
        if p==path:continue
        assert subprocess.check_output(['git','show',base+':'+p])==(root/p).read_bytes(),p
assert old('README.md')==(root/'README.md').read_text()
private=(root/'.work/vtm/analytic_face_search.py').read_text().replace('def symbolic(', 'def symbolic_sphere(').replace('return symbolic_sphere(points,ids,query)','return legacy_symbolic_sphere(points,ids,query)')
tested=functions(private);adopted=functions((root/'warpower/_analytic_predicates.py').read_text())
assert all(tested[k]==value for k,value in adopted.items())
constants=lambda s:{n.targets[0].id:ast.dump(n.value) for n in ast.parse(s).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.endswith('_BOUND')}
assert constants(private)==constants((root/'warpower/_analytic_predicates.py').read_text())
result=dict(baseline=base,changed_existing_functions=changed,all_existing_other_production_tests_benchmarks_examples_and_readme_identical=True,new_filter_function_asts_and_constants_equal_validated_private=True,source_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['warpower/_proved_completion.py','warpower/_analytic_predicates.py','tests/test_analytic_predicates.py','tests/test_analytic_face_search.py']})
(root/'.work/vtm/analytic-source-isolation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
