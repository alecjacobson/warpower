"""Tie integrated consumers to measured private kernels and unchanged defaults."""
import ast,copy,hashlib,json,subprocess
from pathlib import Path
w=Path('.work/vtm');base='c5b75f44cccb696ae1ad01ea8257039e803fb2ad'
def fn(path,name):
 return copy.deepcopy(next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
checks={}
for label,a,an,b,bn in [('seed','warpower/_grouped_seeds.py','check_groups',w/'float32_seed_queries.py','check_float32_groups'),('face','warpower/_proved_completion.py','search_unproved_faces',w/'float32_face_search.py','search_float32')]:
 x,y=fn(a,an),fn(b,bn);x.name=y.name='checked';checks[label+'_kernel_ast_equal']=ast.dump(x)==ast.dump(y)
for name in ['bounded','unavailable','orientation_filter','sphere_filter','orient3d','symbolic_sphere']:
 checks[name+'_ast_equal']=ast.dump(fn('warpower/_bounded_predicates.py',name))==ast.dump(fn(w/'bounded_float32_filters.py',name))
old=ast.parse(subprocess.check_output(['git','show',base+':warpower/_proved_completion.py'],text=True));new=ast.parse(Path('warpower/_proved_completion.py').read_text())
old.body=[n for n in old.body if not isinstance(n,(ast.Import,ast.ImportFrom))];new.body=[n for n in new.body if not isinstance(n,(ast.Import,ast.ImportFrom))];checks['other_completion_logic_unchanged']=ast.dump(old)==ast.dump(new)
changed=subprocess.check_output(['git','diff',base,'--name-only','--','warpower'],text=True).splitlines();checks['only_three_private_runtime_modules_changed']=set(changed)=={'warpower/_bounded_predicates.py','warpower/_grouped_seeds.py','warpower/_proved_completion.py'}
checks['standalone_benchmarks_tests_examples_readme_unchanged']=not subprocess.check_output(['git','diff',base,'--','benchmarks/mesher.py','examples','README.md','tests/test_analytic_face_search.py','tests/test_grouped_query_masks.py'],text=True)
assert all(checks.values()),checks
checks['sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('warpower/_bounded_predicates.py'),Path('warpower/_grouped_seeds.py'),Path('warpower/_proved_completion.py'),Path('benchmarks/bounded_predicate_proof.py')]}
(w/'float32-source-isolation.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
