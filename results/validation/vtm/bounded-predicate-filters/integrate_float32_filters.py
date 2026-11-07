"""Adopt the verified private arithmetic in the two existing VTM kernels."""
from pathlib import Path
import ast,json,shutil,subprocess
root=Path.cwd();w=root/'.work/vtm';stage=w/'float32-staged'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip().startswith('c5b75f4')
assert not subprocess.check_output(['git','diff','--','warpower','tests','benchmarks'],text=True)
checks=json.loads((w/'float32-broad-independent.json').read_text())
assert len(checks['cases'])==6
for row in checks['cases'].values():
 assert row['independent']['full_hull']['valid'] and all(row['work_equal'].values())
 assert row['after_ms']<row['before_ms']
for name,directory in [('_bounded_predicates.py','warpower'),('_flushed_bounded_predicates.py','tests'),('test_bounded_predicates.py','tests'),('test_bounded_predicate_proof.py','tests'),('bounded_predicate_proof.py','benchmarks')]:
 dest=root/directory/name;assert not dest.exists();shutil.copyfile(stage/name,dest)
p=root/'warpower/_proved_completion.py';s=p.read_text();assert s.count('from . import _analytic_predicates as af')==1;p.write_text(s.replace('from . import _analytic_predicates as af','from . import _bounded_predicates as af'))
p=root/'warpower/_grouped_seeds.py';s=p.read_text();module=ast.parse(s);f=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='check_groups')
private=ast.parse((w/'float32_seed_queries.py').read_text());new=next(n for n in private.body if isinstance(n,ast.FunctionDef) and n.name=='check_float32_groups');new.name='check_groups'
lines=s.splitlines(keepends=True);s=''.join(lines[:f.decorator_list[0].lineno-1])+ast.unparse(new)+'\n'+''.join(lines[f.end_lineno:])
s=s.replace('from . import _sphere_bounds as sb','from . import _bounded_predicates as low\nfrom . import _sphere_bounds as sb\nfrom ._analytic_predicates import strict')
s=s.replace('strict Ball64 test and complete symbolic fallback.','bounded binary32 first stage, strict Ball64 test and complete symbolic fallback.')
p.write_text(s)
print('Integrated bounded arithmetic, two private consumers, proof and tests.')
