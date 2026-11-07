"""Verify private grouped queries and isolate the only existing runtime edit."""
import ast,hashlib,json,subprocess
from pathlib import Path
w=Path('.work/vtm');base='ce510966c37143d0df27a53b930918651abdd8d1'
old=subprocess.check_output(['git','show',base+':warpower/_facet_seeds.py'],text=True)
new=Path('warpower/_facet_seeds.py').read_text()
restored=new.replace('from ._grouped_seeds import GroupedSeedQueries','from ._prepared_seeds import PreparedSeedQueries').replace('self._prepared = GroupedSeedQueries(','self._prepared = PreparedSeedQueries(')
assert restored==old
assert subprocess.check_output(['git','diff','--name-only',base,'--','warpower','benchmarks','examples','README.md'],text=True).splitlines()==['warpower/_facet_seeds.py','warpower/_grouped_seeds.py']
def nodes(s):return {n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
a=nodes((w/'prepared_group_queries.py').read_text());b=nodes(Path('warpower/_grouped_seeds.py').read_text())
for name in ['group_bounds','check_groups']:assert dump(a[name])==dump(b[name])
a['GroupPreparedSeedQueries'].name='GroupedSeedQueries'
assert dump(a['GroupPreparedSeedQueries'])==dump(b['GroupedSeedQueries'])
assert set(b)=={'group_bounds','check_groups','GroupedSeedQueries'}
record=dict(baseline=base,only_existing_runtime_change='CertifiedSeeds uses the grouped subclass only when prepare_geometry=True; all default kernels and dispatch remain unchanged.',private_kernel_and_workspace_AST_equal=True,unmodified_reference='warpower/_prepared_seeds.py',source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('warpower/_facet_seeds.py'),Path('warpower/_grouped_seeds.py')]})
(w/'prepared-group-source-isolation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
