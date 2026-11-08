"""Static self/super callback reachability for the issue241 setup-only proposal."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';dest=w/'quality-query-storage-static-v1.json';assert not dest.exists()
paths={'QualityMove':root/'warpower/_quality_moves.py','TopologyRecovery':root/'warpower/_topology_recovery.py'}
methods={}
for owner,path in paths.items():
 cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name==owner)
 methods[owner]={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
def resolve(name):
 return ('TopologyRecovery',name) if name in methods['TopologyRecovery'] else ('QualityMove',name)
all_names=set(methods['QualityMove'])|set(methods['TopologyRecovery'])
edges={}
for owner,group in methods.items():
 for name,node in group.items():
  refs=set()
  for n in ast.walk(node):
   if not isinstance(n,ast.Attribute) or n.attr not in all_names:continue
   if isinstance(n.value,ast.Name) and n.value.id=='self':refs.add(resolve(n.attr))
   elif isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='super':
    assert owner=='TopologyRecovery';refs.add(('QualityMove',n.attr))
  edges[(owner,name)]=refs
roots=[resolve(n) for n in ['run','reset','_record','_finish_setup']];pending=roots[:];reached=set()
while pending:
 node=pending.pop()
 if node in reached:continue
 reached.add(node);pending.extend(edges[node])
fields={'_query_cache','_query_error','_query_count','_query_minimum','_query_keep','_query_quality'}
users={f:[] for f in fields}
for owner,group in methods.items():
 for name,node in group.items():
  touched={n.attr for n in ast.walk(node) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self' and n.attr in fields}
  for f in touched:users[f].append((owner,name));assert (owner,name) not in reached
assert ('QualityMove','_approximate_quality') not in reached and ('TopologyRecovery','_quality') in reached
for f in fields:assert set(users[f])=={('QualityMove','__init__'),('QualityMove','_approximate_quality')}
r=dict(scope=__doc__,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()},roots=roots,reachable_methods=sorted(reached),unused_approximate_fields=users,prototype_sha256=hashlib.sha256((w/'quality_query_storage_v1.py').read_bytes()).hexdigest(),limitations=['AST audit resolves current explicit self/super method references, including callbacks passed to graph helpers; it is not a general Python alias proof.','The candidate extractor owns a different active cache and is not changed.','Runtime graph capture, memory/lifetime and end-to-end performance/geometry checks remain unexecuted.'])
dest.write_text(json.dumps(r,indent=2)+'\n');print(len(reached),'reachable methods; six approximate-query fields only in unreachable approximate callback and constructor.')
