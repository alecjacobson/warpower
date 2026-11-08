"""Bind production ordering arithmetic and private constructor scope to measured probes."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

w=Path('.work/vtm')
dest=w/'surface-query-order-production-source-v1.json'
assert not dest.exists()
baseline='6d362e7b44779e8a8cb7fe4fa81fc124ccf93ff7'
runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'])
source=Path('warpower/_ordered_surface_queries.py')
tree=ast.parse(source.read_text())
original=ast.parse(Path('warpower/_surface_query_cache.py').read_text())
private=ast.parse((w/'surface_query_order_v1.py').read_text())
gate=ast.parse((w/'surface_query_order_v4.py').read_text())
sort=ast.parse((w/'surface_query_order_v2.py').read_text())

def function(t,n):
    return next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==n)

def equal(a,b):
    return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)

arithmetic={}
for name in ['classify_cached','selection_cached','retained_cached']:
    a=function(tree,name+'_ordered')
    assert equal(a,function(private,name+'_ordered'))
    b=copy.deepcopy(a)
    b.name=name
    assert b.args.args.pop().arg=='order'
    b.body[0]=copy.deepcopy(function(original,name).body[0])
    assert equal(b,function(original,name))
    arithmetic[name]=dict(private_probe_identical=True,original_arithmetic_identical_except_tid_permutation=True)
assert equal(function(tree,'centroid_keys'),function(private,'centroid_keys'))
assert equal(function(tree,'choose_order'),function(gate,'choose_order'))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OrderedSurfaceQueryCache')
oldcls=next(n for n in gate.body if isinstance(n,ast.ClassDef) and n.name=='SizeGatedSurfaceQueryCache')
sortcls=next(n for n in sort.body if isinstance(n,ast.ClassDef))
for name in ['launch','_ordered']:
    assert equal(function(cls,name),function(oldcls,name)),name
assert equal(function(cls,'_sort'),function(sortcls,'_sort'))
threshold=next(n for n in cls.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_minimum_query_count')
assert ast.literal_eval(threshold.value)==32768
for t in [tree,original,private,gate]:
    opts=next(n.value for n in t.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='set_module_options')
    assert ast.literal_eval(opts.args[0])=={'enable_backward':False,'fast_math':False}
imports=[]
for name in ['_extraction.py','_quality_moves.py','_selective.py']:
    path='warpower/'+name
    old=subprocess.check_output(['git','show',baseline+':'+path],text=True)
    new=Path(path).read_text()
    target='from ._ordered_surface_queries import OrderedSurfaceQueryCache as SurfaceQueryCache'
    assert new.count(target)==1
    assert new.replace(target,'from ._surface_query_cache import SurfaceQueryCache')==old
    imports.append(path)
prior=json.loads(Path('results/validation/vtm/recovery-query-storage/manifest.json').read_text())
paths=sorted(set(prior['runtime_source_sha256'])|{'warpower/_ordered_surface_queries.py','tests/test_ordered_surface_queries.py'})
hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
unchanged=[]
changed=[]
for p in paths:
    if p not in prior['runtime_source_sha256']:
        changed.append(p)
    elif hashlib.sha256(subprocess.check_output(['git','show',baseline+':'+p])).hexdigest()==hashes[p]:
        unchanged.append(p)
    else:
        changed.append(p)
assert set(changed)==set(imports)|{'warpower/_ordered_surface_queries.py','tests/test_ordered_surface_queries.py'}
assert len(paths)==264 and len(unchanged)==259
assert 'warpower/_surface_query_cache.py' in unchanged and 'README.md' in unchanged
report=dict(scope=__doc__,baseline=baseline,runtime=runtime,source_sha256=hashes,unchanged_files=unchanged,changed_files=changed,private_constructor_imports_only=imports,ordered_arithmetic=arithmetic,centroid_keys_and_live_gate_identical=True,launch_ordered_sort_methods_identical=True,minimum_query_count=32768,original_scalar_cache_unchanged=True,readme_unchanged=True,independent_diagram_and_delaunay_sources_unchanged=True,private_source_sha256={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in ['surface_query_order_v1.py','surface_query_order_v2.py','surface_query_order_v4.py']},limitations=['Production folds the measured inherited constructor into one private class/module and permits a private threshold override for forced-path tests; default threshold/storage/capture behavior needs production runtime qualification.','The explicit source audit is not a general Python alias proof.','Runtime/native/lifetime and full paired qualification are separate gates.'])
dest.write_text(json.dumps(report,indent=2)+'\n')
print(dict(runtime=runtime,pinned_files=len(paths),unchanged=len(unchanged),changed=changed))
