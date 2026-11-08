"""Prepare private register-limit variants with unchanged ordered classifier arithmetic."""
import ast
import base64
import copy
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

w=Path('.work/vtm');source=Path('warpower/_ordered_surface_queries.py')
dest=w/'surface-register-limits-source-v1.json';assert not dest.exists()
tree=ast.parse(source.read_text())
names=['classify_cached_ordered','selection_cached_ordered','retained_cached_ordered']
functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
assert set(functions)==set(names)
prefix=[]
for n in tree.body:
    if isinstance(n,(ast.Import,ast.ImportFrom)):
        n=copy.deepcopy(n)
        if isinstance(n,ast.ImportFrom) and n.level:
            assert n.level==1;n.module='warpower'+('.'+n.module if n.module else '');n.level=0
        prefix.append(n)
    elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='wp.set_module_options':prefix.append(copy.deepcopy(n))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
report=dict(scope=__doc__,baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),original_source_sha256=sha(source),variants={},warp={})
for limit in [96,128,192]:
    nodes=copy.deepcopy(prefix)
    for name in names:
        f=copy.deepcopy(functions[name]);f.decorator_list=[ast.parse(f'wp.kernel(cuda_max_registers={limit})',mode='eval').body];nodes.append(f)
    p=w/f'surface_register_limit_{limit}_v1.py';assert not p.exists();p.write_text('"""Private register-limit experiment; no changed classifier arithmetic."""\n'+ast.unparse(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])))+'\n')
    parsed=ast.parse(p.read_text())
    for f in [n for n in parsed.body if isinstance(n,ast.FunctionDef)]:
        a,b=copy.deepcopy(f),copy.deepcopy(functions[f.name]);a.decorator_list=[];b.decorator_list=[];assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    report['variants'][str(limit)]=dict(path=str(p),sha256=sha(p),classifier_arithmetic_ast_equal=True)
dist=importlib.metadata.distribution('warp-lang');report['warp']['distribution_version']=dist.version
for name in ['warp/_src/context.py','warp/native/warp.cu','warp/bin/warp.so']:
    f=next(p for p in dist.files if str(p)==name);actual=sha(Path(dist.locate_file(f)));expected=base64.urlsafe_b64decode(f.hash.value+'='*(-len(f.hash.value)%4)).hex();assert actual==expected
    report['warp'][name]=dict(sha256=actual,installed_record_matches=True)
dest.write_text(json.dumps(report,indent=2)+'\n')
print('Prepared three AST-identical private classifier variants; installed Warp API/native bytes match RECORD.')
