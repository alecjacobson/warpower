"""Bind the production launch mapping to the qualified private arithmetic and unchanged APIs."""
import ast
import copy
import hashlib
import json
import subprocess
from pathlib import Path

w=Path('.work/vtm');dest=w/'surface-sample-production-source-v2.json';assert not dest.exists()
baseline='c4dcae4bb764ad6640a81308ffdace1c3f27fd8f'
runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'])
def function(t,name):return next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name)
def equal(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
original=ast.parse(subprocess.check_output(['git','show',baseline+':warpower/_extraction.py'],text=True))
production=ast.parse(Path('warpower/_extraction.py').read_text())
candidate=ast.parse((w/'surface_sample_parallel_v1.py').read_text())
a=copy.deepcopy(function(production,'surface_error'));a.name='surface_error_samples';assert isinstance(a.body.pop(0),ast.Expr)
assert equal(a,function(candidate,'surface_error_samples'))
reference=ast.parse(Path('tests/surface_quadrature_reference.py').read_text());a=copy.deepcopy(function(reference,'face_quadrature'));a.name='surface_error';assert equal(a,function(original,'surface_error'))
restored=copy.deepcopy(production)
for i,n in enumerate(restored.body):
    if isinstance(n,ast.FunctionDef) and n.name=='surface_error':restored.body[i]=copy.deepcopy(function(original,'surface_error'))
cls=next(n for n in restored.body if isinstance(n,ast.ClassDef) and n.name=='ExtractionWorkspace')
launch=function(cls,'_surface_metrics').body[0].value
assert ast.unparse(launch.args[1])=='3 * len(self.boundary_faces)';launch.args[1]=launch.args[1].right
assert equal(restored,original)
paths=subprocess.check_output(['git','ls-files','warpower','tests','benchmarks','examples','README.md','pyproject.toml'],text=True).splitlines()
hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
changed=[];unchanged=[]
for p in paths:
    value=subprocess.run(['git','show',baseline+':'+p],capture_output=True)
    (unchanged if value.returncode==0 and hashlib.sha256(value.stdout).hexdigest()==hashes[p] else changed).append(p)
assert set(changed)=={'warpower/_extraction.py','tests/surface_quadrature_reference.py','tests/test_surface_quadrature.py'}
report=dict(scope=__doc__,baseline=baseline,runtime=runtime,source_sha256=hashes,changed_files=changed,unchanged_files=unchanged,production_kernel_matches_private=True,reference_matches_original=True,only_runtime_changes='One sample per thread plus threefold launch dimension; docstring only otherwise.',independent_diagram_and_delaunay_sources_unchanged=True,readme_unchanged=True)
dest.write_text(json.dumps(report,indent=2)+'\n');print(dict(runtime=runtime,pinned_files=len(paths),unchanged=len(unchanged),changed=changed))
