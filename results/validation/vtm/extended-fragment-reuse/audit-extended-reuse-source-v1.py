"""Prove runtime diff is exactly the previously measured private setup decision."""
import ast,hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();w=root/'.work/vtm';base='689fd31382c490251b953046176b15ccc35e5753'
old=subprocess.check_output(['git','show',base+':warpower/_active_delaunay.py'],text=True)
new=Path('warpower/_active_delaunay.py').read_text()
assert new==old.replace('2048 < len(points) <= 4096','2048 < len(points) <= 16384')
assert ast.dump(ast.parse(new.replace('2048 < len(points) <= 16384','2048 < len(points) <= 4096')))==ast.dump(ast.parse(old))
changed=subprocess.check_output(['git','diff','--name-only',base,'--','warpower','tests','benchmarks','examples','README.md'],text=True).splitlines()
assert changed==['tests/test_large_reuse.py','warpower/_active_delaunay.py'],changed
sources={}
for folder in ['warpower','tests','benchmarks','examples']:
    for p in sorted(Path(folder).glob('*.py')):sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(baseline=base,scope=__doc__,changed_paths=changed,old_runtime_sha256=hashlib.sha256(old.encode()).hexdigest(),new_runtime_sha256=hashlib.sha256(new.encode()).hexdigest(),only_runtime_change='ActiveDelaunayWorkspace full-route reuse upper reserved capacity4096->16384',all_other_runtime_kernels_and_standalone_paths_identical=True,source_sha256=sources)
p=w/'extended-reuse-source-audit-v1.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)
