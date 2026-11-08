"""Materialize only reproducible source files at the pinned control commit."""
import json,subprocess
from pathlib import Path
root=Path.cwd();baseline=root/'.work/vtm/extended-reuse-baseline';assert not baseline.exists()
base='689fd31382c490251b953046176b15ccc35e5753'
def git(args,**kw):return subprocess.run(['git','-c','gc.auto=0','-c','maintenance.auto=false',*args],check=True,**kw)
git(['worktree','add','--detach','--no-checkout',str(baseline),base])
git(['-C',str(baseline),'sparse-checkout','init','--no-cone'])
git(['-C',str(baseline),'sparse-checkout','set','--no-cone','--stdin'],input='/warpower/\n/tests/\n/examples/\n/benchmarks/\n/pyproject.toml\n/README.md\n',text=True)
git(['-C',str(baseline),'read-tree','-mu','HEAD'])
assert (baseline/'warpower/_active_delaunay.py').exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=baseline,text=True)
(root/'.work/vtm/extended-reuse-baseline-v1.json').write_text(json.dumps(dict(path=str(baseline),commit=base,sparse_checkout=True,scope='Control source only; all unique files and Git metadata remain persistent.'),indent=2)+'\n')
