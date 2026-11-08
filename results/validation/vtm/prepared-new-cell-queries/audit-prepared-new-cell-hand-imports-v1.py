"""Loaded project dependencies of the unchanged full hand benchmark; timings diagnostic only."""
import hashlib,json,sys
from pathlib import Path
from benchmarks.delaunay_recovery import main
main()
root=Path.cwd().resolve();out=Path(sys.argv[sys.argv.index('--output')+1]);rows={}
for name,module in list(sys.modules.items()):
 if name.startswith(('warpower','benchmarks','tests')) and getattr(module,'__file__',None):
  p=Path(module.__file__).resolve()
  try:rel=str(p.relative_to(root))
  except ValueError:continue
  rows[name]=dict(path=rel,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
r=dict(scope=__doc__,modules=rows,changed_modules_loaded={n:n in sys.modules for n in ['warpower._proved_completion','warpower._unproved_queries']});out.with_suffix('.imports.json').write_text(json.dumps(r,indent=2)+'\n');print('Changed modules loaded',r['changed_modules_loaded'],flush=True)
