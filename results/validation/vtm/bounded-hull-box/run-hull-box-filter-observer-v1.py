"""Complete gallery observation; added instrumentation is not performance evidence."""
import hashlib,json,sys
from pathlib import Path
from hull_box_filter_observer_v1 import install,COLUMNS
instances=install()
from examples.vtm_gallery import main
try:main()
finally:
 out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
 report=dict(scope=__doc__,columns=COLUMNS,instances=[dict(capacity=len(c.points),statistics=c.hull_box_stats.numpy().tolist()) for c in instances],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/hull_box_filter_observer_v1.py'),Path('.work/vtm/bounded_hull_box_v1.py')]})
 path=out/'observer.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
