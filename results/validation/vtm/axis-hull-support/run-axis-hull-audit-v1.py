"""Full support decision comparison; diagnostic replay timing is not performance evidence."""
import hashlib,json,sys
from pathlib import Path
import warp as wp
wp.init()
from axis_hull_audit_v1 import install,COLUMNS
instances=install()
from examples.vtm_gallery import main
try:main()
finally:
 out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
 report=dict(scope=__doc__,columns=COLUMNS,instances=[dict(capacity=len(c.points),statistics=c.axis_audit_stats.numpy().tolist()) for c in instances],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/axis_hull_audit_v1.py'),Path('.work/vtm/axis_hull_support_v1.py')]})
 path=out/'decision-audit.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 assert all(row['statistics'][2]==row['statistics'][3]==0 for row in report['instances']),report
