"""Observe one warmed fixed16K native trajectory; existing reconstruction always runs."""
import hashlib,json,runpy,sys
from pathlib import Path
from unchanged_input_observer_v1 import install,COLUMNS
instances=install()
runpy.run_path('.work/vtm/profile-larger-prefix-production-v1.py',run_name='__main__')
output=Path(sys.argv[sys.argv.index('--output')+1]).with_suffix('.observations.json');assert not output.exists()
rows=[]
for m in instances:
 o=m.observer;calls=int(o.calls.numpy()[0]);assert o.overflow.numpy()[0]==0 and calls<=len(o.records)
 rows.append(dict(capacity=m.n,records=o.records.numpy()[:calls].tolist()))
output.write_text(json.dumps(dict(columns=COLUMNS,instances=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/unchanged_input_observer_v1.py'),Path('.work/vtm/profile-larger-prefix-production-v1.py')]}),indent=2)+'\n')
