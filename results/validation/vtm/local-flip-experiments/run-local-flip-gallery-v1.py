"""Private complete VTM experiment with bounded exact local candidates."""
import json,hashlib,sys
from pathlib import Path
from local_flip_mesh_v1 import install,COLUMNS
instances=install()
from examples.vtm_gallery import main
try:main()
finally:
 out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
 report=dict(scope=__doc__,columns=COLUMNS,instances=[dict(capacity=m.n,statistics=m.flip_counters.numpy().tolist()) for m in instances],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/local_flip_mesh_v1.py'),Path('.work/vtm/local_flips_v1.py')]})
 path=out/'flips.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
