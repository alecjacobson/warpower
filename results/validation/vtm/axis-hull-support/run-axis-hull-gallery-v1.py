"""Private complete VTM experiment using exact coordinate-plane support."""
import hashlib,json,sys
from pathlib import Path
import warp as wp
from axis_hull_support_v1 import install
wp.init();wp.load_module(module='axis_hull_support_v1',device='cuda:0');install()
from examples.vtm_gallery import main
try:main()
finally:
 out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
 report=dict(scope=__doc__,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/axis_hull_support_v1.py')]})
 path=out/'candidate-source.json';assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
