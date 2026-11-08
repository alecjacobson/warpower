"""Complete gallery observation with all original queries retained; timings diagnostic."""
import hashlib,json,sys
from pathlib import Path
from surface_query_reuse_observer_v2 import install,report
install()
from examples.vtm_gallery import main
try:main()
finally:
    out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
    report(out/'query-observer.json')
    files=[Path(__file__),Path('.work/vtm/surface_query_reuse_observer_v2.py'),Path('.work/vtm/surface_query_reuse_kernels_v1.py')]
    (out/'observer-source-sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
