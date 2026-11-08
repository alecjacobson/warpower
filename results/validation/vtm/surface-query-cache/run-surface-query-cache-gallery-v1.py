"""Fresh complete gallery solve with optional private exact surface-query cache."""
import hashlib,json,sys
from pathlib import Path
candidate='--candidate' in sys.argv
instances=[]
if candidate:
    sys.argv.remove('--candidate')
    from surface_query_cache_v1 import install,FIELDS
    instances=install()
from examples.vtm_gallery import main
main()
out=Path(sys.argv[sys.argv.index('--output')+1]);rows=[]
for c in instances:
    arrays=[getattr(state,field) for state in [c.current,c.previous] for field in FIELDS]+[c.table]
    rows.append(dict(label=c.label,tet_capacity=c.capacity,array_bytes=sum(a.capacity for a in arrays)))
files=[Path(__file__)]
if candidate:files += [Path('.work/vtm/surface_query_cache_v1.py'),Path('.work/vtm/surface_query_cache_kernels_v1.py')]
p=out/'query-cache.json';assert not p.exists();p.write_text(json.dumps(dict(scope=__doc__,candidate=candidate,instances=rows,total_extra_array_bytes=sum(r['array_bytes'] for r in rows),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
