"""Observe actual production caches after unchanged constructor/replay measurements."""
import gc
import hashlib
import json
import os
import runpy
from pathlib import Path

from warpower import _surface_query_cache as original

w=Path(__file__).parent
runpy.run_path(str(w/'run-tangential-production-gallery-v1.py'))
caches=[obj for obj in gc.get_objects() if isinstance(obj,original.SurfaceQueryCache)]
rows=[]
for c in caches:
    ordered=getattr(c,'_has_order_storage',False)
    rows.append(dict(capacity=c.capacity,owns_order_storage=ordered,order_payload_bytes=24*c.capacity+4 if ordered else 0))
paths=[Path(__file__),Path(original.__file__)]
ordered_source=Path(original.__file__).with_name('_ordered_surface_queries.py')
if ordered_source.exists():paths.append(ordered_source)
report=dict(scope=__doc__,phase=os.environ['WARPOWER_QUERY_ORDER_PHASE'],caches=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},limits='GC observation occurs after timing; payload excludes retained CUB scratch. Constructor pool observations include actual workspace costs. No private cache installation or runtime patch is used.')
p=Path(os.environ['WARPOWER_QUERY_ORDER_REPORT'])
assert not p.exists()
p.write_text(json.dumps(report,indent=2)+'\n')
