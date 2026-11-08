"""Private size-gated query-order full-solve wrapper; include ordering work and storage."""
import hashlib
import json
import os
import runpy
from pathlib import Path

from warpower import _surface_query_cache as cache_module

phase = os.environ['WARPOWER_QUERY_ORDER_PHASE']
assert phase in ['before', 'after']
if phase == 'after':
    from surface_query_order_v4 import install
    install()
Base = cache_module.SurfaceQueryCache
instances = []


class ObservedCache(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        instances.append(self)


cache_module.SurfaceQueryCache = ObservedCache
w = Path(__file__).parent
if os.environ['WARPOWER_QUERY_ORDER_KIND'] == 'benchmark':
    runpy.run_module('benchmarks.mesher', run_name='__main__')
else:
    runpy.run_path(str(w / 'run-tangential-production-gallery-v1.py'))
paths = [Path(__file__), w/'surface_query_order_v1.py', w/'surface_query_order_v2.py', w/'surface_query_order_v4.py', Path('warpower/_surface_query_cache.py')]
report = dict(scope=__doc__, phase=phase, source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, caches=[dict(capacity=c.capacity, owns_order_storage=getattr(c, '_has_order_storage', False), order_payload_bytes=24*c.capacity+4 if getattr(c, '_has_order_storage', False) else 0) for c in instances], limits='Payload excludes CUB retained scratch; constructor pool measurements include actual workspace costs. Eager private calls order large caches, while complete captured solves choose by live count/error.')
p = Path(os.environ['WARPOWER_QUERY_ORDER_REPORT'])
assert not p.exists()
p.write_text(json.dumps(report, indent=2)+'\n')
