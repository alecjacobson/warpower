"""Private size-gated ordering: original cache contracts and live threshold tests."""
import pytest
import tests.test_surface_query_cache as contracts
from surface_query_order_v3 import SizeGatedSurfaceQueryCache

contracts.Cache = SizeGatedSurfaceQueryCache
raise SystemExit(pytest.main(['-q', 'tests/test_surface_query_cache.py', '.work/vtm/test_surface_query_order_gate_v1.py']))
