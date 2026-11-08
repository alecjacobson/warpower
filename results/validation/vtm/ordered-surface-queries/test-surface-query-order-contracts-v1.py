"""Run unchanged cache contracts with the private ordered cache class."""
import pytest
import tests.test_surface_query_cache as contracts
from surface_query_order_v1 import OrderedSurfaceQueryCache

contracts.Cache = OrderedSurfaceQueryCache
raise SystemExit(pytest.main(["-q", "tests/test_surface_query_cache.py"]))
