"""Unchanged cache contracts with directly recorded child-sort setup."""
import pytest
import tests.test_surface_query_cache as contracts
from surface_query_order_v2 import OrderedSurfaceQueryCache

contracts.Cache = OrderedSurfaceQueryCache
raise SystemExit(pytest.main(["-q", "tests/test_surface_query_cache.py"]))
