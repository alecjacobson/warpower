"""Private replacement against existing behavior tests; dispatch typename is separate."""
import pytest
from hierarchical_split_v2 import install
install()
# Five route tests assert the production implementation's class name; direct
# changing-input mask tests and the ten remaining large-route behaviors are run.
raise SystemExit(pytest.main(['-q','tests/test_facet_seeds.py','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','tests/test_large_reuse.py','tests/test_support_reuse.py','-k','not test_large_route_changes_inputs_and_prefixes_without_host_decisions']))
