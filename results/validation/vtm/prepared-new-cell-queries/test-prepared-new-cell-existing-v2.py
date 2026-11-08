"""Private setup binding against all existing relevant behavior cases."""
import pytest
from prepared_new_cell_integration_v2 import install
install()
raise SystemExit(pytest.main(['-q','tests/test_facet_seeds.py','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','tests/test_large_reuse.py','tests/test_support_reuse.py']))
