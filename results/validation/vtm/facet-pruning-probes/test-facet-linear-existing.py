import pytest
from install_facet_linear import install
install()
raise SystemExit(pytest.main(['-q','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','tests/test_large_reuse.py','tests/test_support_reuse.py']))
