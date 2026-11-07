import pytest
from install_analytic_face import install
install()
raise SystemExit(pytest.main(['-q','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','tests/test_large_reuse.py','tests/test_support_reuse.py']))
