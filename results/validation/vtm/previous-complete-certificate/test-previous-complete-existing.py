import pytest
from whole_previous_certificate import install
install()
raise SystemExit(pytest.main(['-q','tests/test_facet_seeds.py','tests/test_proved_completion.py','tests/test_reuse_delaunay.py','.work/vtm/test_previous_complete_large.py','tests/test_support_reuse.py']))
