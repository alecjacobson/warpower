import pytest
from current_selection_cache_v1 import install
install()
raise SystemExit(pytest.main(['-q', 'tests/test_selective.py', 'tests/test_vtm_controls.py',
                             'tests/test_containment_selection.py', 'tests/test_active_sites.py']))
