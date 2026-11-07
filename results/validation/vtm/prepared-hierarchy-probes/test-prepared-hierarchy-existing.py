import os
import pytest
from install_prepared_hierarchy import install
install(os.environ['WARPOWER_PREPARED_TREE_NODES'] == 'polynomial')
raise SystemExit(pytest.main(['-q', 'tests/test_facet_seeds.py', 'tests/test_proved_completion.py', 'tests/test_reuse_delaunay.py', 'tests/test_large_reuse.py', 'tests/test_support_reuse.py']))
