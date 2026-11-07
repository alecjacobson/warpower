from recovery_classification_inside_v1 import install
install()
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','.work/vtm/test_recovery_classification_inside_v2.py','tests/test_selection_extraction.py']))
