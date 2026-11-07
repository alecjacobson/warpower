from recovery_diagnostics_witness_v1 import install
install()
import pytest
raise SystemExit(pytest.main(['-q','.work/vtm/test_recovery_diagnostics_witness_v1.py','tests/test_containment_selection.py','tests/test_current_selection.py','tests/test_vtm_controls.py','tests/test_exterior_scanned.py']))
