from recovery_diagnostics_witness_v1 import install
install()
import pytest
raise SystemExit(pytest.main(['-q','.work/vtm/test_recovery_diagnostics_full_oracle_v1.py']))
