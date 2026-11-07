import pytest
from whole_previous_certificate import install
install()
raise SystemExit(pytest.main(['-q','.work/vtm/test_previous_complete_replay.py']))
