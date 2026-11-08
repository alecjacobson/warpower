"""Run private arithmetic tests and unchanged public extraction contracts."""
import hashlib
import json
from pathlib import Path
import pytest
from surface_sample_parallel_install_v1 import install

w=Path(__file__).parent
p=w/'surface-sample-parallel-contracts-v1.json';assert not p.exists()
paths=[Path(__file__), w/'surface_sample_parallel_v1.py', w/'surface_sample_parallel_install_v1.py', w/'test_surface_sample_parallel_v1.py', Path('warpower/_extraction.py'), Path('tests/test_extraction.py'), Path('tests/test_selection_extraction.py')]
report=dict(scope=__doc__,complete=False,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
p.write_text(json.dumps(report,indent=2)+'\n')
install()
code=int(pytest.main(['-q',str(w/'test_surface_sample_parallel_v1.py'),'tests/test_extraction.py','tests/test_selection_extraction.py']))
report.update(complete=True,exit_code=code,passed=code==0);p.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(code)
