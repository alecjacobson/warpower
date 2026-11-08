"""Strong private nearest contracts and real extraction callers, before timing."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
import pytest
from surface_push_pruning_install_v1 import install

w=Path(__file__).parent
out=w/'surface-push-pruning-contracts-v1.json'
assert not out.exists()
paths=[Path(__file__),w/'surface_push_pruning_v1.py',w/'surface_push_pruning_install_v1.py',w/'test_surface_push_pruning_v1.py',*Path('warpower').glob('*.py'),Path('tests/test_surface_quadrature.py'),Path('tests/surface_quadrature_reference.py'),Path('tests/test_extraction.py'),Path('tests/test_selection_extraction.py')]
report=dict(scope=__doc__,complete=False,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
out.write_text(json.dumps(report,indent=2)+'\n')
install()
start=time.time()
code=int(pytest.main(['-q',str(w/'test_surface_push_pruning_v1.py'),'tests/test_surface_quadrature.py','tests/test_extraction.py','tests/test_selection_extraction.py']))
report.update(complete=True,passed=code==0,exit_code=code,wall_seconds=time.time()-start)
out.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(code)
