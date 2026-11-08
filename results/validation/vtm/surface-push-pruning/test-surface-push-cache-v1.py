"""Check pruned cached/ordered classifiers against unmodified ordinary callers."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
import pytest
from surface_push_pruning_install_v1 import install
from warpower import _extraction, _selective
from warpower._surface_bvh import closest_point

w=Path(__file__).parent; out=w/'surface-push-pruning-cache-contracts-v1.json'; assert not out.exists()
paths=[Path(__file__),w/'surface_push_pruning_v1.py',w/'surface_push_pruning_install_v1.py',*Path('warpower').glob('*.py'),Path('tests/test_surface_query_cache.py'),Path('tests/test_ordered_surface_queries.py')]
report=dict(scope=__doc__,complete=False,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
out.write_text(json.dumps(report,indent=2)+'\n')
install()
# Preserve ordinary reference arithmetic and traversal in this process.
_extraction.closest_point=closest_point
_selective.closest_point=closest_point
start=time.time()
code=int(pytest.main(['-q','tests/test_surface_query_cache.py','tests/test_ordered_surface_queries.py']))
report.update(complete=True,passed=code==0,exit_code=code,wall_seconds=time.time()-start)
out.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(code)
