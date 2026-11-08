"""Exercise the unchanged nineteen ordered-query contracts with one private register cap."""
import argparse
import hashlib
import json
from pathlib import Path
import pytest
import warp as wp
from surface_register_limits_v1 import install

p=argparse.ArgumentParser();p.add_argument('--limit',type=int,required=True,choices=[96,128,192]);p.add_argument('--output',type=Path,required=True);args=p.parse_args();assert not args.output.exists()
wp.init()
properties=install(args.limit)
w=Path('.work/vtm')
paths=[Path(__file__),w/'surface_register_limits_v1.py',w/f'surface_register_limit_{args.limit}_v1.py',Path('tests/test_ordered_surface_queries.py'),Path('warpower/_ordered_surface_queries.py')]
report=dict(scope=__doc__,limit=args.limit,properties=properties,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},complete=False)
args.output.write_text(json.dumps(report,indent=2)+'\n')
code=int(pytest.main(['-q','tests/test_ordered_surface_queries.py']))
report.update(complete=True,exit_code=code,passed=code==0)
args.output.write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(code)
