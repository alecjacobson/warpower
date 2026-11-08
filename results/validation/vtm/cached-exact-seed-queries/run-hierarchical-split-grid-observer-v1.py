"""Diagnostic fallback incidence on the integer grid; not clean timing evidence."""
import hashlib,json,runpy,sys
from pathlib import Path
import hierarchical_split_v1 as base
from hierarchical_split_observer_v1 import install,instances
base.install=install
w=Path('.work/vtm');out=w/'hierarchical-split-grid-observer-v1.json'
sys.argv=['grid-observer','--candidate','--repeats','1','--output',str(out)]
runpy.run_path(str(w/'bench-hierarchical-split-grid-v1.py'),run_name='__main__')
r=dict(scope=__doc__,fields=['calls','cell_instances','fallback_cells','fallback_lanes'],rows=[dict(points=len(x._check_inputs[0]),capacity=x.capacity,statistics=x.stats.numpy().tolist()) for x in instances],source_sha256={n:hashlib.sha256((w/n).read_bytes()).hexdigest() for n in ['hierarchical_split_v1.py','hierarchical_split_observer_v1.py','bench-hierarchical-split-grid-v1.py','run-hierarchical-split-grid-observer-v1.py']})
(w/'hierarchical-split-grid-observer-v1-counts.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
