"""One fixed16k benchmark with pending-mask instrumentation, not a timing sample."""
import hashlib,json,sys
from pathlib import Path
w=Path('.work/vtm');out=w/'hierarchical-split-observer-v2';assert not out.with_suffix('.json').exists()
from hierarchical_split_observer_v2 import install,instances
install()
ref=json.loads((w/'hierarchical-split-pilot-v1/commands.json').read_text())[1]
cmd=ref['command'][:];assert ref['name']=='torus16384-fixed32' and ref['phase']=='after'
cmd=cmd[2:];cmd.remove('--candidate')
cmd[cmd.index('--repeats')+1]='1'
cmd[cmd.index('--output')+1]=str(out.with_suffix('.json'))
cmd[cmd.index('--save-mesh')+1]=str(out.with_suffix('.npz'))
sys.argv=['observer',*cmd]
from benchmarks.mesher import main
main()
rows=[dict(points=len(x._check_inputs[0]),candidate_capacity=x.capacity,statistics=x.stats.numpy().tolist()) for x in instances]
r=dict(scope=__doc__,fields=['calls','cell_instances','fallback_cells','fallback_lanes','cached_lanes','rescan_lanes'],rows=rows,source_sha256={name:hashlib.sha256((w/name).read_bytes()).hexdigest() for name in ['hierarchical_split_v2.py','hierarchical_split_observer_v2.py','run-hierarchical-split-observer-v2.py']})
(w/'hierarchical-split-observer-v2-counts.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
