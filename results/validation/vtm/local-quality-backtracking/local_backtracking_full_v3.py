"""Fresh gallery pilot with private local retries and optional midpoint correction."""
import argparse,hashlib,json,sys
from pathlib import Path
import warp as wp
from warpower import _vtm
from local_backtracking_v3 import install
p=argparse.ArgumentParser(add_help=False);p.add_argument('--local-midpoint',type=int,choices=[0,1],required=True);p.add_argument('--local-budget',type=int,default=12);a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
original=_vtm.VTMWorkspace.__init__;cores=[]
def patched(self,*args,**kwargs):
    original(self,*args,**kwargs);install(self,midpoint=bool(a.local_midpoint),budget=a.local_budget);cores.append(self)
_vtm.VTMWorkspace.__init__=patched
run=_vtm.VTMWorkspace.run
# Gallery calls mesh.run inside capture, so reset diagnostics once at the start
# of each full mesh invocation rather than each stage's core invocation.
from warpower import VariationalMeshingWorkspace
fullrun=VariationalMeshingWorkspace.run
def measured(self,*args,**kwargs):
    self.core._local_stats.zero_();self.core._tie_stats.zero_();return fullrun(self,*args,**kwargs)
VariationalMeshingWorkspace.run=measured
from examples.vtm_gallery import main
try:main()
finally:
    output=Path(rest[rest.index('--output')+1]);output.mkdir(parents=True,exist_ok=True)
    record=dict(parameters=vars(a),source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,'.work/vtm/local_backtracking_v3.py','.work/vtm/representable_boundary_v1.py']},
                local_columns=['local_retries','blocked_sites','global_retries','core_accepts_after_local'],
                cores=[dict(local=c._local_stats.numpy().tolist(),midpoint=c._tie_stats.numpy().tolist()) for c in cores])
    (output/'pilot.json').write_text(json.dumps(record,indent=2)+'\n')
