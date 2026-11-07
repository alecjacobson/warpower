"""Run an unmodified gallery entry with an optional private proposal patch."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import warp as wp
from warpower import _vtm,_mesher
from representable_boundary_v1 import install

p=argparse.ArgumentParser(add_help=False);p.add_argument('--pilot-mode',choices=['baseline','candidate'],required=True)
a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
out=Path(rest[rest.index('--output')+1]);registry=[]
initialize=_vtm.VTMWorkspace.__init__
def build(self,*args,**kwargs):
    initialize(self,*args,**kwargs)
    if a.pilot_mode=='candidate':install(self)
    registry.append(self)
_vtm.VTMWorkspace.__init__=build
record=_mesher.VariationalMeshingWorkspace._record
def capture_record(self):
    if a.pilot_mode=='candidate':self.core._tie_stats.zero_()
    record(self)
_mesher.VariationalMeshingWorkspace._record=capture_record
from examples.vtm_gallery import main
try:
    main()
finally:
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in registry:
        row=dict(core_error=int(c.error.numpy()[0]),mesh_error=int(c.mesh.error.numpy()[0]),
                 last_iterations=int(c.iterations.numpy()[0]),last_termination=int(c.termination.numpy()[0]))
        if a.pilot_mode=='candidate':row['full_solve_correction_stats']=c._tie_stats.numpy().tolist()
        rows.append(row)
    result=out/'result.json'
    if registry and result.exists() and not json.loads(result.read_text())['valid']:
        c=registry[-1]
        np.savez_compressed(out/'failed-state.npz',points=c.points.numpy(),active_count=c.active_count.numpy(),
                            full_tetrahedra=c.full_tetrahedra.numpy()[:int(c.full_tetrahedra_count.numpy()[0])],
                            input_points=c.domain.points.numpy(),input_faces=c.domain._indices.numpy().reshape(-1,3),
                            roles=c.boundary.roles.numpy())
    data=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),mode=a.pilot_mode,
              scope='Private complete gallery solve using ordinary initialization, growth, quality and failure/rollback paths. Candidate includes diagnostic atomics and acceptance counters; no production speed claim.',
              command=sys.argv,cores=rows,stats_columns=['proposed_components','proposed_sites','accepted_components','accepted_sites'],
              source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('.work/vtm/representable_boundary_v1.py')]})
    path=out/'pilot.json';assert not path.exists();path.write_text(json.dumps(data,indent=2)+'\n')
