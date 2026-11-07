"""Private current-runtime transactions starting from retained fitted snapshots."""
import argparse
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import warp as wp
from warpower import SurfaceDomain, ExtractionWorkspace
from warpower import _vtm
from representable_boundary_v1 import install

wp.set_module_options({'enable_backward':False,'fast_math':False})


@wp.kernel
def observe(alpha:wp.array[wp.float64],move:wp.array[wp.float64],accepted:wp.array[int],
            energy:wp.array[wp.float64],old_energy:wp.array[wp.float64],
            boundary:wp.array[wp.float64],old_boundary:wp.array[wp.float64],
            stats:wp.array[int],count:wp.array[int],rows:wp.array2d[wp.float64]):
    at=count[0]
    count[0]+=1
    if at<rows.shape[0]:
        rows[at,0]=alpha[0];rows[at,1]=move[0];rows[at,2]=wp.float64(accepted[0])
        rows[at,3]=energy[0];rows[at,4]=old_energy[0]
        rows[at,5]=boundary[0];rows[at,6]=old_boundary[0]
        rows[at,7]=boundary[1];rows[at,8]=old_boundary[1]
        rows[at,9]=wp.float64(stats[0]);rows[at,10]=wp.float64(stats[1])


p=argparse.ArgumentParser();p.add_argument('--case',choices=['old-hand','current-hand','torus'],required=True)
p.add_argument('--mode',choices=['baseline','candidate'],required=True);p.add_argument('--steps',type=int,default=10)
p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
root=Path.cwd();w=root/'.work/vtm'
paths={'old-hand':w/'stable-center-final-hand-final-candidate.npz',
       'current-hand':w/'surface-bound-reuse-final/thin-hand-after/mesh.npz',
       'torus':w/'surface-bound-reuse-final/torus1000-after.npz'}
path=paths[a.case];data=np.load(path)
subdivisions=20 if a.case=='current-hand' else 4
domain=SurfaceDomain(data['input_points'],data['input_faces'],surface_subdivisions=subdivisions,
                     crease_subdivisions=subdivisions)
assert domain.error.numpy()[0]==0
core=_vtm.VTMWorkspace(domain,data['points'].copy(),max_iterations=a.steps)
if a.mode=='candidate':install(core)
stats=core._tie_stats if a.mode=='candidate' else wp.zeros(4,dtype=int)
count=wp.zeros(1,dtype=int);trace=wp.zeros((a.steps*13,11),dtype=wp.float64)
launch=core._launch
def measured(self,kernel,dim,args,*,block_dim=256):
    if kernel is _vtm.finish_attempt:
        launch(observe,1,[self._alpha,self._max_move,self._accepted,self.energy,
                         self._old_energy,self.boundary.energy,self._old_boundary,stats,count,trace])
    launch(kernel,dim,args,block_dim=block_dim)
core._launch=types.MethodType(measured,core)
extract=ExtractionWorkspace(domain,core.points,core.full_tetrahedra,core.full_tetrahedra_count,
                            source_error=core.error)
wp.load_module(module=__name__)
with wp.ScopedCapture() as cap:
    domain.refit();stats.zero_();count.zero_();trace.zero_();core.run();extract.run()
wp.capture_launch(cap.graph);wp.synchronize()
calls=int(count.numpy()[0]);assert calls<=len(trace)
points=core.points.numpy();n=int(core.full_tetrahedra_count.numpy()[0]);m=int(extract.tetrahedra_count.numpy()[0])
arrays=dict(points=points,input_points=data['input_points'],input_faces=data['input_faces'],
            full_tetrahedra=core.full_tetrahedra.numpy()[:n],tetrahedra=extract.tetrahedra.numpy()[:m],
            roles=core.boundary.roles.numpy(),targets=core.boundary.targets.numpy(),
            last_original=core._original.numpy(),last_direction=core._direction.numpy(),
            boundary=extract.boundary_triangles.numpy()[:int(extract.boundary_triangle_count.numpy()[0])])
if a.mode=='candidate':arrays['last_frozen_targets']=core._tie_targets.numpy()
mesh=a.output.with_suffix('.npz');assert not mesh.exists();np.savez_compressed(mesh,**arrays)
record=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            scope='Instrumented normal core proposal/reconstruction/acceptance/backtracking from a fitted snapshot, followed by ordinary extraction. Not a complete fresh solver or a performance measurement.',
            parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
            input=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),w/'representable_boundary_v1.py']},
            domain_error=int(domain.error.numpy()[0]),core_error=int(core.error.numpy()[0]),
            mesh_error=int(core.mesh.error.numpy()[0]),boundary_error=int(core.boundary.error.numpy()[0]),
            iterations=int(core.iterations.numpy()[0]),termination=int(core.termination.numpy()[0]),
            rejected_steps=int(core.rejected_steps.numpy()[0]),last_geometry_error=int(core.last_geometry_error.numpy()[0]),
            last_boundary_error=int(core.last_boundary_error.numpy()[0]),stats=stats.numpy().tolist(),
            stats_columns=['proposed_components','proposed_sites','accepted_components','accepted_sites'],
            trace_columns=['alpha','maximum_normalized_movement','accepted','energy','old_energy',
                           'surface','old_surface','crease','old_crease','cumulative_proposed_components','cumulative_proposed_sites'],
            trace=trace.numpy()[:calls].tolist(),energy_history=core.energy_history.numpy()[:int(core.iterations.numpy()[0])+1].tolist(),
            extraction_error=int(extract.error.numpy()[0]),extraction_valid=int(extract.valid.numpy()[0]),
            extraction_diagnostics=extract.diagnostics.numpy().tolist(),full_count=n,material_count=m,
            minimum_material_radius_ratio=float(extract.minimum_radius_ratio.numpy()[0]),
            surface_rms_distance=float(extract.surface_rms_distance.numpy()[0]),
            mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest())
a.output.write_text(json.dumps(record,indent=2)+'\n')
print({k:v for k,v in record.items() if k not in ['trace','energy_history','source_sha256']},flush=True)
