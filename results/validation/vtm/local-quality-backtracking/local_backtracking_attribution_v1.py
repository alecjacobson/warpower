"""Snapshot the first selective proposal and limiting cells, without changing decisions."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
from warpower import SurfaceDomain,VariationalMeshingWorkspace,_vtm
from examples.vtm_inputs import surface
from representable_boundary_v1 import install
wp.set_module_options({'enable_backward':False,'fast_math':False})
@wp.kernel
def choose(mode:wp.array[int],rounds:wp.array[int],iterations:wp.array[int],backtracks:wp.array[int],flag:wp.array[int]):
    flag[0]=int(mode[0]!=0 and rounds[2]==0 and iterations[0]==0 and backtracks[0]==0)

w=Path('.work/vtm');stem=w/'local-backtracking-attribution-v1';assert not stem.with_suffix('.json').exists()
domain=SurfaceDomain(*surface('thin-hand'),surface_subdivisions=20,crease_subdivisions=20,feature_angle=45,corner_angle=45)
mesh=VariationalMeshingWorkspace(domain,1000,initial_count=256,seed=2026,quality_rounds=4)
c=mesh.core;install(c);s=c.selection
sources=dict(points=c.points,original=c._original,direction=c._direction,edge=c._edge,mask=s.mask,
             frozen_targets=c._tie_targets,roles=c.boundary.roles,tets=c.full_tetrahedra,count=c.full_tetrahedra_count,
             old_tets=c._old_tets,old_count=c._old_count,keep=s.keep,quality=s.quality,minimum=s.minimum,
             previous_minimum=s.previous_minimum,guard_error=s.last_error,accepted=c._accepted,
             boundary=c.boundary.energy,old_boundary=c._old_boundary,energy=c.energy,old_energy=c._old_energy)
snap={k:wp.empty_like(v) for k,v in sources.items()};flag=wp.zeros(1,dtype=int);hits=wp.zeros(1,dtype=int)
def record():
    for k,v in sources.items():wp.copy(snap[k],v)
    hits.fill_(1)
launch=c._launch
def measured(kernel,dim,args,*,block_dim=256):
    if kernel is _vtm.finish_attempt:
        launch(choose,1,[mesh._mode,mesh.rounds,c.iterations,c._backtracks,flag])
        wp.capture_if(flag,record)
    launch(kernel,dim,args,block_dim=block_dim)
c._launch=measured
wp.load_module(module=__name__,device=domain.device)
with wp.ScopedCapture() as capture:
    domain.refit();hits.zero_();mesh.run()
wp.capture_launch(capture.graph);wp.synchronize()
assert hits.numpy()[0]==1 and mesh.valid.numpy()[0]==1 and mesh.error.numpy()[0]==0
a={k:v.numpy() for k,v in snap.items()}
a['tets']=a['tets'][:int(a['count'][0])];a['old_tets']=a['old_tets'][:int(a['old_count'][0])]
a['quality']=a['quality'][:len(a['tets'])];a['keep']=a['keep'][:len(a['tets'])]
np.savez_compressed(stem.with_suffix('.npz'),**a)
threshold=float(a['previous_minimum'][0])*(1-1e-10)
bad=np.flatnonzero((a['keep']!=0)&(a['quality']<threshold));ids=np.unique(a['tets'][bad]);mov=np.linalg.norm(a['points'].astype(float)-a['original'],axis=1)
oldset={tuple(sorted(t)) for t in a['old_tets'].tolist()}
rows=[]
for i in bad:
 t=a['tets'][i]
 rows.append(dict(index=int(i),vertices=t.tolist(),quality=float(a['quality'][i]),
                  old_cell=tuple(sorted(t.tolist())) in oldset,selected=a['mask'][t].tolist(),
                  roles=a['roles'][t].tolist(),moved=mov[t].tolist(),distance_to403=np.linalg.norm(a['original'][t].astype(float)-a['original'][403],axis=1).tolist()))
r=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probe_sha256=hashlib.sha256((w/'representable_boundary_v1.py').read_bytes()).hexdigest(),
       snapshot_sha256=hashlib.sha256(stem.with_suffix('.npz').read_bytes()).hexdigest(),scope='First selective proposal of fresh dense hand; observers read only. No timing claim.',
       valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),steps=mesh.steps.numpy().tolist(),moves=mesh.moves.numpy().tolist(),
       previous_minimum=float(a['previous_minimum'][0]),candidate_minimum=float(a['minimum'][0]),guard_error=int(a['guard_error'][0]),bad_cell_count=len(bad),bad_vertices=ids.tolist(),
       focus403_in_bad_cells=bool(403 in ids),bad_cells=rows)
stem.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
