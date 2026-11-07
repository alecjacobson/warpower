"""Read-only complete-solve checkpoints localize the hand-fit tradeoff."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
from examples.vtm_inputs import surface
from warpower import SurfaceDomain,VariationalMeshingWorkspace
from representable_boundary_v1 import install

@wp.kernel
def reserve(event:int,stage:int,rounds:wp.array[int],active:wp.array[int],valid:wp.array[int],
            error:wp.array[int],rollback:wp.array[int],rms:wp.array[wp.float64],
            minimum:wp.array[wp.float64],boundary_energy:wp.array[wp.float64],
            counter:wp.array[int],labels:wp.array2d[int],metrics:wp.array2d[wp.float64]):
    row=counter[0];counter[0]+=1
    if row<labels.shape[0]:
        labels[row,0]=event;labels[row,1]=stage
        labels[row,2]=rounds[0];labels[row,3]=rounds[1];labels[row,4]=rounds[2]
        labels[row,5]=active[0];labels[row,6]=valid[0];labels[row,7]=error[0];labels[row,8]=rollback[0]
        metrics[row,0]=rms[0];metrics[row,1]=minimum[0]
        metrics[row,2]=boundary_energy[0];metrics[row,3]=boundary_energy[1]

@wp.kernel
def copy_points(counter:wp.array[int],points:wp.array[wp.vec3],targets:wp.array[wp.vec3],
                roles:wp.array[int],positions:wp.array2d[wp.vec3],fitting:wp.array2d[wp.vec3],
                assignments:wp.array2d[int]):
    i=wp.tid();row=counter[0]-1
    if row<positions.shape[0]:
        positions[row,i]=points[i];fitting[row,i]=targets[i];assignments[row,i]=roles[i]

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','candidate'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
domain=SurfaceDomain(*surface('thin-hand'),surface_subdivisions=20,crease_subdivisions=20,feature_angle=45,corner_angle=45)
mesh=VariationalMeshingWorkspace(domain,1000,initial_count=256,seed=2026,quality_rounds=4)
if a.mode=='candidate':install(mesh.core)
counter=wp.zeros(1,dtype=int);labels=wp.zeros((64,9),dtype=int);metrics=wp.zeros((64,4),dtype=wp.float64)
positions=wp.empty((64,1000),dtype=wp.vec3);targets=wp.empty_like(positions);roles=wp.empty((64,1000),dtype=int)
extract,assess=mesh._extract,mesh._assess
def record(event,stage):
    e=mesh.extraction;c=mesh.core
    wp.launch(reserve,1,inputs=[event,stage,mesh.rounds,mesh.active_count,e.valid,e.error,mesh.rollback,e.surface_rms_distance,e.minimum_radius_ratio,c.boundary.energy,counter,labels,metrics])
    wp.launch(copy_points,1000,inputs=[counter,c.points,c.boundary.targets,c.boundary.roles,positions,targets,roles])
def observed_extract():
    extract();record(0,-1)
def observed_assess(stage,error,accepted):
    assess(stage,error,accepted);record(1,stage)
mesh._extract=observed_extract;mesh._assess=observed_assess
wp.load_module(module=__name__,device=domain.device)
with wp.ScopedCapture() as capture:
    domain.refit();counter.zero_()
    if a.mode=='candidate':mesh.core._tie_stats.zero_()
    mesh.run()
wp.capture_launch(capture.graph);wp.synchronize()
n=int(counter.numpy()[0]);assert n<=64
snap=a.output.with_suffix('.npz');assert not snap.exists()
np.savez_compressed(snap,positions=positions.numpy()[:n],targets=targets.numpy()[:n],roles=roles.numpy()[:n],labels=labels.numpy()[:n],metrics=metrics.numpy()[:n])
reference=Path('.work/vtm/representable-full-v1/dense-hand-'+a.mode+'/mesh.npz');d=np.load(reference)
same=bool(np.array_equal(d['points'].view(np.uint32),mesh.points.numpy().view(np.uint32)))
r=dict(runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),mode=a.mode,
       scope='Read-only device checkpoint observers after full extraction and after each outer quality acceptance/rollback. No instrumented timing claim. Boundary objectives are point-sample energies; extraction RMS is directed sampled surface distance.',
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(Path('.work/vtm/representable_boundary_v1.py').read_bytes()).hexdigest(),
       valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),steps=mesh.steps.numpy().tolist(),rounds=mesh.rounds.numpy().tolist(),moves=mesh.moves.numpy().tolist(),
       final_positions_bitwise_equal_to_unobserved_pilot=same,reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),snapshot_sha256=hashlib.sha256(snap.read_bytes()).hexdigest(),
       label_columns=['event_extract0_assess1','stage_selective0_jitter1_interior2','growth_round','recovery_round','quality_round','active','extraction_valid','extraction_error','rollback'],
       metric_columns=['directed_surface_rms','minimum_material_radius_ratio','surface_sample_energy','crease_sample_energy'],labels=labels.numpy()[:n].tolist(),metrics=metrics.numpy()[:n].tolist())
a.output.write_text(json.dumps(r,indent=2)+'\n');print({k:v for k,v in r.items() if k not in ['labels','metrics']},flush=True)
assert r['valid']==1 and r['error']==0
