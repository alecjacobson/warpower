"""Private #228 full-solve exhaustion, invalid producer inputs and exact recovery loop."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
import warp as wp
import tangential_topology_recovery_v2 as prototype
prototype.install()
from warpower import SurfaceDomain
from tests.test_vtm_core import consume

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def produce(source:wp.array[wp.vec3],points:wp.array[wp.vec3],phase:wp.array[int],target:wp.array[int],short:wp.array[int],save:wp.array[int]):
    i=wp.tid();p=source[i]
    if phase[0]==3:p=wp.vec3(0.)
    points[i]=p
    if i==0:
        target[0]=8192
        if phase[0]==2:target[0]=8193
        short[0]=int(phase[0]==0);save[0]=int(phase[0]==0 or phase[0]==1 or phase[0]==4)

@wp.kernel
def save_points(phase:wp.array[int],save:wp.array[int],active:wp.array[int],points:wp.array[wp.vec3],old:wp.array[wp.vec3],roles:wp.array[int],out:wp.array2d[wp.vec3],baseline:wp.array2d[wp.vec3],saved_roles:wp.array2d[int]):
    i=wp.tid();r=phase[0]
    if save[0]!=0 and i<active[0]:
        out[r,i]=points[i];baseline[r,i]=old[i];saved_roles[r,i]=roles[i]

@wp.kernel
def save_cells(phase:wp.array[int],save:wp.array[int],count:wp.array[int],cells:wp.array2d[int],out:wp.array3d[int]):
    i=wp.tid()
    if save[0]!=0 and i<count[0]:
        for k in range(cells.shape[1]):out[phase[0],i,k]=cells[i,k]

@wp.kernel
def finish(phase:wp.array[int],condition:wp.array[int],error:wp.array[int],valid:wp.array[int],counts:wp.array[int],topology:wp.array[int],active:wp.array[int],core_count:wp.array[int],core_error:wp.array[int],mesh_error:wp.array[int],accepted:wp.array[int],attempts:wp.array[int],invoked:wp.array[int],chosen:wp.array[int],recovery_error:wp.array[int],recovery_status:wp.array[int],old_count:wp.array[int],volume:wp.array[wp.float64],reported:wp.array[wp.float64],rows:wp.array2d[int],values:wp.array2d[wp.float64]):
    r=phase[0];rows[r,0]=error[0];rows[r,1]=valid[0]
    for k in range(4):rows[r,2+k]=counts[k];rows[r,6+k]=topology[k]
    rows[r,10]=active[0];rows[r,11]=core_count[0];rows[r,12]=core_error[0];rows[r,13]=mesh_error[0]
    rows[r,14]=accepted[0];rows[r,15]=attempts[0];rows[r,16]=invoked[0];rows[r,17]=chosen[0]
    rows[r,18]=recovery_error[0];rows[r,19]=recovery_status[0];rows[r,20]=old_count[0]
    values[r,0]=volume[0];values[r,1]=reported[0]
    phase[0]=r+1;condition[0]=int(r<4)


def test_exhaustion_then_success_invalid_target_domain_and_recovery(monkeypatch):
    w=Path('.work/vtm');out=w/'tangential-rollback-loop-v1';assert not out.exists();out.mkdir()
    source_path=w/'uniform-armadillo-8192-transaction-v1/failed-candidate.npz'
    with np.load(source_path) as d:input_points=d['input_points'];input_faces=d['input_faces']
    domain=SurfaceDomain(input_points,input_faces,surface_subdivisions=1,crease_subdivisions=1,feature_angle=120,corner_angle=45)
    source=wp.clone(domain.points);target=wp.full(1,8192,dtype=int,device=domain.device)
    m=prototype.Mesher(domain,8192,initial_count=256,target_count=target,seed=2026,quality_rounds=0)
    r=m.topology_recovery;c=m.core
    phase=wp.zeros(1,dtype=int,device=domain.device);condition=wp.ones(1,dtype=int,device=domain.device)
    short=wp.zeros_like(phase);save=wp.zeros_like(phase);rows=wp.empty((5,21),dtype=int,device=domain.device)
    volume=wp.zeros(1,dtype=wp.float64,device=domain.device);values=wp.zeros((5,2),dtype=wp.float64,device=domain.device)
    points=wp.empty((5,8192),dtype=wp.vec3,device=domain.device);baseline=wp.empty_like(points);roles=wp.empty((5,8192),dtype=int,device=domain.device)
    full=wp.empty((5,len(m.full_tetrahedra),4),dtype=int,device=domain.device);old_full=wp.empty_like(full)
    material=wp.empty((5,len(m.tetrahedra),4),dtype=int,device=domain.device);boundary=wp.empty((5,len(m.boundary_faces),3),dtype=int,device=domain.device)
    wp.load_module(module=__name__,device=domain.device);wp.load_module(module='tests.test_vtm_core',device=domain.device)
    pointers=[a.ptr for a in [m.points,m.full_tetrahedra,m.tetrahedra,m.boundary_faces,m.counts]]
    stream=wp.Stream(domain.device);wp.synchronize_device(domain.device)
    def exhausted():
        # Setup-time specialization of one graph branch; replay selects on device.
        r._state.max_attempts=1;m.run();r._state.max_attempts=32
    def step():
        wp.launch(produce,len(source),inputs=[source,domain.points,phase,target,short,save],device=domain.device)
        domain.refit();wp.capture_if(short,exhausted,m.run)
        volume.zero_();wp.launch(consume,len(m.tetrahedra),inputs=[m.points,m.tetrahedra,m.tetrahedra_count,m.error,volume],device=domain.device)
        wp.launch(save_points,8192,inputs=[phase,save,m.active_count,m.points,r._old_points,c.boundary.roles,points,baseline,roles],device=domain.device)
        for count,cells,dest in [(c.full_tetrahedra_count,m.full_tetrahedra,full),(r._old_count,r._old_tets,old_full),(m.tetrahedra_count,m.tetrahedra,material),(m.boundary_count,m.boundary_faces,boundary)]:
            wp.launch(save_cells,len(cells),inputs=[phase,save,count,cells,dest],device=domain.device)
        wp.launch(finish,1,inputs=[phase,condition,m.error,m.valid,m.counts,m.topology,m.active_count,c.full_tetrahedra_count,c.error,c.mesh.error,r.accepted,r.attempts,r.invoked,r.chosen,r.error,r.status,r._old_count,volume,m.volume,rows,values],device=domain.device)
    def forbidden(*args,**kwargs):raise AssertionError('host readback/allocation/synchronization during capture/replay')
    with monkeypatch.context() as patch:
        for name in ['empty','zeros','ones','full','empty_like','zeros_like','clone','synchronize','synchronize_device','synchronize_stream']:patch.setattr(wp,name,forbidden)
        patch.setattr(wp.array,'numpy',forbidden)
        with wp.ScopedStream(stream),wp.ScopedCapture(stream=stream) as cap:
            phase.zero_();condition.fill_(1);wp.capture_while(condition,step)
        wp.capture_launch(cap.graph,stream=stream)
    wp.synchronize_stream(stream)
    result=rows.numpy();consumed=values.numpy();p=points.numpy();b=baseline.numpy();rr=roles.numpy();f=full.numpy();old=old_full.numpy();t=material.numpy();bf=boundary.numpy()
    report=dict(scope=__doc__,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),prototype_sha256=hashlib.sha256((w/'tangential_topology_recovery_v2.py').read_bytes()).hexdigest(),rows=result.tolist(),values=consumed.tolist(),stable_pointers=pointers==[a.ptr for a in [m.points,m.full_tetrahedra,m.tetrahedra,m.boundary_faces,m.counts]],phases=['one-attempt-exhaustion','accepted','invalid-target8193','collapsed-domain','recovered'])
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
    for i in [0,1,4]:
        n,nf=int(result[i,10]),int(result[i,11]);nm,nb=int(result[i,4]),int(result[i,5]);no=int(result[i,20])
        np.savez_compressed(out/f'frame{i}.npz',points=p[i,:n],baseline_points=b[i,:n],roles=rr[i,:n],full_tetrahedra=f[i,:nf],baseline_full_tetrahedra=old[i,:no],tetrahedra=t[i,:nm],boundary=bf[i,:nb],input_points=input_points,input_faces=input_faces)
    np.testing.assert_array_equal(result[:,0],[64,0,1,1,0]);np.testing.assert_array_equal(result[:,1],[0,1,0,0,1])
    np.testing.assert_array_equal(result[[0,2,3],2:10],0);np.testing.assert_array_equal(consumed[[0,2,3]],0)
    np.testing.assert_array_equal(result[:,14],[0,1,0,0,1]);np.testing.assert_array_equal(result[:,15],[1,22,0,0,22]);np.testing.assert_array_equal(result[:,16],[1,1,0,0,1])
    np.testing.assert_array_equal(result[:,17],[3406,3406,-1,-1,3406]);assert result[0,12]==result[0,13]==result[0,18]==0 and result[0,19]==2
    np.testing.assert_array_equal(p[0].view(np.uint32),b[0].view(np.uint32));np.testing.assert_array_equal(p[1].view(np.uint32),p[4].view(np.uint32))
    np.testing.assert_array_equal(result[1,2:10],result[4,2:10]);assert report['stable_pointers']
    np.testing.assert_allclose(consumed[:,0],consumed[:,1],rtol=1e-11);assert np.all(consumed[[1,4]]>0)
