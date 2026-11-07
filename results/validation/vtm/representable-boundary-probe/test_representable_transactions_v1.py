"""Real accepted and rejected midpoint moves, including captured prefixes/masks."""
import numpy as np
import warp as wp

from warpower import SurfaceDomain, _vtm, _selective
from tests.test_domain import cube
from tests.test_vtm_core import fail_after_initial, valid_core
from representable_boundary_v1 import install

wp.set_module_options({'enable_backward':False,'fast_math':False})


def fixture():
    corners,faces=cube();scale=np.float32(.12);corners*=scale
    surface=np.concatenate([np.eye(3,dtype=np.float32)*scale,-np.eye(3,dtype=np.float32)*scale])
    surface[0,0]=np.nextafter(scale,np.float32(0))
    source=np.concatenate([corners,surface,np.zeros((1,3),np.float32)])
    assert np.float32((float(source[8,0])+float(scale))*.5)==source[8,0]
    return SurfaceDomain(corners,faces,surface_subdivisions=4),source,scale


def test_accepted_midpoint_reaches_existing_boundary_target():
    domain,source,target=fixture()
    baseline=_vtm.VTMWorkspace(domain,source.copy(),max_iterations=1).run()
    candidate=install(_vtm.VTMWorkspace(domain,source.copy(),max_iterations=1)).run()
    valid_core(baseline);points,_=valid_core(candidate)
    assert baseline.points.numpy()[8,0]==source[8,0]
    assert points[8,0]==target
    assert candidate._tie_targets.numpy()[8,0]==target
    assert candidate._tie_stats.numpy()[2]>0
    np.testing.assert_array_equal(points[:8],source[:8])
    before,after=candidate.energy_history.numpy()[:2]
    assert after[2]<=before[2]*(1+1e-10)+1e-30
    assert after[1]<=before[1]*(1+1e-10)+1e-30


def test_short_edge_cap_does_not_force_endpoint():
    domain,source,_=fixture()
    c=install(_vtm.VTMWorkspace(domain,source.copy(),max_iterations=1,step_limit=1e-9)).run()
    points,_=valid_core(c)
    assert points[8,0]==source[8,0]
    assert c._tie_stats.numpy()[0]==0


@wp.kernel
def record_flags(backtracks:wp.array[int],flags:wp.array[int],rows:wp.array2d[int]):
    i=wp.tid();rows[backtracks[0],i]=flags[i]


def test_rejected_endpoint_backtracks_and_restores():
    domain,source,_=fixture()
    c=install(_vtm.VTMWorkspace(domain,source.copy(),max_iterations=2,max_backtracks=2))
    calls=wp.zeros(1,dtype=int);flags=wp.zeros((3,c.n),dtype=int)
    run=c.mesh.run;launch=c._launch
    def failing_run(stream=None):
        run(stream)
        wp.launch(fail_after_initial,1,inputs=[c.mesh.error,c.full_tetrahedra_count,calls])
    def measured(kernel,dim,args,*,block_dim=256):
        if kernel is _vtm.finish_attempt:
            launch(record_flags,c.n,[c._backtracks,c._tie_flags,flags])
        launch(kernel,dim,args,block_dim=block_dim)
    c.mesh.run=failing_run;c._launch=measured
    wp.load_module(module=__name__,device=c.device)
    with wp.ScopedCapture() as cap:
        calls.zero_();flags.zero_();c._tie_stats.zero_();c.run()
    wp.capture_launch(cap.graph)
    points,_=valid_core(c)
    np.testing.assert_array_equal(points,source)
    assert c.termination.numpy()[0]==2 and c.rejected_steps.numpy()[0]==3
    assert c._tie_stats.numpy()[0]>0 and c._tie_stats.numpy()[2]==0
    assert flags.numpy()[0].sum()>0 and flags.numpy()[1:].sum()==0


def test_trial_boundary_recomputation_cannot_change_frozen_target():
    domain,source,target=fixture()
    c=install(_vtm.VTMWorkspace(domain,source.copy(),max_iterations=1))
    altered=wp.full(c.n,wp.vec3(50),dtype=wp.vec3);launch=c._launch
    def mutate(kernel,dim,args,*,block_dim=256):
        if kernel is _vtm.start_attempt:wp.copy(c.boundary.targets,altered)
        launch(kernel,dim,args,block_dim=block_dim)
    c._launch=mutate
    with wp.ScopedCapture() as cap:c._tie_stats.zero_();c.run()
    wp.capture_launch(cap.graph)
    points,_=valid_core(c)
    assert points[8,0]==target and c._tie_targets.numpy()[8,0]==target
    assert c._tie_stats.numpy()[2]>0


def test_captured_active_prefix_mask_failure_and_recovery():
    domain,source,target=fixture()
    padded=np.concatenate([source,np.full((8,3),np.nan,np.float32)])
    active=wp.array([len(source)],dtype=int);mask=wp.ones(len(padded),dtype=int)
    c=_vtm.VTMWorkspace(domain,padded.copy(),active_count=active,max_iterations=1)
    c._directions=_selective.masked_directions;c._direction_args=[mask]
    install(c)
    with wp.ScopedCapture() as cap:
        domain.refit();c._tie_stats.zero_();c.run()
    for n,enabled in [(15,False),(15,True),(0,True),(15,True)]:
        c.points.assign(padded);active.fill_(n)
        values=np.ones(len(padded),np.int32);values[8]=int(enabled);mask.assign(values)
        wp.capture_launch(cap.graph)
        result=c.points.numpy()
        np.testing.assert_array_equal(result[15:].view(np.uint32),padded[15:].view(np.uint32))
        if n==0:
            assert c.error.numpy()[0]==8 and c.full_tetrahedra_count.numpy()[0]==0
            np.testing.assert_array_equal(result.view(np.uint32),padded.view(np.uint32))
        else:
            assert c.error.numpy()[0]==0 and c.mesh.error.numpy()[0]==0
            assert result[8,0]==(target if enabled else source[8,0])
            np.testing.assert_array_equal(result[:8],source[:8])
