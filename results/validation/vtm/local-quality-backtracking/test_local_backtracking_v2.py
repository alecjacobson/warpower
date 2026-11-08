"""Private local-retry transactions: bounded failure, stale guards and baseline fallback."""
import importlib
import numpy as np
import pytest
import warp as wp
from warpower import _vtm,_selective
from tests.test_selective import initial,retained_oracle
from test_representable_transactions_v2 import valid_core
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.kernel
def force_quality(enabled:wp.array[int],keep:wp.array[int],quality:wp.array[wp.float64]):
    i=wp.tid()
    if enabled[0]!=0 and keep[i]!=0:quality[i]=wp.float64(0)

@wp.kernel
def force_minimum(enabled:wp.array[int],minimum:wp.array[wp.float64]):
    if enabled[0]!=0:minimum[0]=wp.float64(0)

@pytest.fixture(scope='module')
def fitted():
    c=initial(64,5)
    return c.domain,c.points.numpy().copy()

@pytest.mark.parametrize('version',[1,2,3])
def test_forced_quality_failure_is_bounded_restores_and_recovers(version,fitted,monkeypatch):
    domain,points=fitted
    c=_vtm.VTMWorkspace(domain,points.copy(),max_iterations=1,max_backtracks=2,selective_threshold=1.0)
    importlib.import_module(f'local_backtracking_v{version}').install(c,midpoint=False,budget=2)
    enabled=wp.ones(1,dtype=int);s=c.selection
    def check():
        s.evaluate()
        wp.launch(force_quality,len(s.keep),inputs=[enabled,s.keep,s.quality])
        wp.launch(force_minimum,1,inputs=[enabled,s.minimum])
        wp.launch(_selective.guard_quality,1,inputs=[s.error,s.retained_count,s.minimum,s.previous_minimum,c._accepted,s.rejections,s.last_error])
        wp.launch(_selective.accepted_evaluation,1,inputs=[s.error,c._accepted,s._current_needed])
    s.check=check
    wp.load_module(module=__name__,device=c.device)
    def forbidden(*args,**kwargs):raise AssertionError('host activity during capture/replay')
    with monkeypatch.context() as m:
        for name in ['empty','zeros','ones','full','empty_like','clone','synchronize','synchronize_device','synchronize_stream']:m.setattr(wp,name,forbidden)
        m.setattr(wp.array,'numpy',forbidden)
        with wp.ScopedCapture() as cap:
            c._local_stats.zero_();c.run()
        wp.capture_launch(cap.graph)
    result,_=valid_core(c)
    np.testing.assert_array_equal(result.view(np.uint32),points.view(np.uint32))
    assert c.iterations.numpy()[0]==0 and c.termination.numpy()[0]==2
    assert 3<=c.rejected_steps.numpy()[0]<=5
    assert 0<c._local_stats.numpy()[0]<=2
    # Replay the same graph with the synthetic failure disabled and fresh input.
    c.points.assign(points);enabled.zero_();wp.capture_launch(cap.graph)
    result,_=valid_core(c)
    fresh=_vtm.VTMWorkspace(domain,points.copy(),max_iterations=1,max_backtracks=2,selective_threshold=1.0)
    importlib.import_module(f'local_backtracking_v{version}').install(fresh,midpoint=False,budget=2).run()
    expected,_=valid_core(fresh)
    np.testing.assert_allclose(result,expected,rtol=0,atol=1e-8)
    for name in ['iterations','termination','rejected_steps']:
        np.testing.assert_array_equal(getattr(c,name).numpy(),getattr(fresh,name).numpy())
    assert c._local_used.numpy()[0]<=2
    _,keep,q=retained_oracle(c)
    assert q[keep].min()>=s.previous_minimum.numpy()[0]*(1-1e-9)

@pytest.mark.parametrize('version',[1,2,3])
def test_sticky_quality_reason_does_not_turn_objective_failure_into_local_retry(version,fitted):
    domain,points=fitted
    c=_vtm.VTMWorkspace(domain,points.copy(),max_iterations=1,max_backtracks=2,selective_threshold=1.0)
    importlib.import_module(f'local_backtracking_v{version}').install(c,midpoint=False,budget=2)
    launch=c._launch
    def forced(kernel,dim,args,*,block_dim=256):
        launch(kernel,dim,args,block_dim=block_dim)
        if kernel is _vtm.candidate_ready:
            c._accepted.zero_();c.selection.last_error.fill_(4)
    c._launch=forced
    with wp.ScopedCapture() as cap:c._local_stats.zero_();c.run()
    wp.capture_launch(cap.graph)
    result,_=valid_core(c);np.testing.assert_array_equal(result,points)
    assert c.rejected_steps.numpy()[0]==3 and c._local_stats.numpy()[0]==0
    assert c.selection.rejections.numpy()[0]==0

@pytest.mark.parametrize('version',[1,2,3])
def test_zero_budget_uses_original_transaction_schedule(version,fitted):
    domain,points=fitted
    baseline=_vtm.VTMWorkspace(domain,points.copy(),max_iterations=2,selective_threshold=1.0).run()
    c=_vtm.VTMWorkspace(domain,points.copy(),max_iterations=2,selective_threshold=1.0)
    importlib.import_module(f'local_backtracking_v{version}').install(c,midpoint=False,budget=0).run()
    p,t=valid_core(c);bp,bt=valid_core(baseline)
    np.testing.assert_allclose(p,bp,rtol=0,atol=1e-8)
    assert {tuple(sorted(x)) for x in t}=={tuple(sorted(x)) for x in bt}
    for name in ['iterations','termination','rejected_steps']:
        np.testing.assert_array_equal(getattr(c,name).numpy(),getattr(baseline,name).numpy())
    assert c._local_stats.numpy()[0]==0
