"""Compare the shortened private extraction with the unchanged full extractor."""
import numpy as np
import pytest
import warp as wp
from tests.test_current_selection import recovered_selection,produce_points,produce_controls

@pytest.mark.parametrize('value',range(6))
def test_private_status_outputs_match_full_extraction_after_device_changes(value):
    c,s=recovered_selection();e=s.recovery_extraction
    base=wp.clone(c.points);surface=wp.clone(c.domain.points)
    phase=wp.full(1,value,dtype=int,device=c.device)
    fields=[e.error,e.valid,e.tetrahedra_count,e.boundary_count,e.selected,e._input_quality]
    copies=[wp.empty_like(a) for a in fields]
    integral,maximum=wp.empty_like(e._surface_integral),wp.empty_like(e.surface_max_sample_distance)
    wp.load_module(module='tests.test_current_selection',device=c.device)
    with wp.ScopedCapture() as capture:
        wp.launch(produce_points,c.n,inputs=[base,c.points,phase,True],device=c.device)
        wp.launch(produce_points,len(surface),inputs=[surface,c.domain.points,phase,False],device=c.device)
        wp.launch(produce_controls,1,inputs=[phase,c.material_recovery,c.mesh.error],device=c.device)
        c.material_recovery.fill_(1)
        c.domain.refit();e._surface_integral.fill_(77);e.surface_max_sample_distance.fill_(77)
        s.evaluate()  # The installed private prototype is active only here.
        for a,b in zip(fields,copies):wp.copy(b,a)
        wp.copy(integral,e._surface_integral);wp.copy(maximum,e.surface_max_sample_distance)
        e.run()  # Its ordinary public entry still computes all diagnostics.
    wp.capture_launch(capture.graph)
    for i,(a,b) in enumerate(zip(fields,copies)):
        if i==5 and e.error.numpy()[0]!=0:continue
        np.testing.assert_array_equal(a.numpy(),b.numpy())
    assert integral.numpy()[0]==maximum.numpy()[0]==0
    if value in [2,4]:assert e.error.numpy()[0]!=0 and e.valid.numpy()[0]==0
    else:assert e.error.numpy()[0]==0 and e.valid.numpy()[0]==1
