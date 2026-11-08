"""Trace exhaustion must flag overflow without writing beyond owned storage."""
import numpy as np
import pytest
import warp as wp
from step_observer_v1 import Inputs, snapshot


@pytest.mark.parametrize("phase", [0,2])
def test_bounded_trace_replay_and_unavailable_fields(phase):
    device=wp.get_device("cuda:0")
    z=wp.zeros(2,dtype=int,device=device)
    f=wp.full(2,7,dtype=wp.float64,device=device)
    s=Inputs()
    for name in ["active","mode","budget","iterations","backtracks","rejected","accepted",
                 "attempt","running","termination","core_error","mesh_error","boundary_error",
                 "invalid","full_count","selection_error","selection_count","selection_rejections",
                 "selection_last_error","local_used","local_statistics"]:
        setattr(s,name,z)
    for name in ["alpha","maximum","energy","old_energy","boundary","old_boundary","minimum","previous_minimum"]:
        setattr(s,name,f)
    count=wp.zeros(1,dtype=int,device=device)
    overflow=wp.zeros(1,dtype=int,device=device)
    ints=wp.full((1,25),-7,dtype=int,device=device)
    floats=wp.full((1,10),-7,dtype=wp.float64,device=device)
    wp.load_module(module="step_observer_v1",device=device)
    with wp.ScopedCapture(device=device) as cap:
        count.zero_();overflow.zero_()
        for _ in range(2):
            wp.launch(snapshot,1,inputs=[s,2,0,phase,count,overflow,ints,floats],device=device)
    for _ in range(2):
        wp.capture_launch(cap.graph)
        assert count.numpy()[0]==2 and overflow.numpy()[0]==1
        expected=np.zeros((1,25),dtype=np.int32);expected[0,0]=2;expected[0,2]=phase
        np.testing.assert_array_equal(ints.numpy(),expected)
        if phase==0:assert np.isnan(floats.numpy()).all()
        else:np.testing.assert_array_equal(floats.numpy(),7)
