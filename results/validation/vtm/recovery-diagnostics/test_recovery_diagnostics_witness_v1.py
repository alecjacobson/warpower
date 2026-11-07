import numpy as np
import pytest
import warp as wp
import warpower
from warpower import _extraction
from tests.test_domain import cube,torus
from recovery_diagnostics_witness_v1 import surface_status,install

@pytest.mark.parametrize('shape',['cube','torus'])
@pytest.mark.parametrize('scale',[1e-30,1.0,1e30])
@pytest.mark.parametrize('mode',['ordinary','nan_sample','initial_error','missing_witness'])
def test_status_matches_full_surface_queries(shape,scale,mode):
    p,t=cube() if shape=='cube' else torus(12,6)
    p=(p*scale).astype(np.float32)
    domain=warpower.SurfaceDomain(p,t)
    assert domain.error.numpy()[0]==0
    if mode=='missing_witness':
        # Corrupt only the witness leaf's vertices after the valid tree is built.
        # This deliberately forces fallback; full traversal may find other leaves.
        first=int(domain.tree.ids.numpy()[0]);changed=p.copy();changed[t[first]]=np.nan
        wp.copy(domain.points,wp.array(changed,dtype=wp.vec3,device=domain.device))
    query=(np.random.default_rng(192).uniform(-4,4,(96,3))*scale).astype(np.float32)
    if mode=='nan_sample':query[0]=np.nan
    points=wp.array(query,dtype=wp.vec3,device=domain.device);faces=wp.array(np.arange(96,dtype=np.int32).reshape(-1,3),device=domain.device);count=wp.full(1,32,dtype=int,device=domain.device)
    errors=[wp.full(1,4 if mode=='initial_error' else 0,dtype=int,device=domain.device) for _ in range(2)]
    integral=wp.zeros(1,dtype=wp.float64,device=domain.device);maximum=wp.zeros_like(integral)
    for kernel,e in zip([_extraction.surface_error,surface_status],errors):wp.launch(kernel,32,inputs=[domain.tree,points,faces,count,integral,maximum,e],device=domain.device)
    np.testing.assert_array_equal(errors[0].numpy(),errors[1].numpy())
    if mode=='ordinary':assert errors[0].numpy()[0]==0
    if mode=='nan_sample':assert errors[0].numpy()[0]==16
