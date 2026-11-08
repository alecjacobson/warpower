"""Original analytic cases plus exhaustive-proof differential tests and captured refit."""
import numpy as np
import pytest
import warp as wp
import tests.test_exterior_proof as original
from tests.test_exterior_proof import *
from warpower._surface_bvh import SurfaceHierarchy,SurfaceTree
from exterior_ray_bvh_v1 import exterior

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def prove_bvh(tree:SurfaceTree,points:wp.array[wp.vec3],tets:wp.array2d[int],result:wp.array[int]):
    i=wp.tid();result[i]=exterior(tree,points,wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3]))

@pytest.fixture(autouse=True)
def compare_original_results(monkeypatch):
    old=original.results
    def compare(vertices,faces,cells):
        expected=old(vertices,faces,cells)
        p=wp.array(vertices,dtype=wp.vec3,device='cuda:0');f=wp.array(faces.ravel(),dtype=int,device='cuda:0')
        hierarchy=SurfaceHierarchy(p,f)
        points=wp.array(cells.reshape(-1,3),dtype=wp.vec3,device='cuda:0')
        tets=wp.array(np.arange(len(cells)*4,dtype=np.int32).reshape(-1,4),dtype=int,device='cuda:0')
        actual=wp.empty(len(cells),dtype=int,device='cuda:0')
        wp.load_module(module=__name__,device='cuda:0')
        with wp.ScopedCapture() as cap:
            hierarchy.refit()
            wp.launch(prove_bvh,len(cells),inputs=[hierarchy.tree,points,tets,actual],block_dim=32)
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(actual.numpy(),expected)
        return actual.numpy()
    monkeypatch.setattr(original,'results',compare)

@pytest.mark.parametrize('exponent',[-140,-100,0,100,125])
def test_random_cells_match_exhaustive_plane_scan(exponent):
    rng=np.random.default_rng(239)
    vertices,faces=original.cube()
    cells=original.positive(rng.uniform(-3,3,size=(256,4,3)))
    scale=np.float32(2.**exponent)
    original.results(vertices*scale,faces,cells*scale)

@wp.kernel
def deform(source:wp.array[wp.vec3],dest:wp.array[wp.vec3],offset:wp.array[float]):
    i=wp.tid();dest[i]=source[i]+wp.vec3(offset[0],0.,0.)

@pytest.mark.parametrize('faces_count',[1,3,4,5,12])
def test_nondefault_stream_refit_updates_bounds_and_odd_leaf_counts(faces_count):
    v,f=original.cube();f=f[:faces_count]
    p=wp.array(v,dtype=wp.vec3,device='cuda:0');source=wp.clone(p)
    faces=wp.array(f.ravel(),dtype=int,device='cuda:0');h=SurfaceHierarchy(p,faces)
    cells=original.positive(np.random.default_rng(14).uniform(-3,3,size=(64,4,3)))
    points=wp.array(cells.reshape(-1,3),dtype=wp.vec3,device='cuda:0')
    tets=wp.array(np.arange(256,dtype=np.int32).reshape(-1,4),dtype=int,device='cuda:0')
    old=wp.empty(64,dtype=int,device='cuda:0');new=wp.empty_like(old)
    offset=wp.zeros(1,dtype=float,device='cuda:0');stream=wp.Stream('cuda:0')
    wp.load_module(module=__name__,device='cuda:0');wp.load_module(module=original.prove.module,device='cuda:0')
    wp.synchronize()
    with wp.ScopedStream(stream),wp.ScopedCapture(stream=stream) as cap:
        wp.launch(deform,len(v),inputs=[source,p,offset]);h.refit()
        wp.launch(original.prove,64,inputs=[h.tree,points,tets,old],block_dim=32)
        wp.launch(prove_bvh,64,inputs=[h.tree,points,tets,new],block_dim=32)
    for value in [0.,100.,-100.,0.]:
        offset.assign(np.array([value],np.float32));wp.synchronize()
        wp.capture_launch(cap.graph,stream=stream);wp.synchronize_stream(stream)
        np.testing.assert_array_equal(new.numpy(),old.numpy())
