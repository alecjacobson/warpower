"""Recorded real failure: identical exact proof decisions with padded inactive rows."""
from pathlib import Path
import numpy as np
import warp as wp
from warpower._exterior_proof import exterior
from warpower._proven_exterior import remove_proved
from warpower._surface_bvh import SurfaceHierarchy,SurfaceTree

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def exhaustive(tree:SurfaceTree,points:wp.array[wp.vec3],tets:wp.array2d[int],count:wp.array[int],
               selected:wp.array[int],distance:wp.array[wp.float64],removed:wp.array[int]):
    i=wp.tid()
    if i<count[0] and selected[i]!=0 and distance[i]>wp.float64(0.0):
        ids=wp.vec4i(tets[i,0],tets[i,1],tets[i,2],tets[i,3])
        if exterior(tree,points,ids)!=0:selected[i]=0;wp.atomic_add(removed,0,1)

def test_real_proofs_padded_count_empty_and_recovery():
    path=Path('.work/vtm/uniform-armadillo-8192-transaction-v1/failed-candidate.npz')
    with np.load(path) as d:data={k:d[k] for k in d.files}
    tree_points=wp.array(data['input_points'],dtype=wp.vec3,device='cuda:0')
    tree_faces=wp.array(data['input_faces'].ravel(),dtype=int,device='cuda:0')
    hierarchy=SurfaceHierarchy(tree_points,tree_faces)
    points=wp.array(data['points'],dtype=wp.vec3,device='cuda:0')
    n=len(data['full_tetrahedra']);capacity=n+32
    tets=wp.array(np.vstack([data['full_tetrahedra'],np.full((32,4),-1,np.int32)]),dtype=int,device='cuda:0')
    distance=wp.array(np.r_[data['signed_distance'],np.full(32,np.nan)],dtype=wp.float64,device='cuda:0')
    saved=wp.array(np.r_[data['selected'],np.ones(32,np.int32)],dtype=int,device='cuda:0')
    before=wp.empty_like(saved);after=wp.empty_like(saved)
    a=wp.zeros(1,dtype=int,device='cuda:0');b=wp.zeros_like(a);count=wp.full(1,n,dtype=int,device='cuda:0')
    wp.load_module(module=__name__,device='cuda:0');wp.load_module(module='warpower._proven_exterior',device='cuda:0')
    with wp.ScopedCapture() as cap:
        hierarchy.refit();wp.copy(before,saved);wp.copy(after,saved);a.zero_();b.zero_()
        wp.launch(exhaustive,capacity,inputs=[hierarchy.tree,points,tets,count,before,distance,a])
        wp.launch(remove_proved,capacity,inputs=[hierarchy.tree,points,tets,count,after,distance,b])
    first=None
    for size in [n,0,n]:
        count.assign(np.array([size],np.int32));wp.capture_launch(cap.graph);wp.synchronize()
        np.testing.assert_array_equal(after.numpy(),before.numpy());np.testing.assert_array_equal(a.numpy(),b.numpy())
        np.testing.assert_array_equal(after.numpy()[n:],1)
        if size==0:np.testing.assert_array_equal(after.numpy(),saved.numpy());assert b.numpy()[0]==0
        elif first is None:first=after.numpy().copy()
        else:np.testing.assert_array_equal(after.numpy(),first)
