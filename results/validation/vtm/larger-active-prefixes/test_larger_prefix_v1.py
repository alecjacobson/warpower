"""Captured threshold, ownership and recovery checks for larger private prefixes."""
from itertools import product
import numpy as np
import pytest
import warp as wp
from scipy.spatial import ConvexHull
from larger_prefix_v1 import LargerPrefixDelaunayWorkspace
from tests.test_active_delaunay import advance,consume,produce
from tests.test_delaunay_recovery import exact_empty
from tests.test_tetrahedra import check_mesh,reference,tet_set
from warpower._active_sites import prepare_mesh

@pytest.mark.parametrize('capacity',[8193,16384])
def test_thresholds_current_input_invalid_and_recovery(capacity,monkeypatch,tmp_path):
    counts=[4095,4096,4097,8191,8192,8193,capacity,4096,8192,4095,0,capacity+1,-1,3,4096,8192,4096]
    rng=np.random.default_rng(230)
    initial=rng.normal(size=(capacity,3)).astype(np.float32)
    frames=np.repeat(initial[None],len(counts),axis=0)
    frames[7]=rng.normal(size=(capacity,3)).astype(np.float32)
    frames[8,5]+=np.float32(1e-5)
    frames[14,4,0]=np.nan
    frames[15,:,2]=0
    for row,n in enumerate(counts):
        if 4<=n<=capacity:frames[row,n:]=np.nan
    device='cuda:0';source=wp.array(frames,dtype=wp.vec3,device=device);sizes=wp.array(counts,dtype=int,device=device)
    phase=wp.zeros(1,dtype=int,device=device);points=wp.zeros(capacity,dtype=wp.vec3,device=device);prepared=wp.zeros_like(points);active=wp.zeros(1,dtype=int,device=device)
    w=LargerPrefixDelaunayWorkspace(prepared,active_count=active)
    assert w.capacities==(1024,2048,4096,8192,capacity)
    history=wp.empty((len(counts),len(w.tetrahedra),4),dtype=int,device=device);meta=wp.empty((len(counts),3),dtype=int,device=device)
    for module in ['tests.test_active_delaunay','warpower._active_sites']:wp.load_module(module=module,device=device)
    pointers=(w.tetrahedra.ptr,w.tetrahedra_count.ptr,w.error.ptr,w.points.ptr);stream=wp.Stream(device)
    def forbidden(*args,**kwargs):raise AssertionError('Host allocation/readback/synchronization during capture')
    with monkeypatch.context() as patch:
        for name in ['empty','zeros','ones','full','empty_like','clone','synchronize','synchronize_device','synchronize_stream']:patch.setattr(wp,name,forbidden)
        patch.setattr(wp.array,'numpy',forbidden)
        with wp.ScopedStream(stream),wp.ScopedCapture(stream=stream) as capture:
            wp.launch(produce,w.n,inputs=[source,sizes,phase,points,active])
            wp.launch(prepare_mesh,w.n,inputs=[points,active,prepared])
            w.run(stream)
            wp.launch(consume,len(w.tetrahedra),inputs=[phase,w.tetrahedra,w.tetrahedra_count,w.error,w.selected_capacity,history,meta])
            wp.launch(advance,1,inputs=[phase])
        for _ in counts:wp.capture_launch(capture.graph,stream=stream)
    wp.synchronize_stream(stream);rows,cells=meta.numpy(),history.numpy()
    for i,n in enumerate(counts):
        if i in [10,11,12,13,14,15]:
            assert rows[i,0]==0 and rows[i,1]!=0,(i,rows[i])
            if i in [10,11,12,13]:assert rows[i,1:].tolist()==[8,0]
        else:
            assert rows[i,0]>0 and rows[i,1]==0,(i,rows[i])
            assert rows[i,2]==next(c for c in w.capacities if n<=c)
            cloud=frames[i,:n];actual=cells[i,:rows[i,0]]
            assert actual.min()>=0 and actual.max()<n
            assert len(tet_set(actual))==len(actual) and tet_set(actual)==reference(cloud)
            p=cloud.astype(np.float64)[actual]
            six=np.einsum('ij,ij->i',p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]))
            assert (six>0).all() and np.isclose(six.sum()/6,ConvexHull(cloud).volume,rtol=1e-9)
    assert tet_set(cells[1,:rows[1,0]])==tet_set(cells[-1,:rows[-1,0]])
    assert pointers==(w.tetrahedra.ptr,w.tetrahedra_count.ptr,w.error.ptr,w.points.ptr)
    np.testing.assert_array_equal(points.numpy().view(np.uint32),frames[-1].view(np.uint32))
    with wp.ScopedCapture() as fresh:
        w.clear_reuse();w.run()
    wp.capture_launch(fresh.graph)
    assert w.error.numpy()[0]==0 and w.route_calls.numpy().tolist()==[0,0,1,0,0]
    assert w._children[2].reuse_attempted.numpy()[0]==0
    assert all(c._previous_count.numpy()[0]==0 for j,c in enumerate(w._children) if j!=2)
    assert w._full._previous_count.numpy()[0]==0
    dot=tmp_path/'larger-prefix.dot';wp.capture_debug_dot_print(capture.graph,str(dot),verbose=True)
    assert all(s not in dot.read_text() for s in ['MEM_ALLOC','MEM_FREE','HOST'])

@pytest.mark.parametrize('exponent',[-60,0,60])
def test_ties_duplicate_first_owners_and_original_scale(exponent):
    unique=np.array(list(product([-1,1],repeat=3)),np.float32)*np.float32(2.**exponent)
    small=np.repeat(unique,2,axis=0);cloud=np.tile(small[0],(8193,1));cloud[:16]=small
    points=wp.array(cloud,dtype=wp.vec3,device='cuda:0');active=wp.array([4096],dtype=int,device='cuda:0')
    w=LargerPrefixDelaunayWorkspace(points,active_count=active)
    with wp.ScopedCapture() as cap:w.run()
    prior=None
    for n in [4096,8192,8193,4096]:
        active.assign(np.array([n],np.int32));wp.capture_launch(cap.graph)
        assert w.error.numpy()[0]==0
        cells=w.tetrahedra.numpy()[:w.tetrahedra_count.numpy()[0]]
        assert (cells%2==0).all() and cells.max()<16
        # Rescale exactly for independent volume tests (the original inputs stay unchanged).
        check_mesh(small/np.float32(2.**exponent),cells);exact_empty(small,cells)
        if prior is not None:assert tet_set(cells)==prior
        prior=tet_set(cells)
    np.testing.assert_array_equal(points.numpy().view(np.uint32),cloud.view(np.uint32))

@pytest.mark.parametrize('active_n',[4096,8192])
def test_large_child_failure_uses_full_constructor_and_recovers(active_n):
    cloud=np.random.default_rng(231).normal(size=(8193,3)).astype(np.float32);cloud[active_n:]=cloud[0]
    points=wp.array(cloud,dtype=wp.vec3,device='cuda:0');active=wp.array([active_n],dtype=int,device='cuda:0')
    w=LargerPrefixDelaunayWorkspace(points,active_count=active);index=w.capacities.index(active_n);child=w._children[index]
    original=child.run
    def fail(stream=None):child.error.fill_(16);child.tetrahedra_count.zero_()
    child.run=fail
    with wp.ScopedCapture() as failure:w.clear_reuse();w.run()
    wp.capture_launch(failure.graph)
    assert w.error.numpy()[0]==0 and w.certificate_error.numpy()[0]==0 and w.selected_capacity.numpy()[0]==8193
    calls=[0]*5;calls[index]=1;calls[-1]=1;assert w.route_calls.numpy().tolist()==calls
    cells=w.tetrahedra.numpy()[:w.tetrahedra_count.numpy()[0]];assert tet_set(cells)==reference(cloud[:active_n])
    child.run=original
    with wp.ScopedCapture() as recovery:w.clear_reuse();w.run()
    wp.capture_launch(recovery.graph)
    assert w.error.numpy()[0]==0 and w.selected_capacity.numpy()[0]==active_n
    assert tet_set(w.tetrahedra.numpy()[:w.tetrahedra_count.numpy()[0]])==tet_set(cells)

@pytest.mark.parametrize('active_n',[4096,8192])
def test_large_prefix_output_exhaustion_fails_closed(active_n):
    cloud=np.random.default_rng(232).normal(size=(8193,3)).astype(np.float32);cloud[active_n:]=cloud[0]
    points=wp.array(cloud,dtype=wp.vec3,device='cuda:0');active=wp.array([active_n],dtype=int,device='cuda:0')
    w=LargerPrefixDelaunayWorkspace(points,active_count=active,tetrahedra_capacity=1)
    with wp.ScopedCapture() as cap:w.clear_reuse();w.run()
    for _ in range(2):
        wp.capture_launch(cap.graph)
        assert w.error.numpy()[0]!=0 and w.tetrahedra_count.numpy()[0]==0
        expected=[0]*5;expected[w.capacities.index(active_n)]=1;expected[-1]=1
        assert w.route_calls.numpy().tolist()==expected
        assert all(c._previous_count.numpy()[0]==0 for c in w._children)
