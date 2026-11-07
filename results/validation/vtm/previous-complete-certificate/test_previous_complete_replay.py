"""Captured fresh-certificate publication, changing owners, corruption and recovery."""
from itertools import combinations,product
import numpy as np
import pytest
import warp as wp
from tests.test_delaunay_recovery import exact_empty,valid
from tests.test_local_certificate import positive
from tests.test_tetrahedra import reference,tet_set,check_mesh
from warpower._reuse_delaunay import ReuseDelaunayWorkspace

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def produce(frames:wp.array2d[wp.vec3],phase:wp.array[int],points:wp.array[wp.vec3]):
    i=wp.tid();points[i]=frames[phase[0],i]

@wp.kernel
def consume(phase:wp.array[int],tets:wp.array2d[int],count:wp.array[int],error:wp.array[int],
            whole:wp.array[int],candidate:wp.array2d[int],candidate_count:wp.array[int],
            boundary:wp.array[int],boundary_count:wp.array[int],cells:wp.array3d[int],
            meta:wp.array2d[int],metadata_error:wp.array[int]):
    i=wp.tid();f=phase[0]
    if i<count[0]:
        for k in range(4):
            cells[f,i,k]=tets[i,k]
            if whole[1]!=0 and candidate[i,k]!=tets[i,k]:wp.atomic_or(metadata_error,0,1)
    if whole[1]!=0:
        if candidate_count[0]!=count[0]:wp.atomic_or(metadata_error,0,2)
        for j in range(i,boundary_count[0],tets.shape[0]):
            if boundary[j]<0 or boundary[j]>=4*count[0]:wp.atomic_or(metadata_error,0,4)
    if i==0:
        meta[f,0]=count[0];meta[f,1]=error[0]
        for k in range(3):meta[f,2+k]=whole[k]

@wp.kernel
def advance(phase:wp.array[int],condition:wp.array[int],total:int):
    phase[0]+=1;condition[0]=int(phase[0]<total)

@pytest.mark.parametrize('n',[32,256,2050])
def test_device_loop_scale_motion_owner_changes_invalid_and_recovery(n,monkeypatch):
    rng=np.random.default_rng(182)
    initial=rng.normal(size=(n,3)).astype(np.float32)
    scaled=initial*np.float32(2)
    nearby=scaled.copy();nearby[5,0]=np.nextafter(nearby[5,0],np.float32(np.inf))
    unrelated=rng.normal(size=(n,3)).astype(np.float32)
    half=n//2;duplicate=unrelated.copy();duplicate[half:]=duplicate[np.arange(n-half)%half]
    promoted=duplicate.copy();promoted[0]=np.array([20,21,22],np.float32)
    shrunk=initial.copy();shrunk[half:]=shrunk[0]
    invalid=initial.copy();invalid[4,0]=np.nan
    frames=np.stack([initial,scaled,nearby,unrelated,duplicate,promoted,shrunk,invalid,initial,initial])
    assert reference(initial)==reference(scaled)==reference(nearby)
    w=ReuseDelaunayWorkspace(initial);device=w.device
    src=wp.array(frames,dtype=wp.vec3,device=device);phase=wp.zeros(1,dtype=int,device=device);condition=wp.ones(1,dtype=int,device=device)
    cells=wp.empty((len(frames),len(w.tetrahedra),4),dtype=int,device=device)
    meta=wp.empty((len(frames),5),dtype=int,device=device);metadata_error=wp.zeros(1,dtype=int,device=device)
    wp.load_module(module=__name__,device=device)
    def step():
        wp.launch(produce,n,inputs=[src,phase,w.points],device=device)
        w.run()
        c=w._certificate
        wp.launch(consume,len(w.tetrahedra),inputs=[phase,w.tetrahedra,w.tetrahedra_count,w.error,w._whole_status,w._candidate,w._candidate_count,c._boundary,c._boundary_count,cells,meta,metadata_error],device=device)
        wp.launch(advance,1,inputs=[phase,condition,len(frames)],device=device)
    def forbidden(*args,**kwargs):raise AssertionError('Host allocation/readback/synchronization during captured work')
    with monkeypatch.context() as patch:
        for name in ['empty','zeros','ones','full','empty_like','clone','synchronize','synchronize_device','synchronize_stream']:patch.setattr(wp,name,forbidden)
        patch.setattr(wp.array,'numpy',forbidden)
        with wp.ScopedCapture() as capture:wp.capture_while(condition,step)
    wp.capture_launch(capture.graph)
    rows,out=meta.numpy(),cells.numpy();assert phase.numpy()[0]==len(frames) and metadata_error.numpy()[0]==0
    for f,points in enumerate(frames):
        if f==7:
            assert rows[f,0]==0 and rows[f,1]!=0 and rows[f,2:5].tolist()==[0,0,0]
            continue
        assert 0<rows[f,0]<=len(w.tetrahedra) and rows[f,1]==0,(f,rows[f])
        owners=np.unique(points,axis=0,return_index=True)[1]
        expected={tuple(sorted(owners[list(t)])) for t in reference(points[owners])}
        actual=out[f,:rows[f,0]];assert tet_set(actual)==expected,(f,rows[f]);check_mesh(points,actual)
        if f==5:assert half in actual  # Former duplicate promoted to a canonical query ID.
    assert rows[0,2:5].tolist()==rows[8,2:5].tolist()==[0,0,0]
    for f in [1,2,9]:assert rows[f,2:5].tolist()==[1,1,0],(f,rows[f])
    for f in [3,4,5,6]:assert rows[f,2]==1 and rows[f,3]==0 and rows[f,4]!=0,(f,rows[f])

@wp.kernel
def corrupt(previous:wp.array2d[int],count:wp.array[int],kind:int,n:int,bad:wp.vec4i):
    if kind==0:count[0]=previous.shape[0]+1
    elif kind==1:count[0]=-1
    elif kind==2:count[0]=0
    elif kind==3:previous[0,0]=-1
    elif kind==4:previous[0,0]=n
    elif kind==5:
        for k in range(4):previous[0,k]=previous[1,k]
    elif kind==6:count[0]-=1
    elif kind==7:
        a=previous[0,0];previous[0,0]=previous[0,1];previous[0,1]=a
    elif kind==8:
        for k in range(4):previous[0,k]=bad[k]

@pytest.mark.parametrize('kind',range(9))
def test_malformed_previous_candidate_is_not_published(kind):
    points=np.random.default_rng(183).normal(size=(32,3)).astype(np.float32)
    w=ReuseDelaunayWorkspace(points).run();expected=reference(points);assert tet_set(valid(w,points))==expected
    key=next(k for k in combinations(range(len(points)),4) if k not in expected)
    bad=positive(points,[key])[0]
    wp.load_module(module=__name__,device=w.device)
    with wp.ScopedCapture() as cap:
        wp.launch(corrupt,1,inputs=[w._previous,w._previous_count,kind,len(points),wp.vec4i(*map(int,bad))],device=w.device)
        w.run()
    wp.capture_launch(cap.graph)
    assert tet_set(valid(w,points))==expected
    status=w._whole_status.numpy()
    if kind in [1,2]:assert status.tolist()==[0,0,0]
    else:assert status[0]==1 and status[1]==0 and status[2]!=0
    # A later valid replay recovers direct certification rather than retaining
    # an old failed selector or candidate prefix.
    w.run();assert w._whole_status.numpy().tolist()==[1,1,0]
    assert tet_set(valid(w,points))==expected

@pytest.mark.parametrize('exponent',[-60,0,60])
def test_compacted_degenerate_rows_keep_current_candidate_and_face_metadata(exponent):
    points=np.array(list(product([-1,1],repeat=3)),np.float32)
    points=np.vstack([points,[[.1,.2,.3],[-.3,.2,.1]]]).astype(np.float32)*np.float32(2.0**exponent)
    w=ReuseDelaunayWorkspace(points).run();tets=valid(w,points);expected=tet_set(tets);exact_empty(points,tets)
    # IDs0..3 lie on x=-1. A harmless exact-zero row precedes the complete mesh;
    # the existing certificate is allowed to discard it after exact orientation.
    data=np.full(w._previous.shape,-1,np.int32);data[0]=[0,1,2,3];data[1:len(tets)+1]=tets
    source=wp.array(data,dtype=int,device=w.device);count=wp.full(1,len(tets)+1,dtype=int,device=w.device)
    with wp.ScopedCapture() as cap:
        wp.copy(w._previous,source);wp.copy(w._previous_count,count);w.run()
    wp.capture_launch(cap.graph)
    actual=valid(w,points);assert w._whole_status.numpy().tolist()==[1,1,0]
    assert tet_set(actual)==expected and len(actual)==len(tets);exact_empty(points,actual)
    np.testing.assert_array_equal(w._candidate.numpy()[:len(actual)],actual)
    assert w._candidate_count.numpy()[0]==len(actual)
    c=w._certificate;face=c._boundary.numpy()[:c._boundary_count.numpy()[0]]
    assert np.all((face>=0)&(face<4*len(actual)))
    w.run();assert w._whole_status.numpy().tolist()==[1,1,0] and tet_set(valid(w,points))==expected
