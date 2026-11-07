"""Direct original-kernel comparisons for the private inside-only probe."""
import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay
import warpower
from warpower import _extraction
from tests.test_domain import cube, torus, combine
from tests.test_extraction import orient
from tests.test_wide_center import bridge, needle
from recovery_classification_inside_v1 import classify_selection_inside

def compare(domain, p, t, tolerance, initial_error=0):
    device=domain.device;n=len(t);capacity=n+8
    points=wp.array(p,dtype=wp.vec3,device=device)
    padded=np.full((capacity,4),-7,np.int32);padded[:n]=t
    cells=wp.array(padded,dtype=int,device=device)
    count=wp.full(1,n,dtype=int,device=device);results=[]
    for kernel in [_extraction.classify,classify_selection_inside]:
        error=wp.full(1,initial_error,dtype=int,device=device)
        centers=wp.full(capacity,wp.vec3d(77),dtype=wp.vec3d,device=device)
        radius=wp.full(capacity,77,dtype=wp.float64,device=device)
        distance=wp.full(capacity,77,dtype=wp.float64,device=device)
        keep=wp.full(capacity,77,dtype=int,device=device)
        volume=wp.full(capacity,77,dtype=wp.float64,device=device)
        quality=wp.full(capacity,77,dtype=wp.float64,device=device)
        wp.launch(kernel,capacity,inputs=[domain.tree,points,cells,count,tolerance,error,centers,radius,distance,keep,volume,quality],device=device,block_dim=32)
        results.append({k:v.numpy() for k,v in dict(error=error,centers=centers,radius=radius,distance=distance,keep=keep,volume=volume,quality=quality).items()})
    a,b=results
    np.testing.assert_array_equal(a['error'],b['error'])
    if a['error'][0]==0:
        for k in ['centers','radius','keep','volume','quality']:
            np.testing.assert_array_equal(a[k],b[k])
        np.testing.assert_array_equal(a['distance']<=0,b['distance']<=0)
        outside=a['distance']>0
        np.testing.assert_array_equal(a['distance'][outside],b['distance'][outside])
    for r in results:
        np.testing.assert_array_equal(r['keep'][n:],0)
        for k in ['centers','radius','distance','volume','quality']:
            np.testing.assert_array_equal(r[k][n:],77)
    return a,b

@pytest.mark.parametrize('shape',['cube','torus','cavity-components'])
@pytest.mark.parametrize('scale',[1e-30,1.,1e30])
@pytest.mark.parametrize('tolerance',[0.,1e-100,.4,.99])
def test_random_scaled_clouds(shape,scale,tolerance):
    if shape=='cube':p,f=cube()
    elif shape=='torus':p,f=torus(16,8)
    else:
        inner=cube(.4)
        p,f=combine(cube(),(inner[0],inner[1][:,::-1].copy()),cube(.5,(4,0,0)))
    domain=warpower.SurfaceDomain((p*scale).astype(np.float32),f)
    assert domain.error.numpy()[0]==0
    points=np.random.default_rng(194).uniform(-3,5,(100,3)).astype(np.float32)
    tets=orient(points,Delaunay(points).simplices)
    a,b=compare(domain,(points*scale).astype(np.float32),tets,tolerance)
    assert a['error'][0]==0

@pytest.mark.parametrize('tolerance',[0.,1e-100,.4,.99])
def test_exact_face_edge_corner_and_interiors(tolerance):
    offsets=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],np.float32)/8
    centers=np.array([[1,0,0],[1,1,0],[1,1,1],[0,0,0]],np.float32)
    p=(centers[:,None,:]+offsets[None,:,:]).reshape(-1,3)
    t=orient(p,np.arange(16).reshape(-1,4))
    a,b=compare(warpower.SurfaceDomain(*cube()),p,t,tolerance)
    assert a['error'][0]==0
    np.testing.assert_array_equal(a['keep'][:4],1)
    assert b['distance'][3]==-1

def test_strict_outside_tie_and_neighbors():
    offsets=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],np.float64)
    p=(offsets*(.4/np.sqrt(3))+[1.1,0,0]).astype(np.float32)
    t=orient(p,[[0,1,2,3]]);domain=warpower.SurfaceDomain(*cube())
    a,_=compare(domain,p,t,.4);ratio=a['distance'][0]/a['radius'][0]
    for tolerance,expected in [(np.nextafter(ratio,-np.inf),0),(ratio,0),(np.nextafter(ratio,np.inf),1)]:
        a,b=compare(domain,p,t,float(tolerance));assert a['keep'][0]==expected

@pytest.mark.parametrize('geometry',['needle','bridge','large-bridge'])
def test_exact_geometry_fallback(geometry):
    p=needle() if geometry=='needle' else bridge()
    if geometry=='large-bridge':p=(p*2.**100).astype(np.float32)
    a,b=compare(warpower.SurfaceDomain(*cube()),p,np.arange(4,dtype=np.int32).reshape(1,4),.4)
    assert a['error'][0]==0
    if geometry=='large-bridge':assert np.max(np.abs(a['centers'][0]))>np.finfo(np.float32).max

@pytest.mark.parametrize('mode',['missing-witness','all-missing','nonfinite-point','invalid-index','reversed','degenerate','initial-error'])
def test_unavailable_witness_and_errors(mode):
    p,f=cube();domain=warpower.SurfaceDomain(p,f)
    points=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],np.float32)
    t=np.arange(4,dtype=np.int32).reshape(1,4)
    if mode in ['missing-witness','all-missing']:
        first=int(domain.tree.ids.numpy()[0]);p=p.copy();p[f[first]]=np.nan
        if mode=='all-missing':p[:]=np.nan
        wp.copy(domain.tree.points,wp.array(p,dtype=wp.vec3,device=domain.device))
        assert np.isnan(domain.tree.points.numpy()[f[first]]).all()
    if mode=='nonfinite-point':points[0]=np.nan
    if mode=='invalid-index':t[0,0]=-1
    if mode=='reversed':t[0,:2]=t[0,1::-1]
    if mode=='degenerate':points[:]=0
    a,b=compare(domain,points,t,.4,4 if mode=='initial-error' else 0)
    if mode=='all-missing':assert a['error'][0]==16
    if mode in ['nonfinite-point','invalid-index','initial-error']:assert a['error'][0]==4
    if mode in ['reversed','degenerate']:assert a['error'][0]==8
