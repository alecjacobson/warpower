"""Compare cold/warm exact-query cache retained decisions with the original kernel."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import Delaunay
import warp as wp
import warpower
from warpower._selective import classify_retained
from tests.test_domain import cube, torus, combine
from tests.test_extraction import orient
from tests.test_wide_center import needle, bridge
import surface_query_cache_kernels_v2 as candidate
from surface_query_cache_v2 import Cache,install

output=Path('.work/vtm/surface-query-cache-retained-check-v2.json')
assert not output.exists()
report=dict(scope='Original versus cold/warm exact-query cache; default caller FMA. Padded inactive outputs and kernel errors included.',
            sources={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,candidate.__file__,'warpower/_selective.py']},cases=[])
install()
wp.load_module(module=candidate,device='cuda:0',block_dim=32)


def compare(name,domain,p,t,tolerance,initial_error=0):
    n=len(t);capacity=n+9
    points=wp.array(p,dtype=wp.vec3)
    padded=np.full((capacity,4),-7,np.int32);padded[:n]=t
    cells=wp.array(padded,dtype=int);count=wp.array([n],dtype=int)
    results=[]
    cache=Cache(domain,capacity,name)
    for kernel in [classify_retained,candidate.retained_cached,candidate.retained_cached]:
        error=wp.full(1,initial_error,dtype=int)
        keep=wp.full(capacity,77,dtype=int);quality=wp.full(capacity,77,dtype=wp.float64)
        args=[domain.tree,points,cells,count,tolerance,error,keep,quality]
        if kernel is classify_retained:wp.launch(kernel,capacity,inputs=args,block_dim=32)
        else:cache.launch(kernel,capacity,args,32)
        results.append([x.numpy() for x in [error,keep,quality]])
    a=results[0]
    for b in results[1:]:
        np.testing.assert_array_equal(a[0],b[0])
        if a[0][0]==0:
            np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2].view(np.uint64),b[2].view(np.uint64))
    for value in results:
        np.testing.assert_array_equal(value[1][n:],0);np.testing.assert_array_equal(value[2][n:],77)
    report['cases'].append(dict(name=name,tolerance=tolerance,active=n,capacity=capacity,error=int(a[0][0]),
        input_sha256=hashlib.sha256(p.tobytes()+np.asarray(t,np.int32).tobytes()).hexdigest(),
        error_equal=True,successful_arrays_bitwise_equal=bool(a[0][0]==0),inactive_outputs_preserved=True))
    output.write_text(json.dumps(report,indent=2)+'\n')
    return a


for shape in ['cube','torus','cavity-components']:
    for scale in [1e-30,1.0,1e30]:
        if shape=='cube':p,f=cube()
        elif shape=='torus':p,f=torus(16,8)
        else:
            inner=cube(.4)
            p,f=combine(cube(),(inner[0],inner[1][:,::-1].copy()),cube(.5,(4,0,0)))
        domain=warpower.SurfaceDomain((p*scale).astype(np.float32),f)
        assert domain.error.numpy()[0]==0
        points=np.random.default_rng(194).uniform(-3,5,(100,3)).astype(np.float32)
        tets=orient(points,Delaunay(points).simplices)
        for tolerance in [0.0,1e-100,.4,.99]:
            compare(f'{shape}-{scale}',domain,(points*scale).astype(np.float32),tets,tolerance)

domain=warpower.SurfaceDomain(*cube())
offsets=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],np.float32)/8
centers=np.array([[1,0,0],[1,1,0],[1,1,1],[0,0,0]],np.float32)
p=(centers[:,None,:]+offsets[None,:,:]).reshape(-1,3);t=orient(p,np.arange(16).reshape(-1,4))
for tolerance in [0.0,1e-100,.4,.99]:
    a=compare('exact-boundaries-and-interior',domain,p,t,tolerance)
    np.testing.assert_array_equal(a[1][:4],1)

offsets=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],np.float64)
p=(offsets*(.4/np.sqrt(3))+[1.1,0,0]).astype(np.float32);t=orient(p,[[0,1,2,3]])
full=warpower.ExtractionWorkspace(domain,p,t).run()
ratio=full.signed_distance.numpy()[0]/full.circumradii.numpy()[0]
for tolerance,expected in [(np.nextafter(ratio,-np.inf),0),(ratio,0),(np.nextafter(ratio,np.inf),1)]:
    a=compare('strict-outside-tie',domain,p,t,float(tolerance));assert a[1][0]==expected
for name in ['needle','bridge','large-bridge']:
    p=needle() if name=='needle' else bridge()
    if name=='large-bridge':p=(p*2.0**120).astype(np.float32)
    compare(name,domain,p,np.arange(4,dtype=np.int32).reshape(1,4),.4)
for mode in ['missing-witness','all-missing','nonfinite-point','invalid-index','reversed','degenerate','initial-error']:
    p,f=cube();domain=warpower.SurfaceDomain(p,f)
    points=np.array([[0,0,0],[.1,0,0],[0,.1,0],[0,0,.1]],np.float32)
    t=np.arange(4,dtype=np.int32).reshape(1,4)
    if mode in ['missing-witness','all-missing']:
        first=int(domain.tree.ids.numpy()[0]);p=p.copy();p[f[first]]=np.nan
        if mode=='all-missing':p[:]=np.nan
        wp.copy(domain.tree.points,wp.array(p,dtype=wp.vec3))
    if mode=='nonfinite-point':points[0]=np.nan
    if mode=='invalid-index':t[0,0]=-1
    if mode=='reversed':t[0,:2]=t[0,1::-1]
    if mode=='degenerate':points[:]=0
    compare(mode,domain,points,t,.4,4 if mode=='initial-error' else 0)
report['finished']=True
output.write_text(json.dumps(report,indent=2)+'\n')
print('All',len(report['cases']),'retained kernel cases match in cold and warm cache phases.')
