"""Direct exhaustive-filter oracle for hierarchical queries and changing inputs."""
import numpy as np
import pytest
import warp as wp
from scipy.spatial import Delaunay
from warpower._facet_seeds import CertifiedSeeds
from warpower._prepared_seeds import check
from warpower._site_bvh import SiteHierarchy
from seed_tree_queries import check_tree
from tests.test_facet_completion import canonical

@wp.kernel
def produce(source:wp.array[wp.vec3],points:wp.array[wp.vec3],phase:wp.array[int],scale:float):
    i=wp.tid()
    shift=wp.vec3(0.0)
    at=i
    if phase[0]==1:
        shift=wp.vec3(20.0,30.0,40.0)*scale
    elif phase[0]==2:
        at=(i+17)%points.shape[0]
        shift=wp.vec3(-30.0,20.0,-40.0)*scale
    points[i]=source[at]+shift

@pytest.mark.parametrize('n',[33,257,4096])
@pytest.mark.parametrize('exponent',[-60,0,60])
@pytest.mark.parametrize('block',[32,256])
def test_subtree_queries_match_exhaustive_on_current_points(n,exponent,block):
    rng=np.random.default_rng(2026)
    if n==4096:
        m=np.load('.work/vtm/extraction-split-4096/control.npz');p=m['points'];cells=m['full_tetrahedra']
    else:
        p=rng.normal(size=(n,3)).astype(np.float32);cells=Delaunay(p).simplices
    cells=cells[np.linspace(0,len(cells)-1,min(128,len(cells)),dtype=int)]
    random=np.asarray([rng.choice(n,4,replace=False) for _ in range(64)],np.int32)
    t=np.vstack([cells,random]).astype(np.int32)
    p=p*np.float32(2.0**exponent)
    assert len(np.unique(p,axis=0))==len(p)
    source=wp.array(p,dtype=wp.vec3,device='cuda:0');points=wp.empty_like(source);wp.copy(points,source)
    hierarchy=SiteHierarchy(points)
    hierarchy.tree.owners=wp.array(np.arange(n,dtype=np.int32),dtype=int,device='cuda:0')
    tets=wp.array(t,dtype=int,device='cuda:0');count=wp.full(1,len(t),dtype=int,device='cuda:0')
    old=CertifiedSeeds(points,tets,count,hierarchy.tree,prepare_geometry=True,block_dim=block)
    new=CertifiedSeeds(points,tets,count,hierarchy.tree,prepare_geometry=True,block_dim=block)
    phase=wp.zeros(1,dtype=int,device='cuda:0')
    wp.load_module(module=__name__,device='cuda:0')
    wp.load_module(module='seed_tree_queries',device='cuda:0',block_dim=block)
    original=wp.launch
    def launch(kernel,*args,**kwargs):
        return original(check_tree if kernel is check else kernel,*args,**kwargs)
    with wp.ScopedCapture() as cap:
        wp.launch(produce,n,inputs=[source,points,phase,float(2.0**exponent)],device='cuda:0')
        hierarchy.rebuild()
        old.run()
        try:
            wp.launch=launch
            new.run()
        finally:
            wp.launch=original
    for value in [0,1,2,0]:
        phase.fill_(value);wp.capture_launch(cap.graph)
        assert old.error.numpy()[0]==new.error.numpy()[0]==0
        a=old.tetrahedra.numpy()[:old.tetrahedra_count.numpy()[0]]
        b=new.tetrahedra.numpy()[:new.tetrahedra_count.numpy()[0]]
        np.testing.assert_array_equal(canonical(a),canonical(b))
