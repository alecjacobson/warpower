"""Additional first-owner/threshold tests, including non-reusing full fallback."""
from itertools import product
import numpy as np
import pytest
import warp as wp
from larger_prefix_v1 import LargerPrefixDelaunayWorkspace
from tests.test_delaunay_recovery import exact_empty
from tests.test_tetrahedra import check_mesh,tet_set

@pytest.mark.parametrize('capacity',[4097,8192,16385])
def test_sparse_first_owners_and_one_ulp_changes_at_route_edges(capacity):
    locations=np.array([0,1,1023,2047,2048,4095,8190,8191],np.int32)
    locations=locations[locations<capacity]
    unique=np.array(list(product([-1,1],repeat=3)),np.float32)[:len(locations)]
    def cloud_for(values,count):
        cloud=np.tile(values[0],(capacity,1))
        for j,i in enumerate(locations):
            if i<count:cloud[i]=values[j]
        return cloud
    active_n=min(capacity,8192)
    points=wp.array(cloud_for(unique,active_n),dtype=wp.vec3,device='cuda:0');active=wp.array([active_n],dtype=int,device='cuda:0')
    w=LargerPrefixDelaunayWorkspace(points,active_count=active)
    assert w._full_reuses_cells==(capacity<=16384)
    with wp.ScopedCapture() as cap:w.run()
    variants=[(4096,unique),(active_n,unique),(capacity,unique),(active_n,unique.copy()),(4096,unique)]
    variants[3][1][0,0]=np.nextafter(np.float32(-1),np.float32(0))
    outputs=[]
    for count,values in variants:
        cloud=cloud_for(values,count);points.assign(cloud);active.assign(np.array([count],np.int32));wp.capture_launch(cap.graph)
        assert w.error.numpy()[0]==0 and w.certificate_error.numpy()[0]==0
        cells=w.tetrahedra.numpy()[:w.tetrahedra_count.numpy()[0]]
        owners=locations[locations<count]
        assert set(cells.ravel())==set(owners)
        reduced=np.searchsorted(owners,cells).astype(np.int32);small=values[:len(owners)]
        check_mesh(small,reduced);exact_empty(small,reduced);outputs.append(tet_set(cells))
        np.testing.assert_array_equal(points.numpy().view(np.uint32),cloud.view(np.uint32))
    assert outputs[0]==outputs[-1]
