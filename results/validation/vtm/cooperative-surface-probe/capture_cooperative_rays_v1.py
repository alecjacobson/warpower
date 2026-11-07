"""Private ray kernels in a device producer/refit/query loop; not full VTM."""
import hashlib
import json
from pathlib import Path
import numpy as np
import warp as wp
from warpower import SurfaceDomain, DomainQueryWorkspace
from tests.test_domain import cube, solid_angle_inside
import cooperative_rays_v2 as kernels

wp.set_module_options({'enable_backward':False,'fast_math':False})


@wp.kernel
def producer(source:wp.array[wp.vec3],out:wp.array[wp.vec3],phase:wp.array[int],is_domain:bool):
    i=wp.tid()
    p=source[i]
    state=phase[0]
    if state==1 or state==6:
        p=wp.vec3(1.5*p[0]+0.2*p[1]+0.125,0.5*p[1]-0.25,2.0*p[2])
    if is_domain:
        if state==2:p[2]=0.0
        elif state==3 and i==0:p[0]=wp.nan
        elif state==4:p[0]=-p[0]
    out[i]=p


@wp.kernel
def query_points(source:wp.array[wp.vec3],out:wp.array[wp.vec3d]):
    i=wp.tid()
    out[i]=wp.vec3d(source[i])


@wp.kernel
def advance(phase:wp.array[int]):phase[0]+=1


def main():
    output=Path('.work/vtm/cooperative-rays-capture-v1.json')
    assert not output.exists()
    p,f=cube();p[:,2]*=np.float32(1e-7)
    domain=SurfaceDomain(p,f,surface_subdivisions=1,crease_subdivisions=1)
    probes=np.array([[0,0,0],[1.1,0,0],[0,0,2e-7],[.2,-.3,-.9e-7]],np.float32)
    query=DomainQueryWorkspace(domain,probes)
    source=wp.array(p,dtype=wp.vec3);query_source=wp.array(probes,dtype=wp.vec3)
    phase=wp.zeros(1,dtype=int);points=wp.empty(len(probes),dtype=wp.vec3d);ex=wp.full(len(probes),-1,dtype=int)
    names=['leaf_1','leaf_2','leaf_4','subtree_2','subtree_4','subtree_8','subtree_16','subtree_32']
    outs={k:wp.zeros((len(probes),3),dtype=int) for k in ['reference']+names}
    wp.load_module(module=kernels,device=domain.device,block_dim=32)
    wp.load_module(module=__name__,device=domain.device)
    with wp.ScopedCapture() as cap:
        wp.launch(producer,len(p),inputs=[source,domain.points,phase,True])
        wp.launch(producer,len(probes),inputs=[query_source,query.points,phase,False])
        domain.refit();query.run()
        wp.launch(query_points,len(probes),inputs=[query.points,points])
        wp.launch(kernels.reference,len(probes),inputs=[domain.tree,points,ex,outs['reference']],block_dim=32)
        for name in names:
            group=int(name.rsplit('_',1)[1])
            wp.launch_tiled(getattr(kernels,name),dim=(len(probes)+32//group-1)//(32//group),inputs=[domain.tree,points,ex,outs[name]],block_dim=32)
        wp.launch(advance,1,inputs=[phase])
    report=dict(scope='Seven captured ray-query/refit phases; offline checks between replays. No native audit, full VTM or invalid-domain ray-semantics claim.',
                sources={str(x):hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in [__file__,kernels.__file__]},phases=[])
    for state in range(7):
        wp.capture_launch(cap.graph);wp.synchronize()
        error=int(domain.error.numpy()[0]);valid=query.valid.numpy()
        row=dict(phase=state,domain_error=error,public_query_error=int(query.error.numpy()[0]),public_query_valid=valid.tolist())
        if state in [2,3,4]:
            assert error!=0
            np.testing.assert_array_equal(valid,0)
        else:
            assert error==query.error.numpy()[0]==0
            np.testing.assert_array_equal(valid,1)
            ref=outs['reference'].numpy()
            expected=solid_angle_inside(domain.points.numpy(),f,query.points.numpy()).astype(np.int32)
            np.testing.assert_array_equal(ref[:,2],expected)
            np.testing.assert_array_equal(query.inside.numpy(),expected)
            row['side']=ref[:,2].tolist();row['candidate_mismatches']={}
            for name in names:
                val=outs[name].numpy()
                np.testing.assert_array_equal(val,ref)
                row['candidate_mismatches'][name]=int(np.count_nonzero(val!=ref))
        report['phases'].append(row)
        output.write_text(json.dumps(report,indent=2)+'\n')
    report['finished']=True
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
