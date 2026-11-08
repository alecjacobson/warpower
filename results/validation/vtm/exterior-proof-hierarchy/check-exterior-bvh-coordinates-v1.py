"""Quantify non-bitwise outputs without weakening geometric validation or hiding differences."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');out=w/'exterior-bvh-final-v2';dest=w/'exterior-bvh-coordinate-differences-v1.json';assert not dest.exists()
prior=w/'exterior-bvh-final-v2-independent.json';report=json.loads(prior.read_text());assert len(report['cases'])==9
result=dict(scope=__doc__,partial_validation_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),cases={})
for name,row in report['cases'].items():
    assert row['independent_before']['full_hull']['valid'] and row['independent_after']['full_hull']['valid']
    assert row['independent_before']['boundary_self_intersection_pairs']==row['independent_after']['boundary_self_intersection_pairs']==0
    assert all(row['work_equal'].values()) and all(row['comparison'][k] for k in ['full_tets_equal','material_tets_equal','roles_equal','oriented_boundary_equal'])
    def path(phase):
        p=out/(name+'-'+phase);return p/'mesh.npz' if p.is_dir() else p.with_suffix('.npz')
    with np.load(path('before')) as a,np.load(path('after')) as b:
        x,y=a['points'],b['points'];xb,yb=x.view(np.uint32),y.view(np.uint32)
        changed=np.argwhere(xb!=yb);diagonal=float(np.linalg.norm(np.ptp(a['input_points'].astype(float),axis=0)))
        values=[]
        for i,k in changed:
            def ordered(bit):return (0xffffffff^int(bit)) if int(bit)&0x80000000 else int(bit)^0x80000000
            values.append(dict(site=int(i),axis=int(k),before=float(x[i,k]),after=float(y[i,k]),before_bits=int(xb[i,k]),after_bits=int(yb[i,k]),ulps=abs(ordered(xb[i,k])-ordered(yb[i,k]))))
        maximum=float(np.max(np.abs(x.astype(float)-y.astype(float))))
        # Quantify physical quality on the same canonical cell connectivity.
        def quality(d):
            t=np.sort(d['tetrahedra'],axis=1);order=np.lexsort(t.T[::-1])
            p=d['points'].astype(float)[d['tetrahedra'][order]]
            edges=p[:,1:]-p[:,:1];vol=np.linalg.det(edges)/6
            offsets=np.linalg.solve(2*edges,np.sum(edges*edges,axis=2)[...,None])[...,0]
            area=np.zeros(len(p))
            for opposite in range(4):
                f=p[:,[j for j in range(4) if j!=opposite]];area+=np.linalg.norm(np.cross(f[:,1]-f[:,0],f[:,2]-f[:,0]),axis=1)/2
            return 9*vol/(area*np.linalg.norm(offsets,axis=1))
        qa,qb=quality(a),quality(b);qd=float(np.max(np.abs(qa-qb)))
        r=dict(before_sha256=hashlib.sha256(path('before').read_bytes()).hexdigest(),after_sha256=hashlib.sha256(path('after').read_bytes()).hexdigest(),changed_coordinates=len(values),values=values,maximum_coordinate_difference=maximum,input_bbox_diagonal=diagonal,relative_coordinate_difference=maximum/diagonal,maximum_radius_ratio_difference=qd,minimum_radius_ratio_before=float(qa.min()),minimum_radius_ratio_after=float(qb.min()))
        assert r['relative_coordinate_difference']<1e-9 and qd<1e-7,r
        result['cases'][name]=r;dest.write_text(json.dumps(result,indent=2)+'\n');print(name,{k:v for k,v in r.items() if k not in ['values','before_sha256','after_sha256']},flush=True)
assert {n for n,r in result['cases'].items() if r['changed_coordinates']}=={'thin-hand-seed7','cavity','disconnected'}
result['complete']=True;dest.write_text(json.dumps(result,indent=2)+'\n')
