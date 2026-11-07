"""Independent full-hull and element-quality checks, including invalid material outputs."""
import hashlib,json,subprocess,tempfile
from pathlib import Path
import numpy as np
from tests.test_tetrahedra import check_mesh
root=Path.cwd();w=root/'.work/vtm';result={};dest=w/'representable-transactions-independent-v1.json';assert not dest.exists()
for case in ['old-hand','current-hand','torus']:
    result[case]={}
    for mode in ['baseline','candidate']:
        stem=w/f'representable-transaction-{case}-{mode}-v2';d=np.load(stem.with_suffix('.npz'))
        original=json.loads(stem.with_suffix('.json').read_text())
        assert original['core_error']==original['mesh_error']==original['boundary_error']==0
        p=d['points'];full=d['full_tetrahedra'];t=d['tetrahedra']
        check_mesh(p,full)
        with tempfile.TemporaryDirectory(prefix='representable-hull-',dir='/opt/warpower-generated-cache') as tmp:
            pp,tp=Path(tmp)/'points.txt',Path(tmp)/'tets.txt'
            np.savetxt(pp,p,fmt='%.17g',header=str(len(p)),comments='')
            np.savetxt(tp,full,fmt='%d',header=str(len(full)),comments='')
            valid=json.loads(subprocess.check_output([str(w/'check-delaunay'),str(pp),str(tp)],text=True))
        assert valid['valid'],valid
        cells=p.astype(float)[t];edges=cells[:,1:]-cells[:,:1]
        volumes=np.linalg.det(edges)/6;assert np.all(volumes>0)
        offset=np.linalg.solve(2*edges,np.sum(edges*edges,axis=2)[...,None])[...,0]
        radii=np.linalg.norm(offset,axis=1);areas=np.zeros(len(t))
        for opposite in range(4):
            face=cells[:,[i for i in range(4) if i!=opposite]]
            areas+=np.linalg.norm(np.cross(face[:,1]-face[:,0],face[:,2]-face[:,0]),axis=1)/2
        quality=9*volumes/(areas*radii)
        np.testing.assert_allclose(quality.min(),original['minimum_material_radius_ratio'],rtol=1e-7,atol=1e-14)
        # Count every unoriented boundary-edge incidence; do not assert manifold
        # material output for the deliberately invalid coarse hand state.
        incidence={}
        for face in d['boundary']:
            for i in range(3):
                edge=tuple(sorted([int(face[i]),int(face[(i+1)%3])]))
                incidence[edge]=incidence.get(edge,0)+1
        row=dict(mesh_sha256=hashlib.sha256(stem.with_suffix('.npz').read_bytes()).hexdigest(),full_hull=valid,
                 material_valid=original['extraction_valid'],minimum_material_radius_ratio=float(quality.min()),
                 radius_ratio_percentiles=np.percentile(quality,[0,1,5,50,95,100]).tolist(),
                 nonmanifold_boundary_edges=sum(v!=2 for v in incidence.values()))
        result[case][mode]=row;dest.write_text(json.dumps(result,indent=2)+'\n');print(case,mode,row['minimum_material_radius_ratio'],row['nonmanifold_boundary_edges'],flush=True)
