"""Record coordinate witnesses and independent quality deltas without changing checks."""
import hashlib
import json
from pathlib import Path
import numpy as np

w=Path('.work/vtm'); source=w/'surface-push-final-v1-independent.json'
proof=json.loads(source.read_text()); assert proof['complete']
out=w/'surface-push-variation-review-v1.json'; assert not out.exists()
report=dict(scope=__doc__,proof_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases={},complete=False,limits='Observed differences only; no causal kernel is identified. Independent proofs and production acceptance checks are unchanged.')
metrics=['mean_radius_ratio','dihedral_degrees_min','dihedral_degrees_max','relative_volume_error','two_sided_area_weighted_rms_percent_bbox_diagonal']
for name,row in proof['cases'].items():
    a,b=[row['independent'][phase]['quality'] for phase in ['before','after']]
    delta={key:b[key]-a[key] for key in metrics}
    delta['radius_ratio_percentiles']={key:b['radius_ratio_percentiles'][key]-value for key,value in a['radius_ratio_percentiles'].items()}
    assert abs(delta['mean_radius_ratio'])<1e-11
    assert all(abs(value)<1e-12 for value in delta['radius_ratio_percentiles'].values())
    assert abs(delta['dihedral_degrees_min'])<1e-10 and abs(delta['dihedral_degrees_max'])<1e-10
    assert abs(delta['relative_volume_error'])<1e-12 and abs(delta['two_sided_area_weighted_rms_percent_bbox_diagonal'])<1e-12
    witnesses=[]; paths=[]
    for phase in ['before','after']:
        p=w/'surface-push-final-v1'/(name+'-'+phase); paths.append(p/'mesh.npz' if p.is_dir() else p.with_suffix('.npz'))
    with np.load(paths[0]) as a,np.load(paths[1]) as b:
        x,y=a['points'],b['points']; xb,yb=x.view(np.uint32),y.view(np.uint32)
        diagonal=float(np.linalg.norm(np.ptp(a['input_points'].astype(np.float64),axis=0)))
        for site,axis in zip(*np.where(xb!=yb)):
            difference=float(y[site,axis])-float(x[site,axis])
            witnesses.append(dict(site=int(site),axis=int(axis),before=float(x[site,axis]),after=float(y[site,axis]),before_bits=int(xb[site,axis]),after_bits=int(yb[site,axis]),difference=difference,difference_over_bbox_diagonal=difference/diagonal))
    assert len(witnesses)==row['comparison']['changed_scalar_coordinates']
    report['cases'][name]=dict(witnesses=witnesses,quality_deltas=delta,mesh_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
report['complete']=True; out.write_text(json.dumps(report,indent=2)+'\n')
print('Reviewed',len(report['cases']),'pairs; maximum mean-quality delta',max(abs(r['quality_deltas']['mean_radius_ratio']) for r in report['cases'].values()))
