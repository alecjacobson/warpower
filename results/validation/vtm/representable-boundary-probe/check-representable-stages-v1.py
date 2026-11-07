"""Validate observer scope and summarize the measured selective backtracking."""
import hashlib,json
from pathlib import Path
import numpy as np
w=Path('.work/vtm');report={'observers':{},'first_selective_round':{}}
for version in [1,2]:
    for mode in ['baseline','candidate']:
        stem=w/f'representable-stages-{mode}-v{version}'
        meta=json.loads(stem.with_suffix('.json').read_text());snap=np.load(stem.with_suffix('.npz'))
        folder=w/'representable-full-v1'/('dense-hand-'+mode)
        raw=json.loads((folder/'result.json').read_text());mesh=np.load(folder/'mesh.npz')
        work={k:meta[k]==raw[k] for k in ['valid','error','counts','steps','rounds','moves']};assert all(work.values())
        delta=np.abs(snap['positions'][-1].astype(float)-mesh['points'])
        assert delta.max()<1e-8
        assert np.array_equal(snap['roles'][-1],mesh['roles'])
        row=dict(snapshot_sha256=hashlib.sha256(stem.with_suffix('.npz').read_bytes()).hexdigest(),work_equal=work,
                 maximum_final_coordinate_difference=float(delta.max()),changed_final_components=int(np.count_nonzero(delta)),
                 labels=meta['labels'],metric_columns=meta['metric_columns'],metrics=meta['metrics'])
        report['observers'][f'{mode}-v{version}']=row
        if version==2:
            steps=snap['steps'];first=steps[steps[:,0]==0];accepted=first[first[:,8]==1];rejected=first[first[:,8]==0]
            assert len(accepted)==10 and np.all(first[:,5]==1)
            if mode=='baseline':assert len(first)==10 and np.all(accepted[:,2]==.5)
            else:
                assert len(first)==110 and np.all(accepted[:,2]==.00048828125)
                assert np.all(rejected[:,10]==4) and np.all(rejected[:,7]<rejected[:,6]*(1-1e-10))
            report['first_selective_round'][mode]=dict(attempts=len(first),accepted=len(accepted),rejected=len(rejected),
                accepted_alpha=accepted[:,2].tolist(),initial_minimum=float(first[0,6]),
                first_trial_minimum=float(first[0,7]),focus_site_always_selected=True,
                first_direction_norm=float(first[0,3]),first_trial=first[0].tolist(),
                endpoint403=snap['positions'][4,403].tolist(),target403=snap['targets'][4,403].tolist())
dest=w/'representable-stages-analysis-v1.json';assert not dest.exists();dest.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['first_selective_round'],indent=2))
