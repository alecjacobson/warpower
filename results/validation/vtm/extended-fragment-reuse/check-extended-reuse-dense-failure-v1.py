"""Record failed-output compatibility; no successful geometry or timing claim."""
import hashlib
import json
from pathlib import Path

w=Path('.work/vtm')
out=w/'extended-reuse-gallery-dense-v1'
dest=w/'extended-reuse-dense-failure-v1.json'
assert not dest.exists()
commands=json.loads((out/'commands.json').read_text())
assert len(commands)==2 and all(r['exit_code']==1 for r in commands)
rows={}
raw={}
keys=['valid','error','status','counts','rounds','steps','moves','last_errors','last_bad_extraction','topology','quality_rejections','refinement_rejections']
for phase in ['before','after']:
    path=out/('armadillo8192-graded-q2-'+phase)/'result.json'
    d=json.loads(path.read_text())
    assert d['valid']==0 and d['error']==64 and d['counts']==[0,0,0,0] and d['topology']==[0,0,0,0]
    raw[phase]=d
    rows[phase]=dict(result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),state={k:d[k] for k in keys},quota={k:d[k] for k in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta']},quota_clean=all(v['throttled_periods']==0 for k in ['first_replay_cpu_quota_delta','additional_replays_cpu_quota_delta'] for v in d[k].values()))
equal={k:raw['before'][k]==raw['after'][k] for k in keys}
report=dict(scope=__doc__,rows=rows,work_and_state_equal=equal,excluded_from_successful_timing=True,reason='Both quadrature2 configurations fail extraction; the before job additionally has observed CPU quota throttling.',limitation='Matching published failures and work do not prove identical internal candidate geometry.',complete=all(equal.values()))
dest.write_text(json.dumps(report,indent=2)+'\n')
assert all(equal.values())
print(json.dumps(dict(complete=True,quota_clean={k:v['quota_clean'] for k,v in rows.items()},successful_speedup_claim=False)),flush=True)
