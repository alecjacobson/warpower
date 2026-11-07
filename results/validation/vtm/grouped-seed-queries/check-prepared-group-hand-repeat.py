"""Retain the original pair and pool all eighteen samples per unchanged path."""
import json,statistics
from pathlib import Path
w=Path('.work/vtm');path=w/'prepared-group-final-independent.json'
initial=w/'prepared-group-final-independent-initial.json';assert not initial.exists();initial.write_bytes(path.read_bytes())
summary=json.loads(path.read_text());summary['initial_standalone']=dict(summary['standalone'])
old=[json.loads((w/f'prepared-group-final/delaunay-fitted-hand-{p}.json').read_text()) for p in ['before','after']]
new=[json.loads((w/f'prepared-group-hand-repeat/{p}.json').read_text()) for p in ['before','after']]
skip={'first_replay_ms','full','base','auxiliary','certificate'}
unchanged={k:all(d[k]==old[0][k] for d in [*old,*new]) for k in old[0] if k not in skip};assert all(unchanged.values())
for d in [*old,*new]:assert d['error']==0 and all(v['throttled_periods']==0 and v['throttled_usec']==0 for v in d['full']['cpu'].values())
samples=[a['full']['samples_ms']+b['full']['samples_ms'] for a,b in zip(old,new)]
assert all(len(s)==18 for s in samples)
medians=list(map(statistics.median,samples))
r=dict(before_ms=medians[0],after_ms=medians[1],relative_change=medians[1]/medians[0]-1,unchanged=unchanged,all_timed_quotas_clean=True,before_samples_ms=samples[0],after_samples_ms=samples[1],scope='Pooled original before/after pair and one fresh reverse-order confirmation pair: eighteen samples per version, none discarded. Unchanged standalone code; no speedup attributed to the VTM optimization.')
summary['standalone']['delaunay-fitted-hand']=r
path.write_text(json.dumps(summary,indent=2)+'\n')
(w/'prepared-group-hand-repeat-comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
