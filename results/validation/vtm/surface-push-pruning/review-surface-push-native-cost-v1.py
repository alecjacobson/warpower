"""Compare native kernel attribution at equal recorded work; diagnostic timing only."""
import hashlib
import json
from pathlib import Path

w=Path('.work/vtm'); out=w/'surface-push-native-cost-review-v1.json'; assert not out.exists()
paths=[w/(label+'-uniform-native-v1-kernels.json') for label in ['surface-sample','surface-push']]
before,after=[json.loads(p.read_text()) for p in paths]
baseline=json.loads((w/'surface-sample-production-source-v2.json').read_text())
current=json.loads((w/'surface-push-production-source-v1.json').read_text())
changed=[p for p,h in baseline['source_sha256'].items() if current['source_sha256'][p]!=h]
assert changed==['warpower/_surface_bvh.py']
report=dict(scope=__doc__,before_runtime=baseline['runtime'],after_runtime=current['runtime'],changed_sources=changed,inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=[],complete=False,limits='Separate instrumented native runs; not clean timing samples or an occupancy measurement. Full-solve performance uses the paired benchmark matrix.')
for prefix in ['selection_cached_ordered','classify_cached_ordered','surface_error','accumulate_samples','search_unproved_faces']:
    pair=[]
    for record in [before,after]:
        rows=[row for row in record['kernels'] if row['kernel'].startswith(prefix+'_')]
        assert len(rows)==1,(prefix,rows)
        pair.append(rows[0])
    a,b=pair; assert a['calls']==b['calls'] and a['block_size']==b['block_size']
    report['rows'].append(dict(kernel=prefix,before=a,after=b))
report['complete']=True; out.write_text(json.dumps(report,indent=2)+'\n')
print('Five native attribution rows retain call counts/block sizes; no clean-timing claim.')
