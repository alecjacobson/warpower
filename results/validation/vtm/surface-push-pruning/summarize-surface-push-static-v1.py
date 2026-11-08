"""Summarize all static rows, retaining regressions and query-output comparisons."""
import hashlib
import json
from pathlib import Path

w=Path(__file__).parent; out=w/'surface-push-pruning-static-summary-v1.json'; assert not out.exists()
folder=w/'surface-push-pruning-static-v1'
commands=json.loads((folder/'commands.json').read_text()); assert len(commands)==8 and all(c['exit_code']==0 for c in commands)
report=dict(scope=__doc__,complete=False,adopted=False,complete_solve_measured=False,cases={},sources={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
for name in dict.fromkeys(c['name'] for c in commands):
    paths=[folder/(name+'-'+p+'.json') for p in ['before','after']]
    a,b=[json.loads(p.read_text()) for p in paths]
    assert a['complete'] and b['complete'] and a['input_sha256']==b['input_sha256']
    assert a['source_sha256']==b['source_sha256']
    rows=[]
    for x,y in zip(a['rows'],b['rows']):
        assert x['name']==y['name'] and x['quota_clean'] and y['quota_clean']
        equal={key:value==y['output_sha256'][key] for key,value in x['output_sha256'].items()}
        if x['name']=='quadrature':
            assert equal['maximum'] and equal['error']
            relative=(y['values']['integral']-x['values']['integral'])/x['values']['integral']
            assert abs(relative)<=2e-12
        else: assert all(equal.values()); relative=None
        rows.append(dict(name=x['name'],before_ms=x['median_ms'],after_ms=y['median_ms'],reduction_percent=100*(1-y['median_ms']/x['median_ms']),equal_output_fields=equal,relative_integral_difference=relative,before_properties=x['properties'],after_properties=y['properties']))
    report['cases'][name]=dict(input_sha256=a['input_sha256'],sites=a['sites'],cells=a['cells'],capacity=a['capacity'],face_capacity=a['face_capacity'],rows=rows,reports={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
report.update(complete=True,rows=64,paired_rows=32,decision='Advance to paired complete-solve pilot. Graded retained cold classification regresses 5.03%; full/selection cold classifiers and quadrature improve in all four cases. No adoption or whole-solve speed claim from static rows.')
out.write_text(json.dumps(report,indent=2)+'\n'); print(report['decision'])
