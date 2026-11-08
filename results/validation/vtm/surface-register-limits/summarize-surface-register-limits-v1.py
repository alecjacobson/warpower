"""Summarize every private register-cap timing without selecting an adoption candidate."""
import hashlib
import json
from pathlib import Path

w=Path('.work/vtm');dest=w/'surface-register-limits-summary-v1.json';assert not dest.exists()
source=json.loads((w/'surface-register-limits-source-v1.json').read_text())
report=dict(scope=__doc__,source_sha256=hashlib.sha256((w/'surface-register-limits-source-v1.json').read_bytes()).hexdigest(),cases={},contracts={},complete=False,adopted=False,complete_solve_measured=False,limits=['Static immutable final meshes only. Includes keying/sorting/query/cache work, not a full trajectory or setup comparison.','Every candidate output is bitwise checked, but no candidate native/memory/lifetime adoption audit is claimed.','Observed register/local-memory properties do not establish an occupancy bottleneck.','No comparable-quality CPU speedup claim.'])
for limit in [96,128,192]:
    p=w/'surface-register-limits-v1'/f'contracts-r{limit}.json';a=json.loads(p.read_text())
    assert a['complete'] and a['passed'] and a['exit_code']==0
    for n,h in a['source_sha256'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h
    assert '19 passed' in p.with_suffix('.log').read_text()
    report['contracts'][str(limit)]=dict(report_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),properties=a['properties'],tests=19)
for name in ['torus1000','torus16384','armadillo-uniform8192','armadillo-graded8192']:
    p=w/'surface-register-limits-v2'/(name+'.json');a=json.loads(p.read_text())
    assert a['complete'] and len(a['rows'])==24 and all(r['quota_clean'] for r in a['rows'])
    rows=[]
    for mode in ['full','selection','retained']:
        for temperature in ['cold','warm']:
            subset=[r for r in a['rows'] if r['mode']==mode and r['cache']==temperature]
            assert [r['phase'] for r in subset]==['before','r96','r128','r192']
            before=subset[0]
            for r in subset[1:]:assert r['exact_outputs_match'] and r['output_sha256']==before['output_sha256']
            rows.append(dict(mode=mode,cache=temperature,ms={r['phase']:r['median_ms'] for r in subset},relative_change={r['phase']:r['median_ms']/before['median_ms']-1 for r in subset[1:]}))
    report['cases'][name]=dict(report_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),input=a['input'],input_sha256=a['input_sha256'],ordering_storage=a['ordering_storage'],properties=a['compiled_kernel_properties'],rows=rows)
report['complete']=True
report['decision']='Reject broad register caps for current full/selection hotspots:96/128 regress substantially and192 is essentially flat. Retained/warm static improvements do not establish a complete-solve gain. Production unchanged.'
dest.write_text(json.dumps(report,indent=2)+'\n')
print('57 contract tests,96 clean static rows, all candidate outputs bitwise equal; no adoption.')
