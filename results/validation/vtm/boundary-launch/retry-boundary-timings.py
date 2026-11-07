"""Preserve two throttled pairs, rerun them, then finish independent validation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

root=Path.cwd();w=root/'.work/vtm'
audits=json.loads((w/'boundary-audit-commands.json').read_text())
assert len(audits)==8 and all(r['exit_code']==0 for r in audits[:7])
assert audits[-1]['name']=='independent' and audits[-1]['exit_code']!=0
assert 'throttled_periods' in (w/'boundary-independent.log').read_text()
for src,dest in [('boundary-audit-commands.json','boundary-audit-initial-commands.json'),
                 ('boundary-audit-driver.log','boundary-audit-initial-driver.log'),
                 ('boundary-independent.log','boundary-initial-quota-rejection.log'),
                 ('boundary-timing-commands.json','boundary-initial-timing-commands.json')]:
    assert not (w/dest).exists();shutil.copyfile(w/src,w/dest)
rows=json.loads((w/'boundary-timing-commands.json').read_text());archive=w/'boundary-initial-throttled'
assert not archive.exists();archive.mkdir()
affected=[r for r in rows if r['name'] in ('torus4096','graded-torus')]
assert len(affected)==4
for r in affected:
    stem=w/f"boundary-{r['phase']}-{r['name']}"
    if stem.is_dir():shutil.copytree(stem,archive/stem.name)
    for suffix in ('.log','.json','.npz'):
        p=stem.with_suffix(suffix)
        if p.exists():shutil.copyfile(p,archive/p.name)
retry=[];env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='4',PYTHONPATH='.')
for r in affected:
    stem=w/f"boundary-{r['phase']}-{r['name']}";start=time.time()
    with stem.with_suffix('.log').open('w') as log:
        result=subprocess.run(r['command'],cwd=r['cwd'],env=env,stdout=log,stderr=subprocess.STDOUT)
    current=dict(r,seconds=time.time()-start,exit_code=result.returncode,replacement_reason='Original paired batch retained under boundary-initial-throttled; one timing window in the pair failed the zero-CPU-throttling gate.')
    retry.append(current);(w/'boundary-timing-retry-commands.json').write_text(json.dumps(retry,indent=2)+'\n')
    print(r['phase'],r['name'],result.returncode,current['seconds'],flush=True)
    assert result.returncode==0
    d=json.loads((stem.with_suffix('.json') if r['name'].startswith('torus') else stem/'result.json').read_text())
    for key in ('cpu_quota_delta','initial_replays_cpu_quota_delta','additional_replays_cpu_quota_delta','first_replay_cpu_quota_delta'):
        if key in d:assert all(x['throttled_periods']==0 for x in d[key].values()),(r['phase'],r['name'],key,d[key])
    time.sleep(1)
replacements={(r['phase'],r['name']):r for r in retry}
rows=[replacements.get((r['phase'],r['name']),r) for r in rows]
(w/'boundary-timing-commands.json').write_text(json.dumps(rows,indent=2)+'\n')
command=[str(root/'.venv/bin/python'),str(w/'check-boundary-results.py')];start=time.time()
with (w/'boundary-independent.log').open('w') as log:
    result=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
row=dict(name='independent',command=command,exit_code=result.returncode,seconds=time.time()-start)
(w/'boundary-audit-commands.json').write_text(json.dumps(audits[:7]+[row],indent=2)+'\n')
with (w/'boundary-audit-driver.log').open('a') as log:log.write(f"after retained quota rejection and paired timing reruns: independent {result.returncode} {row['seconds']:.3f}\n")
print('independent',result.returncode,row['seconds'],flush=True)
raise SystemExit(result.returncode)
