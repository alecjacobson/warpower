"""Build fresh optional-feature-angle native references; preserve historical binaries."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

w=Path('.work/vtm')
dest=w/'cpu-feature-policy-build-v1.json'
assert not dest.exists()
assert json.loads((w/'surface-query-order-pilot-v2-independent-command.json').read_text())['exit_code']==0
source=w/'cpu-mesh3-feature-policy-v1.cpp'
prior=json.loads((w/'cpu-mesh3-target-build.json').read_text())
assert hashlib.sha256(Path('benchmarks/cpu/mesh3.cpp').read_bytes()).hexdigest()==prior['source_sha256']
report=dict(scope=__doc__,compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),historical_source_sha256=prior['source_sha256'],historical_binary_sha256={r['mode']:r['binary_sha256'] for r in prior['builds']},cgal_version='6.2.1',cgal_source_revision=prior['cgal_source_revision'],builds=[],complete=False,limits='Build qualification only. No native benchmark or geometry validation has run. Native studies must run separately from clean GPU timings.')
for mode in ['sequential','parallel']:
    binary=w/('cpu-mesh3-feature-policy-v1-'+mode)
    log=w/('cpu-feature-policy-build-v1-'+mode+'.log')
    assert not binary.exists() and not log.exists()
    options=['-DWARPOWER_CGAL_PARALLEL','-DCGAL_LINKED_WITH_TBB'] if mode=='parallel' else []
    libraries=['-lgmp','-lmpfr']+(['-ltbb','-ltbbmalloc'] if mode=='parallel' else [])
    cmd=['c++','-O3','-DNDEBUG','-std=c++17',*options,'-I/home/horde/references/CGAL-6.2.1/include','-I/usr/include/eigen3',str(source),*libraries,'-o',str(binary)]
    start=time.time()
    with log.open('w') as stream:
        p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=p.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            p.wait()
            code=124
    row=dict(mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=1200,log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
    if binary.exists():
        row['binary_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        row['bytes']=binary.stat().st_size
    report['builds'].append(row)
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(row,flush=True)
    assert code==0,row
for r in prior['builds']:
    assert hashlib.sha256((w/('cpu-mesh3-target-'+r['mode'])).read_bytes()).hexdigest()==r['binary_sha256']
report['complete']=True
report['historical_binaries_unchanged']=True
dest.write_text(json.dumps(report,indent=2)+'\n')
