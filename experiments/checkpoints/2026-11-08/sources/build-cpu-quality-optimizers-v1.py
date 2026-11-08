"""Build fresh optional CGAL quality-optimizer references after clean GPU timings end."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
w=Path('.work/vtm');out=w/'cpu-quality-optimizers-build-v1.json';assert not out.exists()
rows=json.loads((w/'selective-step-history-final-v1/commands.json').read_text());assert len(rows)==32 and all(r['exit_code']==0 and r['quota_clean'] for r in rows)
assert json.loads((w/'selective-step-history-final-v1-independent.json').read_text())['complete']
assert json.loads((w/'selective-step-history-feature-fit-v1.json').read_text())['complete']
record=json.loads((w/'cpu-quality-optimizers-source-v1.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in {**record['production'],**record['private']}.items():assert sha(Path(p))==h
source=w/'cpu-mesh3-quality-optimizers-v1.cpp';prior=json.loads((w/'cpu-feature-policy-build-v2.json').read_text())
r=dict(scope=__doc__,source=str(source),source_sha256=sha(source),compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0],cgal_version=prior['cgal_version'],cgal_source_revision=prior['cgal_source_revision'],builds=[],complete=False)
cache=Path('/opt/warpower-generated-cache/compiler-tmp');cache.mkdir(exist_ok=True)
for mode in ['sequential','parallel']:
 binary=w/('cpu-mesh3-quality-optimizers-v1-'+mode);log=w/('cpu-quality-optimizers-build-v1-'+mode+'.log');assert not binary.exists() and not log.exists()
 options=['-DWARPOWER_CGAL_PARALLEL','-DCGAL_LINKED_WITH_TBB'] if mode=='parallel' else []
 libraries=['-lgmp','-lmpfr']+(['-ltbb','-ltbbmalloc'] if mode=='parallel' else [])
 cmd=['c++','-O3','-DNDEBUG','-std=c++17',*options,'-I/home/horde/references/CGAL-6.2.1/include','-I/usr/include/eigen3',str(source),*libraries,'-o',str(binary)]
 start=time.time()
 with log.open('w') as stream:
  p=subprocess.Popen(cmd,env=dict(os.environ,TMPDIR=str(cache)),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=p.wait(timeout=1200)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
 row=dict(mode=mode,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=1200,log_sha256=sha(log))
 if binary.exists():row.update(binary_sha256=sha(binary),bytes=binary.stat().st_size)
 r['builds'].append(row);out.write_text(json.dumps(r,indent=2)+'\n');print(row,flush=True);assert code==0
for b in prior['builds']:
 p=w/('cpu-mesh3-feature-policy-v2-'+b['mode']);assert sha(p)==b['binary_sha256']
for p,h in record['production'].items():assert sha(Path(p))==h
r.update(complete=True,historical_binaries_and_public_source_unchanged=True);out.write_text(json.dumps(r,indent=2)+'\n')
