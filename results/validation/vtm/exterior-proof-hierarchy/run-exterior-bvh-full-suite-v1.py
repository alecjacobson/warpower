import hashlib,json,os,signal,subprocess,time
from pathlib import Path
w=Path('.work/vtm');out=w/'exterior-bvh-full-suite-v1';assert not out.with_suffix('.log').exists()
cmd=[str(Path('.venv/bin/python').absolute()),'-m','pytest','-q','-x'];start=time.time()
with out.with_suffix('.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=900)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
r=dict(command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=900,runtime=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),log_sha256=hashlib.sha256(out.with_suffix('.log').read_bytes()).hexdigest())
out.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True);assert code==0,r
