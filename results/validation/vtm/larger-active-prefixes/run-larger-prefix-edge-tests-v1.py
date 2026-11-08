"""Record private correctness gate; no performance measurements."""
import hashlib,json,os,subprocess,time
from pathlib import Path
w=Path('.work/vtm');log=w/'larger-prefix-edge-tests-v1.log';result=w/'larger-prefix-edge-tests-v1.json';assert not log.exists() and not result.exists()
cmd=[str(Path.cwd()/'.venv/bin/python'),'-m','pytest','-q','-p','no:cacheprovider',str(w/'test_larger_prefix_edges_v1.py')]
start=time.time()
with log.open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=600)
r=dict(command=cmd,source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),exit_code=p.returncode,wall_seconds=time.time()-start,sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [w/'larger_prefix_v1.py',w/'test_larger_prefix_edges_v1.py',log]})
result.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True);raise SystemExit(p.returncode)
