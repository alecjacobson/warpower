"""One padded memory audit of private observer graph contracts and overflow guard."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd();w=root/".work/vtm"
out=w/"step-observer-memory-v1.json"
assert not out.exists()
base=json.loads((w/"step-observer-contracts-v1.json").read_text());assert base["exit_code"]==0
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
cmd=["/usr/local/cuda-12.9/bin/compute-sanitizer","--tool","memcheck","--padding","32","--error-exitcode","99",
     *base["command"],str(w/"test_step_observer_buffer_v1.py")]
env=dict(os.environ,WARPOWER_STEP_CONTRACT_REPORT=str(w/"step-observer-memory-details-v1.json"))
sources=[Path(__file__),w/"test_step_observer_buffer_v1.py",w/"step_observer_v1.py",w/"step_observer_pytest_plugin_v1.py"]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
log=w/"step-observer-memory-v1.log"
start=time.time()
with log.open("w") as stream:
    p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=p.wait(timeout=600)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
text=log.read_text()
result=dict(command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=600,source_sha256=hashes,
            zero_memcheck_errors="ERROR SUMMARY: 0 errors" in text,passed_11="11 passed" in text)
out.write_text(json.dumps(result,indent=2)+"\n");print(result,flush=True)
assert code==0 and result["zero_memcheck_errors"] and result["passed_11"]
