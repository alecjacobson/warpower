"""Private CUDA state-machine and unchanged graph composition/recovery contracts."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd();w=root/".work/vtm";out=w/"selective-step-history-cuda-contracts-v1.json"
assert not out.exists()
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
cpu=json.loads((w/"selective-step-history-cpu-contracts-v1.json").read_text());assert cpu["exit_code"]==0 and cpu["passed"]==13
for p,h in cpu["source_sha256"].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
sources=[Path(__file__),w/"selective_step_history_v1.py",w/"selective_step_history_pytest_plugin_v1.py",w/"test_selective_step_history_v1.py",root/"tests/test_mesher.py",root/"tests/test_local_backtracking.py",*root.glob("warpower/*.py")]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
cmd=[str(root/".venv/bin/python"),"-m","pytest","-q","-p","selective_step_history_pytest_plugin_v1",
     str(w/"test_selective_step_history_v1.py"),
     "tests/test_mesher.py::test_full_device_producer_mesher_consumer_with_changing_input_and_recovery",
     "tests/test_mesher.py::test_mesher_composes_inside_an_outer_device_controlled_loop",
     "tests/test_local_backtracking.py::test_optional_public_mesher_preserves_topology_and_corners"]
env=dict(os.environ,WARPOWER_HISTORY_TEST_DEVICE="cuda:0")
start=time.time();log=w/"selective-step-history-cuda-contracts-v1.log"
with log.open("w") as stream:
    p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=p.wait(timeout=300)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
passed=17 if "17 passed" in log.read_text() else 0
result=dict(command=cmd,exit_code=code,passed=passed,wall_seconds=time.time()-start,timeout_seconds=300,source_sha256=hashes)
out.write_text(json.dumps(result,indent=2)+"\n");print(result,flush=True)
assert code==0 and passed==17
