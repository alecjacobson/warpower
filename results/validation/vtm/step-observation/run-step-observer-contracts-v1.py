"""Bounded graph composition and failed-stage compatibility checks for observer."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd();w=root/".work/vtm"
out=w/"step-observer-contracts-v1.json"
assert not out.exists()
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
names=["test_budget_failures_publish_zero_counts_and_metrics",
       "test_invalid_sizing_and_sampling_failures_gate_outputs_and_recover",
       "test_failed_insertion_batch_retries_one_site_and_reaches_target",
       "test_failed_optional_stages_restore_last_valid_mesh",
       "test_failed_restore_publishes_nothing_and_later_replay_recovers",
       "test_full_device_producer_mesher_consumer_with_changing_input_and_recovery",
       "test_mesher_composes_inside_an_outer_device_controlled_loop"]
cmd=[str(root/".venv/bin/python"),"-m","pytest","-q","-p","step_observer_pytest_plugin_v1",
     *["tests/test_mesher.py::"+n for n in names],
     "tests/test_local_backtracking.py::test_optional_public_mesher_preserves_topology_and_corners"]
sources=[Path(__file__),w/"step_observer_v1.py",w/"step_observer_pytest_plugin_v1.py",
         root/"tests/test_mesher.py",root/"tests/test_local_backtracking.py",*root.glob("warpower/*.py")]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
env=dict(os.environ,WARPOWER_STEP_CONTRACT_REPORT=str(w/"step-observer-contract-details-v1.json"))
start=time.time()
with (w/"step-observer-contracts-v1.log").open("w") as stream:
    p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=p.wait(timeout=300)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
result=dict(command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=300,source_sha256=hashes)
out.write_text(json.dumps(result,indent=2)+"\n")
print(result,flush=True)
assert code==0
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
