"""Isolated observer buffer and complete captured-workspace memory checks.

Two earlier multi-workspace batches were stopped; v3 producer reached its
300-second deadline. These remain incomplete; v4 allows 1200 seconds per case. This scope verifies buffer exhaustion and real captured
producer/consumer/failure/local-retry execution in separate processes.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root=Path.cwd();w=root/".work/vtm"
out=w/"step-observer-memory-v4.json";assert not out.exists()
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
sources=[Path(__file__),w/"test_step_observer_buffer_v1.py",w/"step_observer_v1.py",w/"step_observer_pytest_plugin_v1.py",w/"large-reuse-isolated-test.py",root/"benchmarks/memcheck.py"]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
cases=[("buffer",str(w/"test_step_observer_buffer_v1.py"),2),
       ("producer","tests/test_mesher.py::test_full_device_producer_mesher_consumer_with_changing_input_and_recovery",1),
       ("local2026","tests/test_local_backtracking.py::test_optional_public_mesher_preserves_topology_and_corners[2026]",1),
       ("local7","tests/test_local_backtracking.py::test_optional_public_mesher_preserves_topology_and_corners[7]",1)]
prior_path=w/"step-observer-memory-v3.json"
prior=json.loads(prior_path.read_text())
assert prior["rows"][0]["name"]=="buffer" and prior["rows"][0]["exit_code"]==0 and prior["rows"][0]["zero_memcheck_errors"] and prior["rows"][0]["passed"]
assert prior["rows"][-1]["name"]=="producer" and prior["rows"][-1]["exit_code"]==124
buffer=dict(prior["rows"][0],log=str(w/"step-observer-memory-v3-buffer.log"),transferred_from=str(prior_path),source_report_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest())
assert "ERROR SUMMARY: 0 errors" in Path(buffer["log"]).read_text()
rows=[buffer]
for name,test,count in cases[1:]:
    cmd=["/usr/local/cuda-12.9/bin/compute-sanitizer","--tool","memcheck","--padding","32","--error-exitcode","99",
         str(root/".venv/bin/python"),str(w/"large-reuse-isolated-test.py"),"-p","step_observer_pytest_plugin_v1",test]
    env=dict(os.environ,WARPOWER_STEP_CONTRACT_REPORT=str(w/("step-observer-memory-v4-"+name+"-details.json")))
    log=w/("step-observer-memory-v4-"+name+".log")
    start=time.time()
    with log.open("w") as stream:
        p=subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=p.wait(timeout=1200)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code=124
    text=log.read_text()
    row=dict(name=name,command=cmd,exit_code=code,wall_seconds=time.time()-start,timeout_seconds=1200,log=str(log),
             zero_memcheck_errors="ERROR SUMMARY: 0 errors" in text,passed=f"{count} passed" in text,expected_tests=count)
    rows.append(row)
    out.write_text(json.dumps(dict(scope=__doc__,source_sha256=hashes,rows=rows,complete=len(rows)==4 and all(r["exit_code"]==0 and r["passed"] and r["zero_memcheck_errors"] for r in rows)),indent=2)+"\n")
    print(row,flush=True)
    assert code==0 and row["zero_memcheck_errors"] and row["passed"]
