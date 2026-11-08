"""Start independent CPU checks only after every clean GPU timing job finishes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
dest = w / "quality-query-storage-independent-v1.json"
assert not dest.exists()
deadline = time.time() + 1800
while True:
    pipeline = w / "quality-query-storage-pipeline-v1.json"
    if pipeline.exists():
        try:
            rows = json.loads(pipeline.read_text())
        except json.JSONDecodeError:
            rows = []
        if rows:
            assert rows[0]["name"] == "timing" and rows[0]["exit_code"] == 0, rows
            break
    assert time.time() < deadline, "timing gate did not finish"
    time.sleep(2)
commands = json.loads((w / "quality-query-storage-final-v1/commands.json").read_text())
assert len(commands) == 20 and all(r["exit_code"] == 0 and r["quota_clean"] for r in commands)
cmd = [str(root / ".venv/bin/python"), str(w / "check-quality-query-storage-final-v1.py")]
log = w / "quality-query-storage-final-v1-independent.log"
assert not log.exists()
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONPATH=".")
start = time.time()
with log.open("w") as stream:
    p = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = p.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGTERM)
        p.wait()
        code = 124
row = dict(command=cmd, exit_code=code, wall_seconds=time.time()-start, timeout_seconds=1200, all_twenty_timings_complete_before_CPU_started=True)
dest.write_text(json.dumps(row, indent=2) + "\n")
print(row, flush=True)
assert code == 0, row
