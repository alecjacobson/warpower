"""Reuse completed native checks and qualify the complete isolated memory coverage."""
import json
from pathlib import Path
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
dest = w / "tangential-production-qualification-v3.json"
assert not dest.exists()
old = json.loads((w / "tangential-production-qualification-v2.json").read_text())
assert len(old) == 3 and old[-1]["exit_code"] == 1
rows = [dict(r, reused_from="tangential-production-qualification-v2.json") for r in old[:2]]
assert all(r["exit_code"] == 0 for r in rows)
dest.write_text(json.dumps(rows, indent=2) + "\n")
cmd = [str(root / ".venv/bin/python"), str(w / "run-tangential-production-memory-v3.py")]
log = w / "tangential-production-qualification-v3-memory.log"
assert not log.exists()
start = time.time()
# The child applies a finite timeout and process-group cleanup to each GPU job.
with log.open("w") as stream:
    result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
row = dict(name="memory", command=cmd, exit_code=result.returncode, wall_seconds=time.time()-start, per_job_timeouts=True)
rows.append(row)
dest.write_text(json.dumps(rows, indent=2) + "\n")
print(row, flush=True)
assert result.returncode == 0, row
