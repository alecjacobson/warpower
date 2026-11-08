"""Sequential GPU timing, native trace and memory gates for spatial query ordering."""
import json
from pathlib import Path
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
dest = w / "surface-query-order-pipeline-v1.json"
assert not dest.exists()
suite = json.loads((w / "surface-query-order-production-full-suite-v1.json").read_text())
assert suite["exit_code"] == 0 and suite["passed"]
rows = []
for name, script in [("timing", "run-surface-query-order-final-v1.py"), ("qualification", "run-surface-query-order-qualification-v1.py")]:
    log = w / ("surface-query-order-pipeline-v1-" + name + ".log")
    assert not log.exists()
    cmd = [str(root / ".venv/bin/python"), str(w / script)]
    start = time.time()
    with log.open("w") as stream:
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(name=name, command=cmd, exit_code=result.returncode, wall_seconds=time.time()-start, child_timeouts=True)
    rows.append(row)
    dest.write_text(json.dumps(rows, indent=2) + "\n")
    print(row, flush=True)
    assert result.returncode == 0, row
