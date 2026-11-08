"""Sequential GPU timing, native trace and memory gates for constructor-only storage reduction."""
import json
from pathlib import Path
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
dest = w / "quality-query-storage-pipeline-v1.json"
assert not dest.exists()
suite = json.loads((w / "quality-query-storage-production-full-suite-v1.json").read_text())
assert suite["exit_code"] == 0 and suite["passed"]
rows = []
for name, script in [("timing", "run-quality-query-storage-final-v1.py"), ("qualification", "run-quality-query-storage-qualification-v1.py")]:
    log = w / ("quality-query-storage-pipeline-v1-" + name + ".log")
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
