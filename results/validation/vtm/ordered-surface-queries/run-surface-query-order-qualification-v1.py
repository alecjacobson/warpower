"""Native and memory qualification after clean complete timing comparisons."""
import json
from pathlib import Path
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
dest = w / "surface-query-order-qualification-v1.json"
assert not dest.exists()
timings = json.loads((w / "surface-query-order-final-v1/commands.json").read_text())
assert len(timings) == 32 and all(r["exit_code"] == 0 and r["quota_clean"] for r in timings)
rows = []
for name, script in [("native-full", "run-surface-query-order-uniform-native-v1.py"), ("native-loop", "run-surface-query-order-loop-native-v1.py"), ("memory", "run-surface-query-order-memory-v1.py")]:
    log = w / ("surface-query-order-qualification-v1-" + name + ".log")
    assert not log.exists()
    cmd = [str(root / ".venv/bin/python"), str(w / script)]
    start = time.time()
    with log.open("w") as stream:
        # Each child bounds its own GPU processes and retains failure metadata.
        result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
    row = dict(name=name, command=cmd, exit_code=result.returncode, wall_seconds=time.time()-start, child_timeouts=True)
    rows.append(row)
    dest.write_text(json.dumps(rows, indent=2) + "\n")
    print(row, flush=True)
    assert result.returncode == 0, row
