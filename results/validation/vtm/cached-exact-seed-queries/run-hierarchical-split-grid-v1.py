"""Fresh structured controls and candidates, after all other GPU work is terminal."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

root = Path.cwd()
w = root / ".work/vtm"
rows = []
dest = w / "hierarchical-split-grid-v1-commands.json"
assert not dest.exists()
assert len(json.loads((w / "hierarchical-split-pilot-v2/commands.json").read_text())) == 10
for phase in ["before", "after"]:
    cmd = [
        str(root / ".venv/bin/python"),
        str(w / "bench-hierarchical-split-grid-v1.py"),
        *(["--candidate"] if phase == "after" else []),
        "--output",
        str(w / f"hierarchical-split-grid-v1-{phase}.json"),
    ]
    start = time.time()
    with (w / f"hierarchical-split-grid-v1-{phase}.log").open("w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait()
            code = 124
    row = dict(
        phase=phase,
        command=cmd,
        exit_code=code,
        wall_seconds=time.time() - start,
        timeout_seconds=600,
    )
    rows.append(row)
    dest.write_text(json.dumps(rows, indent=2) + "\n")
    print(row, flush=True)
    assert code in [0, 1], row
