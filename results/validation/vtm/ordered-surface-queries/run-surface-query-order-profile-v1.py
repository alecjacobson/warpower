"""Private ordering contract and static profiles; no production adoption claim."""
import json
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
out = w / "surface-query-order-profile-v1"
assert not out.exists()
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
memory = json.loads((w / "tangential-production-memory-v3-commands.json").read_text())
assert len(memory) == 10 and all(r["exit_code"] == 0 for r in memory)
out.mkdir()
python = root / ".venv/bin/python"
env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", PYTHONPATH="." + os.pathsep + str(w))
assert env["WARP_CACHE_PATH"] == "/opt/warpower-generated-cache"
rows = []


def run(name, cmd):
    start = time.time()
    with (out / (name + ".log")).open("w") as stream:
        p = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait()
            code = 124
    row = dict(name=name, command=cmd, exit_code=code, wall_seconds=time.time()-start, timeout_seconds=600)
    rows.append(row)
    (out / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(row, flush=True)
    assert code == 0, row


prior = w / "surface-query-order-contracts-v2.json"
if prior.exists():
    contract = json.loads(prior.read_text())
    assert contract["exit_code"] == 0 and contract["passed"]
    for path, digest in contract["source_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
    os.link(w / "surface-query-order-contracts-v2.log", out / "contracts.log")
    rows.append(dict(name="contracts", reused_from=str(prior), **contract))
    (out / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
else:
    run("contracts", [str(python), str(w / "test-surface-query-order-contracts-v2.py")])
assert "10 passed" in (out / "contracts.log").read_text()
base = w / "tangential-production-final-v1"
for name, source in [
    ("torus1000", base / "torus1000-after.npz"),
    ("torus16384", base / "torus16384-fixed32-after.npz"),
    ("armadillo-uniform8192", base / "armadillo-uniform8192-recovery-after/mesh.npz"),
    ("armadillo-graded8192", base / "armadillo-graded8192-recovery-after/mesh.npz"),
]:
    run(name, [str(python), str(w / "profile-surface-query-order-v1.py"), "--input", str(source), "--output", str(out / (name + ".json"))])
