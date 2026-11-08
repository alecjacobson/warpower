"""Private issue241 before/after pairs; run only after issue228 qualification."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

root = Path.cwd()
w = root / ".work/vtm"
out = w / "quality-query-storage-pilot-v1"
assert not out.exists()
memory = json.loads((w / "tangential-production-memory-v3-commands.json").read_text())
assert len(memory) == 10 and all(r["exit_code"] == 0 and r["zero_memcheck_errors"] for r in memory)
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True)
runtime = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
python = root / ".venv/bin/python"
assert os.environ["WARP_CACHE_PATH"] == "/opt/warpower-generated-cache"
out.mkdir()
rows = []
archives = []
controls = json.loads((w / "tangential-production-final-v1/commands.json").read_text())
for name in ["thin-hand-seed7", "armadillo-graded4096", "armadillo-graded8192-recovery", "armadillo-uniform8192-recovery"]:
    control = next(r for r in controls if r["name"] == name and r["phase"] == "after")
    for phase in ["before", "after"]:
        label = name + "-" + phase
        cmd = control["command"][:]
        cmd[0] = str(python)
        cmd[1] = str(w / "run-quality-query-storage-gallery-v1.py")
        cmd[cmd.index("--output") + 1] = str(out / label)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", PYTHONPATH="." + os.pathsep + str(w), WARPOWER_QUERY_STORAGE_PHASE=phase, WARPOWER_QUERY_STORAGE_REPORT=str(out / (label + "-storage.json")), WARPOWER_TOPOLOGY_DIAGNOSTICS=str(out / (label + "-recovery.json")))
        start = time.time()
        with (out / (label + ".log")).open("w") as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait()
                code = 124
        row = dict(name=name, phase=phase, source=runtime, command=cmd, exit_code=code, wall_seconds=time.time()-start, timeout_seconds=600)
        result = out / label / "result.json"
        if result.exists():
            d = json.loads(result.read_text())
            row.update(result=str(result), result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(), valid=d["valid"], error=d["error"], median_ms=d["median_additional_replay_ms"], quota_clean=all(v["throttled_periods"] == 0 for k in ["cpu_quota_delta", "first_replay_cpu_quota_delta", "additional_replays_cpu_quota_delta"] for v in d[k].values()))
        rows.append(row)
        (out / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(row, flush=True)
        assert code == 0 and row["valid"] == 1 and row["error"] == 0 and row["quota_clean"], row
        if "armadillo" in name:
            duplicate = out / label / "input.off"
            canonical = w / "scanned-armadillo-input/armadillo.off"
            assert hashlib.sha256(duplicate.read_bytes()).digest() == hashlib.sha256(canonical.read_bytes()).digest()
            duplicate.unlink()
            os.link(canonical, duplicate)
        for name_to_archive in ["mesh.vtu", "boundary.off"]:
            raw = out / label / name_to_archive
            data = raw.read_bytes()
            packed = raw.with_suffix(raw.suffix + ".gz")
            assert not packed.exists()
            packed.write_bytes(gzip.compress(data, mtime=0))
            assert gzip.decompress(packed.read_bytes()) == data
            archives.append(dict(original=str(raw), path=str(packed), raw_sha256=hashlib.sha256(data).hexdigest(), sha256=hashlib.sha256(packed.read_bytes()).hexdigest()))
            (out / "lossless-archives.json").write_text(json.dumps(archives, indent=2) + "\n")
            raw.unlink()
assert len(rows) == 8
