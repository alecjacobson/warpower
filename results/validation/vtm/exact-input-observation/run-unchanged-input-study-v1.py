"""Observe repeated exact inputs on current complete VTM trajectories, without skipping work."""
import gzip, hashlib, json, os, signal, subprocess, time
from pathlib import Path

root = Path.cwd(); w = root / ".work/vtm"; out = w / "unchanged-input-study-v1"
assert not out.exists()
assert "1 passed" in (w / "unchanged-input-observer-tests-v1.log").read_text()
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == "c2843f658e49c45aff87973c9f70ccaea9dddc88"
assert not subprocess.check_output(["git", "status", "--porcelain"])
out.mkdir()
env = dict(os.environ, PYTHONPATH=".:.work/vtm", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
assert env.get("WARP_CACHE_PATH") == "/opt/warpower-generated-cache"
refs = json.loads((w / "larger-prefix-final-v1/commands.json").read_text())
lookup = {r["name"]: r for r in refs if r["phase"] == "after"}
old = json.loads((w / "bounded-hull-support-final/commands.json").read_text())
lookup["thin-hand"] = next(r for r in old if r["name"] == "thin-hand" and r["phase"] == "after")
rows = []; archives = []
for name, observed in [("thin-hand", False), *[(n, True) for n in ["torus1000", "torus4096", "torus8192-progressive", "torus16384-fixed32", "graded-torus16384", "reserved16384-target4096", "thin-hand"]]]:
    label = name + ("-observed" if observed else "-control")
    cmd = lookup[name]["command"][:]
    gallery = cmd[2] == "examples.vtm_gallery"
    cmd[cmd.index("--repeats") + 1] = "1" if not gallery else "2"
    destination = out / label if gallery else out / (label + ".json")
    cmd[cmd.index("--output") + 1] = str(destination)
    if "--save-mesh" in cmd:
        cmd[cmd.index("--save-mesh") + 1] = str(out / (label + ".npz"))
    observation = out / (label + "-observation.json")
    if observed:
        cmd[1:2] = [str(w / "run-unchanged-input-observer-v1.py")]
    runenv = dict(env, WARPOWER_UNCHANGED_OBSERVATION=str(observation))
    start = time.time()
    with (out / (label + ".log")).open("w") as stream:
        p = subprocess.Popen(cmd, env=runenv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=600)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM); p.wait(); code = 124
    row = dict(name=name, observed=observed, command=cmd, exit_code=code, wall_seconds=time.time()-start, timeout_seconds=600)
    rows.append(row)
    (out / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert code == 0, row
    result = destination / "result.json" if gallery else destination
    data = json.loads(result.read_text()); assert data["valid"] == 1 and data["error"] == 0
    row["result"] = str(result)
    if observed:
        info = json.loads(observation.read_text())
        row["routes"] = [{k: v for k, v in r.items() if k != "records"} for r in info["instances"]]
    if gallery:
        for name_ in ["mesh.vtu", "boundary.off"]:
            raw = destination / name_; packed = raw.with_suffix(raw.suffix + ".gz")
            original = raw.read_bytes(); packed.write_bytes(gzip.compress(original, mtime=0))
            assert gzip.decompress(packed.read_bytes()) == original
            archives.append(dict(original=str(raw), path=str(packed), raw_sha256=hashlib.sha256(original).hexdigest(), sha256=hashlib.sha256(packed.read_bytes()).hexdigest()))
            (out / "lossless-archives.json").write_text(json.dumps(archives, indent=2) + "\n")
            raw.unlink()
    (out / "commands.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(row, flush=True)
assert len(rows) == 8
