"""Bounded private observation jobs; retain every failure. No timing speed claims."""
import argparse
import gzip
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--name", action="append", required=True)
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--paired", action="store_true")
args = parser.parse_args()
root = Path.cwd()
w = root/".work/vtm"
out = args.folder.resolve()
assert not out.exists()
assert not subprocess.check_output(["git", "status", "--porcelain"])
assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],text=True).strip()
commands = json.loads((w/"surface-push-final-v1/commands.json").read_text())
sources = [Path(__file__), w/"step_observer_v1.py", w/"run-step-observer-case-v1.py", *root.glob("warpower/*.py")]
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
out.mkdir()
(out/"source.json").write_text(json.dumps(dict(commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),source_sha256=hashes),indent=2)+"\n")
rows = []
for name in args.name:
    reference = next(r for r in commands if r["name"]==name and r["phase"]=="after")
    for mode in (["reference","observed"] if args.paired else ["observed"]):
        label = name+"-"+mode
        cmd = reference["command"][:]
        is_benchmark = cmd[1:3]==["-m","benchmarks.mesher"]
        cmd = [cmd[0], str(w/"run-step-observer-case-v1.py"), *cmd[3 if is_benchmark else 2:]]
        cmd[cmd.index("--output")+1] = str(out/(label+".json" if is_benchmark else label))
        if is_benchmark:
            cmd[cmd.index("--save-mesh")+1] = str(out/(label+".npz"))
        # Retain the original replay count to compare final warmed state.
        report = out/(label+"-observer.json")
        env = dict(os.environ, WARPOWER_STEP_MODE=mode, WARPOWER_STEP_REPORT=str(report),
                   WARPOWER_STEP_KIND="benchmark" if is_benchmark else "gallery")
        start = time.time()
        with (out/(label+".log")).open("w") as stream:
            process = subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                process.wait()
                code = 124
        row = dict(name=name, mode=mode, command=cmd, exit_code=code, wall_seconds=time.time()-start,
                   timeout_seconds=600, reference_result=reference["result"], report=str(report))
        result = Path(cmd[cmd.index("--output")+1])
        if result.is_dir():
            result = result/"result.json"
        if result.exists():
            d = json.loads(result.read_text())
            row.update(result=str(result),valid=d.get("valid"),error=d.get("error"))
        rows.append(row)
        (out/"commands.json").write_text(json.dumps(rows,indent=2)+"\n")
        print(row,flush=True)
        assert code==0 and row["valid"]==1 and row["error"]==0,row
        if not is_benchmark:
            for filename in ["input.off","mesh.vtu","boundary.off"]:
                raw=result.parent/filename
                data=raw.read_bytes()
                packed=raw.with_suffix(raw.suffix+".gz")
                assert not packed.exists()
                packed.write_bytes(gzip.compress(data,mtime=0))
                assert gzip.decompress(packed.read_bytes())==data
                raw.unlink()
        assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
