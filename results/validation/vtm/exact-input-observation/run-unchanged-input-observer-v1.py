"""Run unchanged public mesher/gallery with a private observation-only constructor."""
import hashlib, json, os, runpy, subprocess, sys
from pathlib import Path
import warp as wp
from unchanged_input_observer_v1 import install, COLUMNS

module = sys.argv.pop(1)
assert module in ["benchmarks.mesher", "examples.vtm_gallery"]
destination = Path(os.environ["WARPOWER_UNCHANGED_OBSERVATION"])
assert not destination.exists()
instances = install()
try:
    runpy.run_module(module, run_name="__main__")
finally:
    wp.synchronize()
    rows = []
    for m in instances:
        o = m.observer
        calls = int(o.calls.numpy()[0])
        assert o.overflow.numpy()[0] == 0 and calls <= len(o.records)
        records = o.records.numpy()[:calls].tolist()
        rows.append(dict(capacity=m.n, calls=calls, records=records,
                         valid_previous=sum(r[1] for r in records),
                         exact_unchanged=sum(r[1] == 1 and r[2] == 0 for r in records)))
    source = [Path(__file__), Path(__file__).with_name("unchanged_input_observer_v1.py")]
    report = dict(scope=__doc__, source=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  columns=COLUMNS, instances=rows,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
                  limitations=["Observation only; every original reconstruction/certificate executes.",
                               "Observer timings are not optimization performance evidence.",
                               "Count comparisons use exact float32 bits, including signed zero.",
                               "Fresh trajectories invalidate each observer checkpoint."])
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print("UNCHANGED_OBSERVATION", [(r["capacity"], r["calls"], r["exact_unchanged"]) for r in rows], flush=True)
