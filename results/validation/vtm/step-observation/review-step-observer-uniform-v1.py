"""Bind the uniform scan's selective guard codes to actual device trace rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

w=Path(".work/vtm");p=w/"step-observer-matrix-v1/armadillo-uniform8192-recovery-observed-observer.json"
meta=json.loads(p.read_text());trace=Path(meta["trace"])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(trace)==meta["trace_sha256"] and meta["overflow"]==0
with np.load(trace) as z:a=z["ints"]
c={k:i for i,k in enumerate(meta["int_columns"])}
rows=a[(a[:,c["event"]]==meta["events"].index("selection_post"))&(a[:,c["accepted"]]==0)]
counts=Counter(tuple(map(int,r[[c["selection_error"],c["selection_count"],c["selection_last_error"]]])) for r in rows)
assert counts=={(16,0,3):57}
out=w/"step-observer-uniform-rejections-v1.json";assert not out.exists()
out.write_text(json.dumps(dict(scope=__doc__,inputs={str(p):sha(p),str(trace):sha(trace)},source_sha256={str(Path(__file__)):sha(Path(__file__))},rows=[dict(error=e,retained_count=n,guard_reason=reason,count=count) for (e,n,reason),count in counts.items()],interpretation="Bit16 is ambiguous: recovered_status also ORs16 on invalid extraction. No specific geometric cause established.",complete=True),indent=2)+"\n")
