"""Observe production issue241 storage policy with the unchanged issue228 timing observer."""
import hashlib
import json
import os
from pathlib import Path
import runpy

from warpower import _topology_recovery as tr


def forbidden_approximate(self):
    raise AssertionError("TopologyRecovery reached approximate quality")


# Apply the same capture-time guard to both phases. Ordinary QualityMove remains
# unchanged; the candidate extractor's own cache must remain present.
tr.TopologyRecovery._approximate_quality = forbidden_approximate
assert os.environ["WARPOWER_QUERY_STORAGE_PHASE"] in ["before", "after"]

w = Path(__file__).parent
ns = runpy.run_path(str(w / "run-tangential-production-gallery-v1.py"))
m = ns["instances"][0]
r = m.topology_recovery
names = ["_query_cache", "_query_error", "_query_count", "_query_minimum", "_query_keep", "_query_quality"]
omitted = {name: getattr(r, name) is None for name in names}
expect = os.environ["WARPOWER_QUERY_STORAGE_PHASE"] == "after"
assert all(value == expect for value in omitted.values())
assert r.extraction._query_cache is not None
ordinary = {}
for name in ["jitter", "interior"]:
    workspace = getattr(m, name)
    ordinary[name] = {field: getattr(workspace, field) is not None for field in names}
    assert all(ordinary[name].values())
capacity = int(m.core.full_tetrahedra.shape[0])
table_capacity = 1 << (4 * capacity - 1).bit_length()
info = dict(
    scope=__doc__,
    phase=os.environ["WARPOWER_QUERY_STORAGE_PHASE"],
    recovery_fields_omitted=omitted,
    ordinary_fields_present=ordinary,
    candidate_cache_retained=True,
    approximate_capture_callback_forbidden=True,
    tetrahedron_capacity=capacity,
    predicted_removed_payload_bytes=180 * capacity + 4 * table_capacity + 24,
    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), Path(tr.__file__)]},
)
path = Path(os.environ["WARPOWER_QUERY_STORAGE_REPORT"])
assert not path.exists()
path.write_text(json.dumps(info, indent=2) + "\n")
print("QUERY_STORAGE", info, flush=True)
