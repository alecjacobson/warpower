"""Gallery entry point for the private last-chance recovery prototype."""
import hashlib
import json
import sys
from pathlib import Path

from full_proven_recovery_v2 import install

instances = install()
from examples.vtm_gallery import main

try:
    main()
finally:
    out = Path(sys.argv[sys.argv.index('--output') + 1]); out.mkdir(parents=True, exist_ok=True)
    paths = [__file__, '.work/vtm/proven_exterior_filter_v2.py', '.work/vtm/full_proven_recovery_v2.py', '.work/vtm/proven_exterior_transaction_v2.py',
             '.work/vtm/proven_exterior_v2.py', '.work/vtm/representable_boundary_v1.py']
    rows = []
    for m in instances:
        rows.append(dict(proof_enabled=int(m._proof_enabled.numpy()[0]), midpoint=m.core._tie_stats.numpy().tolist(),
                         material_recovery=int(m._material_recovery.numpy()[0]),
                         last_parent_proof_removed=int(m.extraction._proof.removed.numpy()[0]),
                         last_parent_proof_rejected=int(m.extraction._proof.rejected.numpy()[0])))
    record = dict(scope='Private complete fresh gallery prototype. Only after the existing recovery policy fails: one accepted/certified midpoint-aware fitting step and conservative proof-removal transaction; all subsequent quality stages assess the same enabled proof policy. Not yet adopted or fully validated.',
                  source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}, instances=rows)
    (out/'pilot.json').write_text(json.dumps(record, indent=2)+'\n')
