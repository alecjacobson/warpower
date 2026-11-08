"""Repeat the failed private recovery experiment and retain its internal candidate."""
import hashlib
import json
import runpy
import sys
from pathlib import Path

from hand_seed_recovery_midpoint_v1 import install
from warpower import VariationalMeshingWorkspace

original = VariationalMeshingWorkspace.__init__


def setup(self, *args, **kwargs):
    original(self, *args, **kwargs)
    install(self)


VariationalMeshingWorkspace.__init__ = setup
runpy.run_path('.work/vtm/export-hand-seed-failure-v1.py', run_name='__main__')
output = Path(sys.argv[sys.argv.index('--output') + 1]).with_suffix('.policy.json')
assert not output.exists()
names = [__file__, '.work/vtm/hand_seed_recovery_midpoint_v1.py', '.work/vtm/representable_boundary_v1.py']
output.write_text(json.dumps(dict(policy='Recovery fitting only, exact midpoint proposals; still invalid.',
                                 source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in names}), indent=2) + '\n')
