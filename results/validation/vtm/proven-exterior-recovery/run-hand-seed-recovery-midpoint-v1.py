"""Complete gallery pilot of recovery-only midpoint rounding; failures retained."""
import hashlib
import json
import sys
from pathlib import Path

import warp as wp
from warpower import VariationalMeshingWorkspace
from hand_seed_recovery_midpoint_v1 import install

original = VariationalMeshingWorkspace.__init__
workspaces = []


def setup(self, *args, **kwargs):
    original(self, *args, **kwargs)
    install(self)
    workspaces.append(self)


VariationalMeshingWorkspace.__init__ = setup
run = VariationalMeshingWorkspace.run


def reset(self, *args, **kwargs):
    # This experiment is only for the explicitly recaptured gallery path.
    assert wp.get_stream(self.device).is_capturing
    self.core._tie_stats.zero_()
    return run(self, *args, **kwargs)


VariationalMeshingWorkspace.run = reset
from examples.vtm_gallery import main

try:
    main()
finally:
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    output.mkdir(parents=True, exist_ok=True)
    names = [__file__, '.work/vtm/hand_seed_recovery_midpoint_v1.py', '.work/vtm/representable_boundary_v1.py']
    record = dict(scope='Private recovery-only midpoint proposal. Existing fitting budget, objective, certification, extraction and quality guards remain unchanged. Ordinary growth/polish/quality stages use the original proposal; no global adoption.',
                  source_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in names},
                  recovery_midpoint_statistics=[m.core._tie_stats.numpy().tolist() for m in workspaces])
    (output / 'pilot.json').write_text(json.dumps(record, indent=2) + '\n')
