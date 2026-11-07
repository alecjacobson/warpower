"""Query compiled kernel resources, separately from timing and graph audits."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import warp as wp
from warpower import _extraction, _selective

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists()
rows = []
for kernel in [_extraction.classify, _extraction.classify_selection,
               _extraction.surface_error, _extraction.selection_surface_status,
               _selective.classify_retained]:
    props = wp.get_cuda_kernel_properties(kernel, device='cuda:0', block_dim=32)
    rows.append(dict(kernel=kernel.key, block_dim=32, **props))
result = dict(runtime=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              warp_version=wp.__version__, device=wp.get_device('cuda:0').name,
              scope='CUDA function attributes for the compiled block-size32 variant. Local-memory bytes per thread are not a measurement of occupied memory, traffic or occupancy. Query is outside graph execution.',
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), kernels=rows)
a.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(rows, indent=2))
