"""Compare the private Warp primitive with every exact hand candidate proof."""
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp
from warpower import SurfaceDomain
from proven_exterior_v2 import prove

w = Path('.work/vtm'); out = w / 'hand-proven-exterior-device-v2.json'; assert not out.exists()
rows = []
for name in ['hand-seed7', 'hand-seed7-midpoint']:
    source = w / (name + '-final-v1.npz'); d = np.load(source)
    refpath = w / (name + '-exterior-proof-v1.json'); ref = json.loads(refpath.read_text())
    domain = SurfaceDomain(d['input_points'], d['input_faces'])
    ids = np.array([r['source_index'] for r in ref['cells']])
    points = wp.array(d['points'], dtype=wp.vec3); tets = wp.array(d['full_tetrahedra'][ids], dtype=int)
    result = wp.empty(len(ids), dtype=int)
    wp.load_module(module=prove.module, device=domain.device)
    with wp.ScopedCapture() as capture:
        wp.launch(prove, len(ids), inputs=[domain.tree, points, tets, result])
    wp.capture_launch(capture.graph)
    actual = result.numpy()
    expected = np.array([r['proved_open_interior_exterior'] for r in ref['cells']], np.int32)
    np.testing.assert_array_equal(actual, expected)
    rows.append(dict(name=name, tested=len(ids), proved=ids[actual != 0].tolist(), matches_all_exact_proofs=True,
                     input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     oracle_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest()))
    out.write_text(json.dumps(dict(scope='Private captured primitive only; not yet recovery integration, robustness or performance validation.',
                                  source_sha256=hashlib.sha256((w/'proven_exterior_v2.py').read_bytes()).hexdigest(), cases=rows), indent=2)+'\n')
    print(rows[-1], flush=True)
