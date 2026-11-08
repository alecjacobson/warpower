"""Retain offline whole-exterior removal proposals for independent checks."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

w = Path('.work/vtm')
for name in ['hand-seed7', 'hand-seed7-midpoint']:
    source = w / (name + '-final-v1.npz')
    proof = w / (name + '-exterior-proof-v1.json')
    d = dict(np.load(source)); r = json.loads(proof.read_text())
    assert r['input_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    old = d['selected'] != 0
    removed = r['proved_exterior_source_indices']
    selected = old.copy(); selected[removed] = False
    d['selected'] = selected.astype(np.int32); d['source_indices'] = np.flatnonzero(selected).astype(np.int32)
    d['tetrahedra'] = d['full_tetrahedra'][selected]
    d['radius_ratio'] = d['input_quality'][selected]
    faces = [tuple(map(int, f)) for t in d['tetrahedra'] for f in
             [t[[1, 2, 3]], t[[0, 3, 2]], t[[0, 1, 3]], t[[0, 2, 1]]]]
    counts = Counter(tuple(sorted(f)) for f in faces)
    d['boundary'] = np.array([f for f in faces if counts[tuple(sorted(f))] == 1], np.int32)
    old_vertices = set(map(int, np.unique(d['full_tetrahedra'][old])))
    vertices = set(map(int, np.unique(d['tetrahedra'])))
    out = w / (name + '-outside-proposal-v1.npz'); assert not out.exists()
    np.savez_compressed(out, **d)
    meta = dict(scope='Offline removal proposal using exact whole-exterior proofs. Not a production output or complete recovered solve.',
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                proof_sha256=hashlib.sha256(proof.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                removed_source_indices=removed, orphaned_vertices=sorted(old_vertices-vertices))
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(name, meta, flush=True)
