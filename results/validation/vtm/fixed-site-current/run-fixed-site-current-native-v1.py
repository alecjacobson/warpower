"""Existing native benchmark with the independently validated AABB distance oracle."""
import hashlib
import json
from pathlib import Path

import numpy as np
import benchmarks.cpu_mesher as benchmark
from benchmarks.surface_distance import NativeSurfaceDistance

root=Path.cwd();w=root/'.work/vtm'
build=json.loads((root/'results/validation/vtm/cpu-mesh3-build.json').read_text())
assert hashlib.sha256((root/'benchmarks/cpu/mesh3.cpp').read_bytes()).hexdigest()==build['source_sha256']
for mode in ['sequential','parallel']:
    binary=w/f'cpu-mesh3-target-{mode}'
    assert hashlib.sha256(binary.read_bytes()).hexdigest()==next(r['binary_sha256'] for r in build['builds'] if r['mode']==mode)
p,f=benchmark.read_surface(root/'results/validation/vtm/cpu-mesh3-torus-input.off')
gpu=np.load(w/'fixed-site-current-larger-v1/torus8192-progressive.npz')
np.testing.assert_array_equal(p.astype(np.float32),gpu['input_points'])
np.testing.assert_array_equal(f,gpu['input_faces'])
oracle=NativeSurfaceDistance(w/'surface-distance',w)
original=benchmark.measure
def measure(*args,**kwargs):
    result=original(*args,distance_query=oracle,**kwargs)
    result['distance_oracle']=oracle.metadata
    return result
benchmark.measure=measure
benchmark.main()
