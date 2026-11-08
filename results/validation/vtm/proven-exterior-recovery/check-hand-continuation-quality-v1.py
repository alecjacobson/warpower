"""Independent physical quality and feature fit for the offline manifold proposal."""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from benchmarks.feature_quality import measure as feature_measure
from benchmarks.mesh_quality import measure
from benchmarks.surface_distance import NativeSurfaceDistance
from examples.vtm_inputs import write_off

w = Path('.work/vtm')
p = argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
source=args.input; out=args.output
assert not out.exists()
d = np.load(source)
scratch = Path('/opt/warpower-generated-cache/validation-derived'); scratch.mkdir(parents=True, exist_ok=True)
distance = NativeSurfaceDistance(w / 'surface-distance', scratch)
quality = measure(d['points'], d['tetrahedra'], d['input_points'], d['input_faces'], distance_query=distance)
assert quality['components'] == 1 and quality['boundary_euler'] == 2
features = feature_measure(d['points'], d['boundary'], d['roles'], np.unique(d['tetrahedra']),
                           d['input_points'], d['input_faces'], feature_angle=45, corner_angle=45,
                           samples_per_edge=16, distance_query=distance)
with tempfile.TemporaryDirectory(prefix='hand-outside-surface-', dir=scratch) as tmp:
    path = Path(tmp) / 'boundary.off'; write_off(path, d['points'], d['boundary'])
    output = subprocess.check_output([str(w / 'surface-reference'), str(path)], text=True)
assert int(output.split()[2]) == 0
record = dict(scope='Private continuation/proof-removal proposal; not a complete fresh recovered solve or performance result.',
              input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              quality=quality, features=features, distance_oracle=distance.metadata,
              boundary_self_intersection_output=output, boundary_self_intersection_pairs=0)
out.write_text(json.dumps(record, indent=2) + '\n')
print(dict(minimum=quality['radius_ratio_percentiles']['0'], rms=quality['two_sided_area_weighted_rms'],
           components=quality['components'], euler=quality['boundary_euler']), flush=True)
