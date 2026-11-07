"""Compare all published quality fields, including nonzero public RMS output."""
import hashlib
import json
import math
from pathlib import Path

root = Path.cwd()
w = root / '.work/vtm'
records = json.loads((w / 'surface-bound-reuse-final-independent.json').read_text())
result = {'absolute_tolerance': 1e-12, 'relative_tolerance': 1e-12, 'cases': {}}
for name, row in records['cases'].items():
    before_path = root / row['before_result']
    after_path = root / row['after_result']
    before = json.loads(before_path.read_text())['quality']
    after = json.loads(after_path.read_text())['quality']
    changes = []
    def compare(a, b, path='quality'):
        assert type(a) is type(b), (name, path, type(a), type(b))
        if isinstance(a, dict):
            assert a.keys() == b.keys(), (name, path)
            for key in a:
                compare(a[key], b[key], path + '/' + key)
        elif isinstance(a, list):
            assert len(a) == len(b), (name, path)
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + '/' + str(i))
        elif isinstance(a, float):
            assert math.isfinite(a) and math.isfinite(b), (name, path, a, b)
            assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (name, path, a, b)
            if a != b:
                changes.append({'path': path, 'before': a, 'after': b, 'absolute_difference': abs(a-b)})
        else:
            assert a == b, (name, path, a, b)
    compare(before, after)
    field = 'surface_rms_distance' if 'surface_rms_distance' in after else 'directed_surface_rms'
    assert after[field] > 0
    result['cases'][name] = {
        'before_sha256': hashlib.sha256(before_path.read_bytes()).hexdigest(),
        'after_sha256': hashlib.sha256(after_path.read_bytes()).hexdigest(),
        'public_rms_field': field, 'public_rms': after[field],
        'all_fields_match': True, 'floating_reduction_differences': changes,
    }
dest = w / 'surface-bound-reuse-public-quality.json'
assert not dest.exists()
dest.write_text(json.dumps(result, indent=2) + '\n')
print('All quality fields match across eight cases; public RMS remains nonzero.')
print('Largest absolute floating difference:', max((c['absolute_difference'] for r in result['cases'].values() for c in r['floating_reduction_differences']),default=0.0))
