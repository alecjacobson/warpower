"""Create fresh qualification drivers from the previously reviewed complete checks."""
import ast
import hashlib
import json
from pathlib import Path

w=Path(__file__).parent; records=[]
copies=[
 ('run-surface-sample-production-tests-v2.py','run-surface-push-production-tests-v1.py'),
 ('run-surface-sample-final-v1.py','run-surface-push-final-v1.py'),
 ('run-surface-sample-qualification-v1.py','run-surface-push-qualification-v1.py'),
 ('run-surface-sample-memory-v1.py','run-surface-push-memory-v1.py'),
 ('run-surface-sample-loop-native-v1.py','run-surface-push-loop-native-v1.py'),
 ('run-surface-sample-uniform-native-v1.py','run-surface-push-uniform-native-v1.py'),
 ('profile-surface-sample-uniform-v1.py','profile-surface-push-uniform-v1.py'),
 ('check-surface-sample-native-full-v1.py','check-surface-push-native-full-v1.py'),
 ('check-surface-sample-loop-v1.py','check-surface-push-loop-v1.py'),
 ('check-surface-sample-pairs-v4.py','check-surface-push-pairs-v1.py'),
 ('run-surface-sample-pipeline-v1.py','run-surface-push-pipeline-v1.py'),
 ('run-surface-sample-independent-v1.py','run-surface-push-independent-v1.py'),
 ('check-surface-sample-qualified-v1.py','check-surface-push-qualified-v1.py'),
]
for old_name,new_name in copies:
    source=w/old_name; target=w/new_name; assert not target.exists()
    original=source.read_text()
    text=original.replace('surface-sample','surface-push')
    # The prior production package used v2 after a test-only fixture cleanup.
    for old,new in [('surface-push-production-tests-v2','surface-push-production-tests-v1'),('surface-push-production-source-v2','surface-push-production-source-v1'),('surface-push-production-loop-v2','surface-push-production-loop-v1'),('surface-push-production-full-suite-v2','surface-push-production-full-suite-v1'),('check-surface-push-pairs-v4','check-surface-push-pairs-v1')]:
        text=text.replace(old,new)
    if new_name=='run-surface-push-production-tests-v1.py':
        text=text.replace("'surface-push-production-'+name+'-v2", "'surface-push-production-'+name+'-v1")
        text=text.replace("1387,1200", "1387,1800")
        text=text.replace('committed sample-parallel quadrature','committed exact nearest traversal')
    if new_name=='run-surface-push-memory-v1.py':
        old="('kernels',['tests/test_surface_quadrature.py'],41,600)"
        new="('kernels',['tests/test_surface_bound_reuse.py','tests/test_surface_query_cache.py','tests/test_ordered_surface_queries.py','tests/test_surface_quadrature.py'],116,600)"
        assert text.count(old)==1; text=text.replace(old,new)
        text=text.replace('Audit quadrature arithmetic/count/error contracts','Audit nearest, cache, ordering and quadrature contracts')
    if new_name=='profile-surface-push-uniform-v1.py':
        old="Path('warpower/_surface_query_cache.py')"
        assert text.count(old)==1; text=text.replace(old,old+",Path('warpower/_surface_bvh.py')")
        text=text.replace('per-sample boundary quadrature','exact child-push pruning')
    if new_name=='run-surface-push-final-v1.py':
        text=text.replace('Production quadrature complete-solve','Production exact nearest-traversal complete-solve')
    if new_name=='check-surface-push-loop-v1.py':
        text=text.replace('parallel-quadrature loop','exact nearest-traversal loop')
    if new_name=='check-surface-push-pairs-v1.py':
        text=text.replace('paired quadrature trajectories','paired nearest-traversal trajectories')
    # Original scientific scripts remain immutable and hashes bind this derivation.
    ast.parse(text)
    target.write_text(text)
    records.append(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),target=str(target),target_sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
out=w/'surface-push-qualification-preparation-v1.json'; assert not out.exists()
out.write_text(json.dumps(dict(scope=__doc__,files=records),indent=2)+'\n')
print('Prepared',len(records),'fresh drivers. No checks executed by this generator.')
