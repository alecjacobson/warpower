"""Create new immutable-version candidates without editing the retained v1 sources."""
from pathlib import Path

w=Path('.work/vtm')
s=(w/'proven_exterior_v1.py').read_text()
s=s.replace('from warpower import _predicates as predicates',
            'from warpower import _filtered_predicates as predicates\nfrom proven_exterior_filter_v2 import triangle')
start=s.index('@wp.func(inline=False)\ndef centroid_ray(')
end=s.index('\n\n@wp.func(inline=False)\ndef exterior',start)
s=s[:start]+'''@wp.func(inline=False)
def centroid_ray(tree: SurfaceTree, p0: wp.vec3, p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, direction: wp.vec3):
    count = int(0)
    for f in range(tree.n):
        q0 = wp.vec3(tree.points[tree.indices[3*f]])
        q1 = wp.vec3(tree.points[tree.indices[3*f+1]])
        q2 = wp.vec3(tree.points[tree.indices[3*f+2]])
        result = triangle(p0, p1, p2, p3, q0, q1, q2, direction)
        if result[1] != 0:
            return result
        count += result[0]
    return wp.vec2i(count, 0)
'''+s[end:]
sources={'proven_exterior_v2.py':s}
for name in ['test_proven_exterior_v1.py','check-proven-exterior-device-v1.py',
             'proven_exterior_transaction_v1.py','full_proven_recovery_v1.py','run-full-proven-recovery-v1.py']:
    dest=name.replace('_v1','_v2').replace('-v1','-v2')
    s=(w/name).read_text()
    for a,b in [('proven_exterior_v1','proven_exterior_v2'),
                ('proven_exterior_transaction_v1','proven_exterior_transaction_v2'),
                ('full_proven_recovery_v1','full_proven_recovery_v2'),
                ('hand-proven-exterior-device-v1.json','hand-proven-exterior-device-v2.json')]:
        s=s.replace(a,b)
    if name=='run-full-proven-recovery-v1.py':
        s=s.replace("paths = [__file__,", "paths = [__file__, '.work/vtm/proven_exterior_filter_v2.py',")
    sources[dest]=s
for name,s in sources.items():
    p=w/name;assert not p.exists();p.write_text(s)
print('created',list(sources))
