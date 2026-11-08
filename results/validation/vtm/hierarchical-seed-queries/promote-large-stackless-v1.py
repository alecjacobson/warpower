"""Promote the measured traversal; setup uses an explicit existing hierarchy."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm')
src=w/'large_stackless_v1.py'
assert hashlib.sha256(src.read_bytes()).hexdigest()=='b40dfd2327d0aa7e0675387d819ab28aa74cc3735596f58bffcf1ded444f7299'
assert json.loads((w/'large-stackless-pilot-v1-independent.json').read_text())['complete']
s=src.read_text().split('\ndef install():')[0].replace('"""Private #177 large-capacity probe: stackless groups with current bounded point filters."""','"""Traverse existing current site bounds in 32-site groups, without query stacks."""').replace('\nimport weakref\n','')
s=s.replace('    def __init__(self, *args, segment, **kwargs):','    """Requires the caller to rebuild the shared segment from current points."""\n\n    def __init__(self, *args, segment, **kwargs):')
s=s.replace('        assert segment.n == len(points) == tree.n\n        assert segment.ids.ptr == tree.ids.ptr and segment.leaf_size == 1\n        assert segment.leaf_base == 1 << (len(points) - 1).bit_length()', '        if (\n            segment.n != len(points)\n            or tree.n != len(points)\n            or segment.ids.ptr != tree.ids.ptr\n            or segment.leaf_size != 1\n            or segment.leaf_base != 1 << (len(points) - 1).bit_length()\n        ):\n            raise ValueError("Query segment must share the current point permutation")')
p=Path('warpower/_stackless_seeds.py');assert not p.exists();p.write_text(s.rstrip()+'\n')
for kind,name in [('flat','seeds'),('traversal','traversal')]:
 s=(w/f'test_large_stackless_{kind}_v1.py').read_text().replace('from large_stackless_v1 import','from warpower._stackless_seeds import')
 if kind=='flat':
  start=s.index('        names = {');end=s.index('\n    else:',start)
  s=s[:start]+'''        # Synthetic tube samples keep this regression independent of saved runs.
        n = int(case[5:])
        u, v = rng.uniform(0, 2 * np.pi, (2, n))
        r = 0.25 * np.sqrt(rng.uniform(0, 1, n))
        p = np.column_stack(((1 + r * np.cos(v)) * np.cos(u),
                             (1 + r * np.cos(v)) * np.sin(u), r * np.sin(v))).astype(np.float32)
        candidates = Delaunay(p).simplices
        candidates = candidates[np.linspace(0, len(candidates) - 1, 32, dtype=int)]'''+s[end:]
 p=Path(f'tests/test_stackless_{name}.py');assert not p.exists();p.write_text(s)
old=w/'handoff.json';archive=w/'handoff-before-stackless-promotion-v1.json';assert not archive.exists();archive.write_bytes(old.read_bytes())
d=json.loads(old.read_text());d['active_gpu_session']=None;d['active_cpu_sessions']=[]
d['active_milestone'].update(stage='Promoting validated private traversal to explicit setup; no weak registry in production',runtime_changed=True,live_processes=[],pilot='12 clean valid outputs, six pairs bitwise coordinates/canonical cells/roles/work equal; gains12.1%16kfixed,8.2%16kgraded,3.3%8k,1.8%reserved;smallneutral',profile='6253.568ms native kernel sum; check_stackless2190.346ms227calls255registers, all canonical/workequal; profile timings diagnostic',pending='Production106 tests/fullsuite, fresh final timing and standalone gates, native/memory, publishPR/selfmerge',next='Run production focused tests then commit runtime and draft PR')
old.write_text(json.dumps(d,indent=2)+'\n')
