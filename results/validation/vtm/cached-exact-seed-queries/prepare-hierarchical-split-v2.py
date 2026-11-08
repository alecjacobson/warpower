"""Bound one saved unresolved point per lane; overflow retains the existing rescan."""
import hashlib,json
from pathlib import Path
w=Path('.work/vtm');source=w/'hierarchical_split_v1.py';s=source.read_text()
s=s.replace('    pending: wp.array[wp.uint32],\n', '    pending: wp.array[wp.uint32],\n    unresolved: wp.array2d[int],\n')
s=s.replace('    uncertain = int(0)', '    first = int(-1)')
s=s.replace('        uncertain = 1', '        if first == -1:\n            first = q\n        else:\n            first = -2')
s=s.replace('    if uncertain != 0:\n        wp.atomic_or(pending, i, wp.uint32(1) << wp.uint32(lane))', '    if first != -1:\n        unresolved[i, lane] = first\n        wp.atomic_or(pending, i, wp.uint32(1) << wp.uint32(lane))')
start=s.index('def exact_check(');at=s.index('    a, b, c, d =',start)
s=s[:at]+'''    saved = unresolved[i, lane]
    if saved >= 0:
        if saved >= points.shape[0]:
            wp.atomic_or(error, 0, 16)
            wp.atomic_and(live, i, 0)
            return
        # This lane already checked the current owner/incidence/box and both
        # strict filters. No input changes between these ordered passes.
        result = symbolic_sphere(points, ids, saved)
        if result.valid == 0:
            wp.atomic_or(error, 0, 2)
            wp.atomic_and(live, i, 0)
        elif result.sign >= 0:
            wp.atomic_and(live, i, 0)
        return
    if saved != -2:
        wp.atomic_or(error, 0, 16)
        wp.atomic_and(live, i, 0)
        return
'''+s[at:]
s=s.replace('class HierarchicalSplitQueries(', 'class CachedHierarchicalSplitQueries(')
s=s.replace('        self._query_inputs = [*self._check_inputs, segment, self.pending]', '        self.unresolved = wp.empty((self.capacity, 32), dtype=int, device=self.device)\n        self._query_inputs = [*self._check_inputs, segment, self.pending, self.unresolved]')
s=s.replace('_stackless_seeds.StacklessSeedQueries = HierarchicalSplitQueries','_stackless_seeds.StacklessSeedQueries = CachedHierarchicalSplitQueries')
s=s.replace('Private #175 follow-up: hierarchical strict queries and ordered exact lanes.', 'Private #175 v2: cache one unresolved point per lane; overflow rescans.')
p=w/'hierarchical_split_v2.py';assert not p.exists();p.write_text(s)
# Stronger forced-invalid-box coverage runs both passes; v1 evidence stays intact.
s=(w/'test_hierarchical_split_masks_v1.py').read_text()
s=s.replace('from hierarchical_split_v1 import HierarchicalSplitQueries as StacklessSeedQueries', 'from hierarchical_split_v2 import CachedHierarchicalSplitQueries as StacklessSeedQueries')
s=s.replace('from hierarchical_split_v1 import exact_check as check_stackless', 'from hierarchical_split_v2 import exact_check, strict_check')
s=s.replace('                    prepared.pending.fill_(wp.uint32(0xFFFFFFFF))\n                    prepared.run()', '                    prepared.pending.fill_(wp.uint32(0xFFFFFFFF))\n                    prepared.unresolved.fill_(-177)\n                    prepared.run()')
s=s.replace('                elif kernel == "invalid":\n                    prepared.pending.fill_(wp.uint32(0xFFFFFFFF))', '                elif kernel == "invalid":\n                    prepared.pending.zero_()\n                    prepared.unresolved.fill_(-177)')
needle='''                    wp.launch(
                        check_stackless,
                        32 * capacity,
                        inputs=prepared._query_inputs,
                        device=device,
                        block_dim=block,
                    )'''
assert needle in s
s=s.replace(needle,'''                    for checker in [strict_check, exact_check]:
                        wp.launch(
                            checker,
                            32 * capacity,
                            inputs=prepared._query_inputs,
                            device=device,
                            block_dim=block,
                        )''')
p=w/'test_hierarchical_split_masks_v2.py';assert not p.exists();p.write_text(s)
s=(w/'test_hierarchical_split_dispatch_v1.py').read_text().replace('from hierarchical_split_v1 import HierarchicalSplitQueries','from hierarchical_split_v2 import CachedHierarchicalSplitQueries').replace('split = HierarchicalSplitQueries(', 'split = CachedHierarchicalSplitQueries(')
s=s.replace('            split.pending.fill_(wp.uint32(0xFFFFFFFF))','            split.pending.fill_(wp.uint32(0xFFFFFFFF))\n            split.unresolved.fill_(-177)')
# If formatting grouped this line with others, make sure poisoning is inserted.
if 'split.unresolved.fill_' not in s:
 needle='split.pending.fill_(wp.uint32(0xFFFFFFFF))';assert needle in s;s=s.replace(needle,needle+'\n            split.unresolved.fill_(-177)')
p=w/'test_hierarchical_split_dispatch_v2.py';assert not p.exists();p.write_text(s)
(w/'hierarchical-split-v2-preparation.json').write_text(json.dumps(dict(scope=__doc__,baseline_prototype_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),status='Prepared only; cache indices are private current-invocation evidence, never retained across changed input. Fixed32ints/candidate; negative2 is safe-rescan overflow, negative1 unavailable. Runtime unchanged.'),indent=2)+'\n')
