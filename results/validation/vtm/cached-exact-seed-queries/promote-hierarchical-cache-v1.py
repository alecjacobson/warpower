"""Promote the independently checked private layout while retaining runtime guards."""
import hashlib
from pathlib import Path
w=Path('.work/vtm')
p=Path('warpower/_stackless_seeds.py');old=p.read_text()
v2=(w/'hierarchical_split_v2.py').read_text()
assert hashlib.sha256(v2.encode()).hexdigest()=='c4138f62f1a70b83cfe45b10748be773674ad1c05d963a9aeba0197f0ae12e98'
prefix=old[:old.index('@wp.kernel')]
kernels=v2[v2.index('@wp.kernel'):v2.index('class CachedHierarchicalSplitQueries')]
cls=old[old.index('class StacklessSeedQueries'):old.index('    def run(self):')]
cls=cls.replace('        self._query_inputs = [*self._check_inputs, segment]', '''        # Only flagged lanes read these current-invocation indices. The first
        # unresolved point is cached; -2 means the lane needs a complete rescan.
        self.pending = wp.zeros(self.capacity, dtype=wp.uint32, device=self.device)
        self.unresolved = wp.empty((self.capacity, 32), dtype=int, device=self.device)
        self._query_inputs = [*self._check_inputs, segment, self.pending, self.unresolved]''')
run=v2[v2.index('    def run(self):'):v2.index('\n\ndef install():')]
p.write_text(prefix+kernels+cls+run+'\n')
s=(w/'test_hierarchical_split_masks_v2.py').read_text().replace('from hierarchical_split_v2 import CachedHierarchicalSplitQueries as StacklessSeedQueries\nfrom hierarchical_split_v2 import exact_check, strict_check\n','').replace('from warpower._site_bvh import SiteHierarchy','from warpower._site_bvh import SiteHierarchy\nfrom warpower._stackless_seeds import StacklessSeedQueries, exact_check, strict_check')
Path('tests/test_stackless_seeds.py').write_text(s)
s=(w/'test_hierarchical_split_dispatch_v2.py').read_text().replace('from hierarchical_split_v2 import CachedHierarchicalSplitQueries\n','').replace('from warpower._site_bvh import SiteHierarchy','from warpower._site_bvh import SiteHierarchy\nfrom warpower._grouped_seeds import GroupedSeedQueries').replace('current = StacklessSeedQueries(\n            points, tets, count, tree, live, error, segment=segment, block_dim=block\n        )','current = GroupedSeedQueries(\n            points, tets, count, tree, live, error, block_dim=block\n        )').replace('split = CachedHierarchicalSplitQueries(', 'split = StacklessSeedQueries(')
assert 'current = GroupedSeedQueries(' in s
p=Path('tests/test_stackless_dispatch.py');assert not p.exists();p.write_text(s)
