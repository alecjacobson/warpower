"""Prepare private unchanged-query and explicit fallback-mask tests; no GPU work."""
from pathlib import Path
w=Path('.work/vtm')
s=Path('tests/test_stackless_seeds.py').read_text().replace('from warpower._stackless_seeds import StacklessSeedQueries, check_stackless','from hierarchical_split_v1 import HierarchicalSplitQueries as StacklessSeedQueries\nfrom hierarchical_split_v1 import exact_check as check_stackless')
s=s.replace('                    prepared.run()', '                    prepared.pending.fill_(wp.uint32(0xFFFFFFFF))\n                    prepared.run()')
s=s.replace('                elif kernel == "invalid":', '                elif kernel == "invalid":\n                    prepared.pending.fill_(wp.uint32(0xFFFFFFFF))')
p=w/'test_hierarchical_split_masks_v1.py';assert not p.exists();p.write_text(s)
s=(w/'test_prepared_split_dispatch.py').read_text()
s=s.replace('from warpower import _prepared_seeds as ps','from warpower import _bvh\nfrom warpower._site_bvh import SiteHierarchy\nfrom warpower._stackless_seeds import StacklessSeedQueries')
s=s.replace('from warpower._bvh import Tree\n','')
s=s.replace('from prepared_split_queries import SplitPreparedSeedQueries','from hierarchical_split_v1 import HierarchicalSplitQueries')
s=s.replace('        tree=Tree();tree.owners=wp.array(np.arange(len(p),dtype=np.int32),dtype=int)', '''        hierarchy=SiteHierarchy(points)
        tree=hierarchy.tree
        tree.owners=wp.array(np.arange(len(p),dtype=np.int32),dtype=int)
        identity=wp.array(np.arange(len(p),dtype=np.int32),dtype=int)
        wp.copy(tree.ids,identity,count=len(p))
        segment=hierarchy._segment
        weights=wp.zeros(len(p),dtype=float)
        wp.load_module(module=_bvh,device='cuda:0',block_dim=block)
        wp.launch(_bvh.make_leaves,segment.leaf_base,inputs=[points,weights,segment],block_dim=block)
        level=segment.leaf_base//2
        while level:
            wp.launch(_bvh.merge_level,level,inputs=[segment,level],block_dim=block)
            level//=2''')
s=s.replace('ps.PreparedSeedQueries(points,tets,count,tree,live,error,block_dim=block)','StacklessSeedQueries(points,tets,count,tree,live,error,segment=segment,block_dim=block)')
s=s.replace('SplitPreparedSeedQueries(points,tets,count,tree,live,error,block_dim=block)','HierarchicalSplitQueries(points,tets,count,tree,live,error,segment=segment,block_dim=block)')
p=w/'test_hierarchical_split_dispatch_v1.py';assert not p.exists();p.write_text(s)
