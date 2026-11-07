import pytest
import warp as wp
from warpower._prepared_seeds import check
from seed_tree_queries import check_tree
for block in [32,256]:
    wp.load_module(module='seed_tree_queries',device='cuda:0',block_dim=block)
original=wp.launch

def launch(kernel,*args,**kwargs):
    return original(check_tree if kernel is check else kernel,*args,**kwargs)
wp.launch=launch
raise SystemExit(pytest.main(['-q','tests/test_facet_seeds.py','tests/test_reuse_delaunay.py']))
