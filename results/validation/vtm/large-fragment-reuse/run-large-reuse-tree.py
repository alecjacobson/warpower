"""Private #161 full-route reuse with hierarchical current-input seed queries."""
import warp as wp
import warpower._active_delaunay as active
from warpower._reuse_delaunay import ReuseDelaunayWorkspace
from warpower._prepared_seeds import check
from seed_tree_queries import check_tree

# Preload separately, so only resident replay timings are compared.
wp.load_module(module='seed_tree_queries',device='cuda:0',block_dim=32)
original_launch=wp.launch

def launch(kernel,*args,**kwargs):
    if kernel is check and kwargs['inputs'][0].shape[0]>2048:
        return original_launch(check_tree,*args,**kwargs)
    return original_launch(kernel,*args,**kwargs)

wp.launch=launch
active.DelaunayWorkspace=ReuseDelaunayWorkspace
original_clear=active.ActiveDelaunayWorkspace.clear_reuse

def clear(self):
    original_clear(self)
    self._full.clear_reuse()

active.ActiveDelaunayWorkspace.clear_reuse=clear
from benchmarks.mesher import main
main()
