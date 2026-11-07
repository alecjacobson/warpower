"""Private split-launch probe; preloaded kernels, complete resident replay only."""
import os
import warp as wp
from warpower._extraction import classify
from extraction_split import prepare_geometry, query_surface

for block in [32, 256]:
    wp.load_module(module='extraction_split', device='cuda:0', block_dim=block)
original = wp.launch


def launch(kernel, *args, **kwargs):
    if kernel is classify:
        original(prepare_geometry, *args, **kwargs)
        return original(query_surface, *args, **kwargs)
    return original(kernel, *args, **kwargs)


wp.launch = launch
if os.environ.get('WARPOWER_SPLIT_GALLERY'):
    from examples.vtm_gallery import main
else:
    from benchmarks.mesher import main
main()
