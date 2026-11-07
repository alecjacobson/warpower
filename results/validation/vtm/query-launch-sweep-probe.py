"""Private launch-configuration experiment; geometry and buffers are unchanged."""
import os
import runpy
import warp as wp
from warpower import _extraction, _selective
block = int(os.environ['WARPOWER_QUERY_BLOCK_DIM'])
original = wp.launch
targets = (_extraction.classify, _extraction.surface_error, _selective.classify_retained)
def launch(*args, **kwargs):
    kernel = args[0] if args else kwargs.get('kernel')
    if kernel in targets:
        kwargs['block_dim'] = block
    return original(*args, **kwargs)
wp.launch = launch
runpy.run_module('benchmarks.mesher', run_name='__main__')
