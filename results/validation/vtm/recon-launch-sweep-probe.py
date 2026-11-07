"""Private follow-up experiment; do not run alongside other GPU jobs."""
import os
import runpy

import warp as wp
from warpower import _facet_completion, _facet_seeds, _refinement, _vtm_domain

block = int(os.environ['WARPOWER_RECON_BLOCK_DIM'])
target = os.environ.get('WARPOWER_RECON_BLOCK_TARGET', 'all')
groups = {
    'extend': (_facet_completion.extend_faces,),
    'search': (_facet_completion.search_faces,),
    'seeds': (_facet_seeds.check_seeds,),
    'constrain': (_vtm_domain.constrain,),
    'candidates': (_refinement.candidates,),
}
targets = sum(groups.values(), ()) if target == 'all' else groups[target]
original = wp.launch


def launch(*args, **kwargs):
    kernel = args[0] if args else kwargs.get('kernel')
    if kernel in targets:
        kwargs['block_dim'] = block
    return original(*args, **kwargs)


wp.launch = launch
runpy.run_module('benchmarks.mesher', run_name='__main__')
