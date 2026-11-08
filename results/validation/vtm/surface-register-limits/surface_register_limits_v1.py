"""Install private kernel choices before capture; production source stays unchanged."""
import importlib
import warp as wp
from warpower import _ordered_surface_queries as ordered

BASELINE=dict(ordered.ORDERED)
MODULES={}

def install(limit):
    if limit==0:
        ordered.ORDERED.update(BASELINE)
    else:
        module=MODULES.setdefault(limit,importlib.import_module(f'surface_register_limit_{limit}_v1'))
        for block in [32,256]:wp.load_module(module=module.__name__,device='cuda:0',block_dim=block)
        ordered.ORDERED.update({k:getattr(module,v.func.__name__) for k,v in BASELINE.items()})
    return {k.func.__name__:wp.get_cuda_kernel_properties(v,device='cuda:0',block_dim=32) for k,v in ordered.ORDERED.items()}
