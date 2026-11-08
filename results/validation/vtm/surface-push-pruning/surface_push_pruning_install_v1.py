"""Patch caller aliases before compilation, retaining the unmodified reference function."""
import importlib
from surface_push_pruning_v1 import pruned_closest_point
from warpower._surface_bvh import closest_point

MODULES = ['_domain', '_extraction', '_jitter', '_selective', '_surface_query_cache', '_topology_recovery', '_vtm_domain']


def install():
    for name in MODULES:
        module = importlib.import_module('warpower.'+name)
        assert module.closest_point is closest_point, name
        module.closest_point = pruned_closest_point
    # Tests' frozen reference imports remain original, so comparison is independent.
    from warpower import _surface_bvh
    assert _surface_bvh.closest_point is closest_point
