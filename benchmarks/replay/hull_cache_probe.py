"""Compatibility installer for the qualified private same-input hull cache."""

import warp as wp

from warpower._completion_hull_cache import HullSupportCache


def HullCache(completion):
    c = completion._certificate
    if c._hull_cache is None:
        c._hull_cache = HullSupportCache(len(c.points), len(c._boundary), c.device)
    return c._hull_cache


def install(workspace):
    cache = HullCache(workspace._reuse_completion)
    with wp.ScopedCapture(device=workspace.device) as capture:
        workspace._record_reuse()
    workspace._reuse_graph = capture.graph
    return cache
