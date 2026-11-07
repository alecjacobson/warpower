"""Private #161 opt-in of existing recertification on the full active route."""
import warpower._active_delaunay as active
from warpower._reuse_delaunay import ReuseDelaunayWorkspace

# Leave the 1024/2048 children unchanged. The full route owns independent
# recertified fragments; every call still has ordinary reconstruction fallback.
active.DelaunayWorkspace = ReuseDelaunayWorkspace
original_clear = active.ActiveDelaunayWorkspace.clear_reuse

def clear(self):
    original_clear(self)
    self._full.clear_reuse()

active.ActiveDelaunayWorkspace.clear_reuse = clear
from benchmarks.mesher import main
main()
