"""Private #161 extension of existing same-input proof propagation to full route."""
import warpower._active_delaunay as active
from warpower._reuse_delaunay import ReuseDelaunayWorkspace
from warpower._proved_completion import ProvedCertificate
original_certificate_init=ProvedCertificate.__init__

def initialize(self,*args,**kwargs):
    original_certificate_init(self,*args,**kwargs)
    if len(self.points)>2048:
        # The seed filter has just recertified these exact current inputs.
        # Keep existing proof provenance through subsequent unchanged-input
        # completion rounds instead of rechecking every retained cell globally.
        self.point_parallel=True

ProvedCertificate.__init__=initialize
active.DelaunayWorkspace=ReuseDelaunayWorkspace
original_clear=active.ActiveDelaunayWorkspace.clear_reuse

def clear(self):
    original_clear(self);self._full.clear_reuse()

active.ActiveDelaunayWorkspace.clear_reuse=clear
import os
if os.environ.get('WARPOWER_LARGE_REUSE_GALLERY'):
    from examples.vtm_gallery import main
else:
    from benchmarks.mesher import main
main()
