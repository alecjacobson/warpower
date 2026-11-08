"""Exercise existing workspace contracts with the private proposal installed."""
import sys
import pytest
from warpower import _vtm
from local_backtracking_v2 import install
original=_vtm.VTMWorkspace.__init__
def create(self,*args,**kwargs):
    original(self,*args,**kwargs)
    if self.selection is not None:install(self,midpoint=False)
_vtm.VTMWorkspace.__init__=create
raise SystemExit(pytest.main(['-q',*sys.argv[1:]]))
