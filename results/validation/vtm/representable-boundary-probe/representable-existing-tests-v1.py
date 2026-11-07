"""Exercise existing workspace contracts with the private proposal installed."""
import sys
import pytest
from warpower import _vtm
from representable_boundary_v1 import install
original=_vtm.VTMWorkspace.__init__
def create(self,*args,**kwargs):
    original(self,*args,**kwargs)
    install(self)
_vtm.VTMWorkspace.__init__=create
raise SystemExit(pytest.main(['-q',*sys.argv[1:]]))
