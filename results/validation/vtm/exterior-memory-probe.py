import sys,gc
from benchmarks.memcheck import configure
configure()
import warp as wp
import warpower
from tests.test_exterior_band import patch, transaction
from tests.test_domain import cube
mode=sys.argv[1]
p,t,ds,removed=patch()
e=warpower.ExtractionWorkspace(warpower.SurfaceDomain(*cube(5)),p,t)
if mode!='no-run':
 e.run();wp.synchronize_device(); print('first complete',flush=True)
if mode=='free':
 del e;gc.collect();wp.synchronize_device(); print('first freed',flush=True)
f,graph=transaction(p,t,ds,padding=1)
wp.synchronize_device();print('second complete',f.selected.numpy(),flush=True)
