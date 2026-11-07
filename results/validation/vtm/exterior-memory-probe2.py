import sys,gc
from benchmarks.memcheck import configure
mode=sys.argv[1]; configure(stack_bytes=131072 if mode=='stack' else 32768)
import warp as wp
import warpower
from tests.test_exterior_band import patch, transaction
from tests.test_domain import cube
p,t,ds,removed=patch()
e=warpower.ExtractionWorkspace(warpower.SurfaceDomain(*cube(5)),p,t)
e.run();wp.synchronize_device(); print('first complete',flush=True)
if mode=='freegraph':
 e.graph=None;gc.collect();wp.synchronize_device()
if mode=='full':
 f=warpower.ExtractionWorkspace(warpower.SurfaceDomain(*cube(5)),p,t,exterior_recovery=True)
 f.run()
elif mode=='plain':
 f=warpower.ExtractionWorkspace(warpower.SurfaceDomain(*cube(5)),p,t)
 f.run()
else:
 f,graph=transaction(p,t,ds,padding=0 if mode=='no-padding' else 1)
wp.synchronize_device(); print('second complete',f.selected.numpy(),flush=True)
