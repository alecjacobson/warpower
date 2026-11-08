"""Fresh gallery solves with optional private current-owner nearest-site seeding."""
import hashlib,json,sys
from pathlib import Path
import warp as wp
candidate='--candidate' in sys.argv;instances=[]
if candidate:
 sys.argv.remove('--candidate')
 import warpower._boundary as boundary
 import boundary_owner_seed_kernels_v1 as kernels
 original=boundary.BoundaryWorkspace.__init__
 def initialize(self,*args,**kwargs):
  original(self,*args,**kwargs)
  with wp.ScopedDevice(self.device):
   self.surface_owners.fill_(-1);self.crease_owners.fill_(-1)
  instances.append(self)
 wp.load_module(module=kernels,device='cuda:0',block_dim=32)
 boundary.accumulate_samples=kernels.accumulate_seeded
 boundary.BoundaryWorkspace.__init__=initialize
from examples.vtm_gallery import main
main()
out=Path(sys.argv[sys.argv.index('--output')+1])/'owner-seed.json';assert not out.exists()
files=[Path(__file__)]
if candidate:files += [Path('.work/vtm/boundary_owner_seed_kernels_v1.py')]
out.write_text(json.dumps(dict(scope=__doc__,candidate=candidate,instances=len(instances),extra_persistent_arrays=0,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
