"""Private paired fresh/cache queries; diagnostic execution only, no timing claim."""
import hashlib
import json
from pathlib import Path
import sys
import warp as wp
from warpower import _surface_query_cache as production
import surface_query_cache_paired_audit_kernels_v1 as audited

instances=[]
original_init=production.SurfaceQueryCache.__init__
def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    with wp.ScopedDevice(self.device):
        self.audit_statistics=wp.zeros(8,dtype=int)
        wp.load_module(module=audited,device=self.device,block_dim=32)
    instances.append(self)
mapping={production.classify_cached:audited.classify_cached,
         production.retained_cached:audited.retained_cached,
         production.selection_cached:audited.selection_cached}
def launch(self,kernel,dim,args,block_dim=32):
    wp.launch(mapping[kernel],dim,
              inputs=[*args,self.current,self.previous,self.table,self.audit_statistics],
              device=self.device,block_dim=block_dim)
    self.after(args[5])
production.SurfaceQueryCache.__init__=initialize
production.SurfaceQueryCache.launch=launch
from examples.vtm_gallery import main
main()
output=Path(sys.argv[sys.argv.index('--output')+1])/'paired-query-audit.json'
assert not output.exists()
columns=['ray_calls','ray_hits','ray_mismatch','nearest_calls','nearest_hits',
         'nearest_face_mismatch','nearest_distance_bits_mismatch','nearest_point_bits_mismatch']
rows=[dict(capacity=c.capacity,statistics=c.audit_statistics.numpy().tolist()) for c in instances]
files=[Path(__file__),Path(audited.__file__),Path(production.__file__)]
result=dict(scope=__doc__,columns=columns,instances=rows,
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
output.write_text(json.dumps(result,indent=2)+'\n')
assert all(all(r['statistics'][i]==0 for i in [2,5,6,7]) for r in rows),result
print(json.dumps(result),flush=True)
