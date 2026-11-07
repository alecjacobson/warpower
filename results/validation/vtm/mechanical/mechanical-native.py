import json
from pathlib import Path
import torch
import warp as wp
from examples.vtm_inputs import counterbored_plate
from warpower import SurfaceDomain, VariationalMeshingWorkspace
w=Path('.work/vtm')
wp.init()
domain=SurfaceDomain(*counterbored_plate(),surface_subdivisions=16,crease_subdivisions=16)
mesh=VariationalMeshingWorkspace(domain,1000,initial_count=256,seed=2026,quality_rounds=4)
with wp.ScopedCapture() as cap:
 domain.refit()
 mesh.run()
wp.capture_launch(cap.graph)
wp.synchronize()
wp.capture_debug_dot_print(cap.graph,str(w/'mechanical-native.dot'),verbose=True)
torch.cuda.cudart().cudaProfilerStart()
for _ in range(3):wp.capture_launch(cap.graph)
wp.synchronize()
torch.cuda.cudart().cudaProfilerStop()
result=dict(valid=int(mesh.valid.numpy()[0]),error=int(mesh.error.numpy()[0]),counts=mesh.counts.numpy().tolist(),topology=mesh.topology.numpy().tolist())
assert result['valid']==1 and result['error']==0,result
(w/'mechanical-native-result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
