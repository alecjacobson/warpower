import json
from pathlib import Path
from warp._src.context import get_cuda_kernel_properties
from warpower._proved_completion import search_unproved_faces
properties=get_cuda_kernel_properties(search_unproved_faces,device='cuda:0',block_dim=32)
Path('.work/vtm/analytic-production-properties.json').write_text(json.dumps(properties,indent=2)+'\n')
print(properties)
