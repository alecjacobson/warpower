import json
from pathlib import Path
from warp._src.context import get_cuda_kernel_properties
from analytic_face_search import search_analytic
from warpower._proved_completion import search_unproved_faces
properties={name:get_cuda_kernel_properties(kernel,device='cuda:0',block_dim=32) for name,kernel in [('original',search_unproved_faces),('analytic',search_analytic)]}
Path('.work/vtm/analytic-face-properties.json').write_text(json.dumps(properties,indent=2)+'\n')
print(properties)
