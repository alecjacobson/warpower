import json
from pathlib import Path
from warp._src.context import get_cuda_kernel_properties
from warpower._prepared_seeds import check
from prepared_split_queries import strict_check,exact_check
p={name:get_cuda_kernel_properties(k,device='cuda:0',block_dim=32) for name,k in [('original',check),('strict',strict_check),('exact',exact_check)]}
Path('.work/vtm/prepared-split-properties.json').write_text(json.dumps(p,indent=2)+'\n')
print(p)
