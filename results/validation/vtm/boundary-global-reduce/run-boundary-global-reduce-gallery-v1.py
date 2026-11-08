"""Fresh gallery solve with optional private block-reduced global boundary diagnostics."""
import hashlib,json,sys
from pathlib import Path
import warp as wp
candidate='--candidate' in sys.argv
if candidate:
 sys.argv.remove('--candidate')
 import warpower._boundary as boundary
 import boundary_global_reduce_kernels_v1 as kernels
 wp.load_module(module=kernels,device='cuda:0',block_dim=32)
 boundary.accumulate_samples=kernels.accumulate_reduced
from examples.vtm_gallery import main
main()
out=Path(sys.argv[sys.argv.index('--output')+1])/'global-reduction.json';assert not out.exists()
files=[Path(__file__)]
if candidate:files += [Path('.work/vtm/boundary_global_reduce_kernels_v1.py')]
out.write_text(json.dumps(dict(scope=__doc__,candidate=candidate,extra_persistent_arrays=0,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
