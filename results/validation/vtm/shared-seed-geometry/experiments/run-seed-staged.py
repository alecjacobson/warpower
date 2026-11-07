import os
import warpower._reuse_delaunay as reuse
from seed_filter_staged import StagedSeeds

reuse.CertifiedSeeds = StagedSeeds
if os.environ.get('WARPOWER_STAGED_GALLERY'):
    from examples.vtm_gallery import main
else:
    from benchmarks.mesher import main
main()
