"""Private experiment in VTM's reuse strategy; ordinary constructors untouched."""
import os

import warpower._reuse_delaunay as reuse
from seed_filter_split import SplitSeeds

reuse.CertifiedSeeds=SplitSeeds
if os.environ.get('WARPOWER_SPLIT_GALLERY'):
    from examples.vtm_gallery import main
else:
    from benchmarks.mesher import main
main()
