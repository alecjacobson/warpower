"""Private current-input geometry preparation; ordinary constructors untouched."""
import os

import warpower._reuse_delaunay as reuse
from seed_filter_prepared import PreparedSeeds

reuse.CertifiedSeeds=PreparedSeeds
if os.environ.get('WARPOWER_PREPARED_GALLERY'):
    from examples.vtm_gallery import main
else:
    from benchmarks.mesher import main
main()
