"""Private traversal installation followed by the unchanged public gallery."""
import sys
if '--candidate' in sys.argv:
    sys.argv.remove('--candidate')
    from large_stackless_v1 import install
    install()
from examples.vtm_gallery import main
main()
