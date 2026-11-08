"""Install private setup before the unchanged benchmark."""
import sys
if '--candidate' in sys.argv:
    sys.argv.remove('--candidate')
    from larger_prefix_v1 import install
    install()
from examples.vtm_gallery import main
main()
