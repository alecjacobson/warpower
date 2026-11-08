"""Private layout selection before unchanged gallery setup."""
import sys
if '--candidate' in sys.argv:
    sys.argv.remove('--candidate')
    from hierarchical_split_v1 import install
    install()
from examples.vtm_gallery import main
main()
