"""Private setup binding before unchanged benchmark setup."""
import sys
if '--candidate' in sys.argv:
    sys.argv.remove('--candidate')
    from prepared_new_cell_integration_v2 import install
    install()
from examples.vtm_gallery import main
main()
