"""Private layout selection before unchanged benchmark setup."""
import sys
if '--candidate' in sys.argv:
    sys.argv.remove('--candidate')
    from hierarchical_split_v1 import install
    install()
from benchmarks.mesher import main
main()
