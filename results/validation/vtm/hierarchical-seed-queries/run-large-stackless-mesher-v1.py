"""Private traversal installation followed by the unchanged public benchmark."""
import sys
if '--candidate' in sys.argv:
    sys.argv.remove('--candidate')
    from large_stackless_v1 import install
    install()
from benchmarks.mesher import main
main()
