import os
from install_prepared_hierarchy import install
install(os.environ['WARPOWER_PREPARED_TREE_NODES'] == 'polynomial')
from benchmarks.mesher import main
main()
