import os
from install_prepared_narrow_tree import install
install(int(os.environ['WARPOWER_PREPARED_TREE_WORKERS']))
from benchmarks.mesher import main
main()
