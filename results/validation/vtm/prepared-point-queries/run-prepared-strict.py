import os
from install_prepared_strict import install
install(os.environ['WARPOWER_PREPARED_STOP'] == '1')
from benchmarks.mesher import main
main()
