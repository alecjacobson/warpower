"""Private #170 whole-solve pilot, after upfront query compilation."""
import os
from install_prepared_point import install
install(os.environ['WARPOWER_PREPARED_POINT_MODE'])
from benchmarks.mesher import main
main()
