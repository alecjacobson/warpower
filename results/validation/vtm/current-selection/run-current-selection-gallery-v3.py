import os
from current_selection_observer_v3 import install,report
install()
from examples.vtm_gallery import main
main()
report(os.environ['WARPOWER_SELECTION_OBSERVER'])
