import runpy,sys
from install_float32_combined import install
install()
module=sys.argv.pop(1)
runpy.run_module(module,run_name='__main__')
