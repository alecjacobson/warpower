import runpy,sys
from prepared_group_queries import install
install()
module=sys.argv.pop(1)
runpy.run_module(module,run_name='__main__')
