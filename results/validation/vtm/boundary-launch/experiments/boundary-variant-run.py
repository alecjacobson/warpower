"""Run an unchanged benchmark with a private boundary launch-size override."""
from pathlib import Path
import runpy
import sys

probe = runpy.run_path(str(Path(__file__).with_name('boundary-reduction-probe.py')))
module = sys.argv[1]
sys.argv = sys.argv[1:]
with probe['variant'](False, 32):
    runpy.run_module(module, run_name='__main__')
