"""Run one unchanged test with explicit instrumentation and timed stack evidence."""
import faulthandler,os,sys
faulthandler.enable()
faulthandler.dump_traceback_later(60,repeat=True)
from benchmarks.memcheck import configure
configure(stack_bytes=131072,disable_mempool=os.environ.get('WARPOWER_AUDIT_DISABLE_POOL')=='1')
import pytest
try:
 raise SystemExit(pytest.main(['-q','-s']+sys.argv[1:]))
finally:
 faulthandler.cancel_dump_traceback_later()
