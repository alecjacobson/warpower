"""Unchanged pytest cases in a fresh sanitizer process, with Python stack diagnostics."""
import faulthandler
import sys
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
from benchmarks.memcheck import configure
configure(stack_bytes=131072)
import pytest
try:
    raise SystemExit(pytest.main(['-q', '-s'] + sys.argv[1:]))
finally:
    faulthandler.cancel_dump_traceback_later()
