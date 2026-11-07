from benchmarks.memcheck import configure
configure(stack_bytes=131072, disable_mempool=True)
from benchmarks.control_loop import main
main()
