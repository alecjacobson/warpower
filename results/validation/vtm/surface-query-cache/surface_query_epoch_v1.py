"""Device refit generation; saturation permanently disables query reuse."""
import warp as wp
wp.set_module_options({'enable_backward':False,'fast_math':False})

@wp.kernel
def advance(version:wp.array[wp.int64]):
    if version[0]<wp.int64(9223372036854775807):version[0]+=wp.int64(1)
