"""Private launch adapter for whole-pipeline experiments; no runtime edit."""
import sys
import warp as wp
from warpower._selective import classify_retained
import cooperative_retained_v1 as kernels


def install():
    wp.load_module(module=kernels, device='cuda:0', block_dim=32)
    original = wp.launch

    def launch(kernel, dim=None, inputs=(), outputs=(), **kwargs):
        if kernel is classify_retained:
            kwargs.pop('block_dim', None)
            return wp.launch_tiled(kernels.retained_4, dim=(dim+7)//8, inputs=inputs,
                                   outputs=outputs, block_dim=32, **kwargs)
        return original(kernel, dim=dim, inputs=inputs, outputs=outputs, **kwargs)

    wp.launch = launch


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    install()
    if mode == 'tests':
        import pytest
        raise SystemExit(pytest.main(sys.argv[1:]))
    if mode == 'mesher':
        from benchmarks.mesher import main
    elif mode == 'gallery':
        from examples.vtm_gallery import main
    else:
        raise ValueError(mode)
    main()
