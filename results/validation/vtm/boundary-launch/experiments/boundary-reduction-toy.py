"""Check padded 1D collective reductions before considering integration."""
import json
from pathlib import Path

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False, "fast_math": False})


@wp.kernel
def reduce_values(values: wp.array[wp.float64], enabled: wp.array[int], out: wp.array[wp.float64]):
    i = wp.tid()
    value = wp.float64(0.0)
    if i < values.shape[0] and enabled[i] != 0:
        value = values[i]
    total = wp.tile_sum(wp.tile(value))[0]
    maximum = wp.tile_max(wp.tile(value))[0]
    if i % wp.block_dim() == 0:
        wp.atomic_add(out, 0, total)
        wp.atomic_max(out, 1, maximum)


def main():
    wp.init()
    rows = []
    rng = np.random.default_rng(154)
    for block in (32, 64, 128, 256):
        for n in (1, 31, 32, 33, 65, 257, 1001):
            for zeros in (False, True):
                values = np.exp2(rng.uniform(-660, 660, n))
                enabled = (rng.random(n) > 0.2).astype(np.int32)
                if zeros:
                    enabled.fill(0)
                x = wp.array(values, dtype=wp.float64, device="cuda:0")
                mask = wp.array(enabled, dtype=int, device="cuda:0")
                output = wp.zeros(2, dtype=wp.float64, device="cuda:0")
                wp.launch(reduce_values, ((n + block - 1) // block) * block,
                          inputs=[x, mask, output], block_dim=block, device="cuda:0")
                result = output.numpy()
                expected = np.array([values[enabled != 0].sum(), values[enabled != 0].max(initial=0)])
                np.testing.assert_allclose(result, expected, rtol=3e-15, atol=0)
                rows.append(dict(block=block, n=n, zeros=zeros, result=result.tolist()))
    Path('.work/vtm/boundary-reduction-toy.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'{len(rows)} collective cases passed', flush=True)


if __name__ == '__main__':
    main()
