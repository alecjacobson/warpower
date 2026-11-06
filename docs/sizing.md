# Graded sizing fields

`SizingField` and `SizingQueryWorkspace` are optional, independently callable
GPU services. They implement a sampled version of the Lipschitz extension in
[Alliez et al., Eq. 9](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf):

```
mu(x) = clamp(min_i(value[i] + K * distance(x, sample[i])), minimum, maximum)
```

Each sample is an **upper constraint**, not an interpolation knot. A lower
neighboring cone can reduce the value at another sample. `K=0` gives the constant
minimum sample value. The envelope, including the two constant clamps, is
K-Lipschitz in exact arithmetic; stored query values have float32 rounding.
Sizing describes relative edge lengths. It is separate from additive power
weights, and graded VTM still constructs ordinary Delaunay connectivity.

```python
import numpy as np
import warp as wp
from warpower import SizingField, SizingQueryWorkspace

# Allocation and optional host uploads happen once, before capture.
anchors = wp.array([[0, 0, 0], [1, 0, 0]], dtype=wp.vec3, device="cuda:0")
sizes = wp.array([0.2, 0.5], dtype=float, device="cuda:0")
queries = wp.array(np.random.default_rng(0).random((1000, 3)),
                   dtype=wp.vec3, device="cuda:0")
field = SizingField(anchors, sizes, gradation=0.3, minimum=0.02, maximum=2.0)
query = SizingQueryWorkspace(field, queries)
with wp.ScopedCapture() as capture:
    # A producer may write anchors, sizes, and queries here.
    field.refit()
    query.run()
    # A consumer may read query.values and query.error here.
wp.capture_launch(capture.graph)
```

Positions accept the same CUDA arrays or setup-time inputs as `SurfaceDomain`.
Sizing samples must be a contiguous CUDA float32 array with one value per
position. They must be finite and strictly positive. `gradation` is a fixed
setup parameter in `[0, 1e30]`; clamps satisfy
`1e-30 <= minimum <= maximum <= 1e30`. Defaults are `K=1`, minimum `1e-6`,
maximum `1e30`; choose the clamps for the coordinate scale of the model.

`refit()` validates and copies current samples, then rebuilds a Morton hierarchy
entirely on device. Negative sample values act as hierarchy bounds so queries
can prune cones that cannot lower the current result. K=0 needs only the root
minimum. Radix scratch is owned by capture setup; replay has no allocation,
host callback, readback, or synchronization inside either service. Sample and
query storage stays fixed. Query workspaces do not refit the field implicitly.

Device errors are bitmasks:

| Service | Bit | Meaning |
|---|---:|---|
| Field | 1 | Nonfinite sample position |
| Field | 2 | Nonfinite or nonpositive sample value |
| Query | 1 | Invalid field |
| Query | 2 | Nonfinite query position |

An invalid field gives zero for every query. An invalid query position gives
zero for that position and sets the shared query error; valid positions still
receive values. A subsequent valid refit/run clears prior errors. Consumers
must check error scalars before using results.

## Graded VTM core

The developing internal `warpower._vtm.VTMWorkspace` accepts `sizing=field`.
Its public full-mesher API remains tracked in #38. The caller refits the field
before each optimization run when sample inputs change. The field stays fixed
through that run; a captured producer can change it between runs.

The graded kernel evaluates sizing at each tetrahedron centroid and weights
its volume, circumcenter moment, and ODT interpolation-error contribution by
`mu**-3`. Surface and crease quadrature use the same field with `mu**-4` and
`mu**-3`, respectively. Reported geometric volume and radius ratio are unweighted.
Corner priority remains unchanged.

This implements the paper's one-point approximation for the interior update
(Eq. 8). The update ignores sizing-field derivatives. The measured energy is a
one-point quadrature surrogate, and device backtracking checks it on rebuilt
connectivity together with the existing boundary-first objectives; this is not
a global convergence guarantee. A unit constant field reproduces the uniform
core. With `sizing=None`, the original uniform kernel is used directly.

Automatic pole-based feature-size estimates and device initial sampling are
remaining parts of #35. This module does not yet provide either operation.

## Validation and measured costs

The [full suite](../results/validation/vtm/sizing-full-tests.txt) passes 211 tests,
including 17 new sizing cases. Independent CPU calculations check the cone
minimum, gradation, scaling, and density-weighted circumcenter contributions.
Tests also cover constant-field reduction to uniform VTM, changing samples,
invalid inputs followed by recovery, and captured producers/consumers.
The [memory check](../results/validation/vtm/sizing-memcheck.txt) passes all 17
cases with zero errors; the [lifetime check](../results/validation/vtm/sizing-lifetime.txt)
passes both captured field and full graded-core cases. The documented
[diagnostic stack and allocator settings](cuda-validation.md) apply.

The [native trace](../results/validation/vtm/sizing-device-audit.json) records
2,145 GPU kernel activities over three graded VTM replays. The graph has no host,
allocation, or free nodes, and all 64 graph copy nodes are device-to-device.
The [installed wheel](../results/validation/vtm/sizing-wheel.json) passes a
captured producer/field/core/consumer test with changing sizing. A
[source audit](../results/validation/vtm/sizing-isolation.json) confirms that
all preexisting runtime modules except the lazy exports and internal VTM wrapper
are byte-identical to `9baabde`; the uniform VTM kernel definitions are unchanged.

CUDA-event medians on the NVIDIA L40, with no observed CPU quota throttling:

| Work | Before | With optional sizing support |
|---|---:|---:|
| Uniform torus, 1,000 sites, 20 iterations | [519.88 ms](../results/validation/vtm/sizing-core-before-torus.json) | [520.98 ms](../results/validation/vtm/sizing-core-after-torus.json) |
| Uniform corner-initialized cube, 200 sites, 12 iterations | [129.07 ms](../results/validation/vtm/sizing-core-before-cube.json) | [128.30 ms](../results/validation/vtm/sizing-core-after-cube.json) |
| Graded torus, 1,000 sites, 20 iterations | — | [529.55 ms](../results/validation/vtm/sizing-graded-torus.json) |

The paired uniform runs use five repetitions, matching iteration and tetrahedron
counts, and remain within 1%. The graded case includes the sizing producer and
refit, completes all 20 iterations without rejected steps, and produces 6,004
full-hull tetrahedra matching SciPy's independent Delaunay result. These timings
exclude domain extraction, automatic sizing, and initial site generation.
They are not a full CPU mesher comparison or a quality claim for the final mesh.

The [standalone field benchmark](../results/validation/vtm/sizing-queries.json)
uses 4,096 random constraints and 100,000 queries. Ten repetitions give 3.03 ms
for queries and 3.15 ms including refit. Setup takes 0.48 s and is reported
separately. A brute-force oracle checks 64 queries with maximum relative error
`5.57e-8`, consistent with float32 output rounding. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.sizing \
  --output results/local/sizing.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.vtm_core \
  --case torus --graded --n 1000 --iterations 20 --reference \
  --output results/local/graded-torus.json
```
