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

## Automatic feature-size estimates

`AutomaticSizingWorkspace(domain, control_subdivisions=1, gradation=1.0, ...)`
constructs a `SizingField` from the finite Voronoi poles of boundary samples.
It includes the original vertices plus `3 * triangle_count * subdivisions**2`
triangle quadrature samples; `control_subdivisions=0` uses vertices alone.
The control mesh is an ordinary, certified full-hull Delaunay triangulation.

For a bounded Voronoi cell, the first pole is its farthest incident tetrahedron
circumcenter. For an unbounded cell, its first pole is at infinity in the
averaged outward unit-normal direction of its incident hull faces. The second
pole is the farthest incident circumcenter in the opposite open hemisphere.
This follows [Amenta and Bern, Figure 1](https://www.cs.jhu.edu/~misha/Fall13b/Papers/Amenta99.pdf).
Infinite poles supply directions but are excluded from distance queries.
Equal-distance choices use canonical vertex-ID ties; hull normals accumulate in
canonical face order, so tetrahedron output-row ordering does not change them.

The estimate at each control point is the distance to the closest **global**
finite pole, found with a float64 bounding hierarchy. Circumcenters, poles, and
raw distances stay float64. Sample values for the resulting field are rounded
to positive float32, bounded to its representable positive range and `1e30`;
field clamps apply after cone extension. `.field` can be used by the sampler,
query service, boundary fitter, and graded VTM independently.

The public diagnostics are `control_points`, `local_feature_size`, `poles`
(two slots per control), `pole_valid`, `pole_source_tets`, `pole_count`, `mesh`,
and `error`. A zero pole-valid flag means that slot is absent; source tet IDs
refer to the current control mesh. Duplicate controls need not own cells but
still receive global nearest-pole distances. The constructor builds the initial
field. Call `domain.refit()` and then `auto.refit()` after changing the boundary.
All subsequent refits are capturable and keep their storage fixed.

This is a sampled pole approximation to local feature size, not an exact medial
axis or a guaranteed feature detector. Enrichment matters: a plate with
half-thickness 0.25 has a median estimate about 1.44 from its eight corners,
versus about 0.31 with 44 controls. Dense controls increase reconstruction cost.
The currently conservative symbolic Delaunay fallback can be especially slow on
structured boundary samples; [#62](https://github.com/alecjacobson/warpower/issues/62)
tracks its acceleration. Finite construction/capacity failures remain explicit.

| Automatic error bit | Meaning |
|---:|---|
| 1 | Invalid domain |
| 2 | Control Delaunay construction failed |
| 4 | Invalid circumcenter or pole geometry, or hull-star capacity exhausted |
| 8 | No finite poles, or invalid nearest-pole distance |

Failure clears the published pole mask/count and feature sizes, and invalidates
`.field`; consumers must check the parent error. Child mesh diagnostics can
refer to an earlier replay when an invalid domain prevents reconstruction.
A subsequent valid refit recovers normally.

## Device initial sampling

`InitialSamplingWorkspace(domain, n, sizing=None, resolution=32, seed=2026,
max_attempts=4096)` allocates fixed storage for exactly `n` initial sites. Its
capturable `run()` samples material uniformly, or with density proportional to
`mu(x)**-3` when supplied a field. `.points` is float32; `.count`, `.error`,
`.attempts`, and `.seed` are device arrays. The constructor allocates and records
work; call `run()` to generate sites. Callers refit changed domain and field
inputs first. The seed can be a caller-owned contiguous device `int32[1]`.
Unchanged inputs and seed reproduce the same sites. A captured producer can
advance the seed for a fresh cloud.

```python
from warpower import AutomaticSizingWorkspace, InitialSamplingWorkspace

# domain is a previously allocated SurfaceDomain.
auto = AutomaticSizingWorkspace(domain, control_subdivisions=1, gradation=0.5)
sampler = InitialSamplingWorkspace(domain, 10000, sizing=auto.field)
with wp.ScopedCapture() as capture:
    # A producer may modify domain.points here.
    domain.refit()
    auto.refit()
    sampler.run()
    # A consumer reads sampler.points[:sampler.count[0]] on device.
wp.capture_launch(capture.graph)
```

The sampler uses bounded rejection with a regular-grid proposal. Boundary cells
are retained conservatively using triangle AABBs, even when their centers lie
outside the material. This avoids dropping thin features or small components
because of grid resolution. Remaining cells use a material-side query. Each
retained cell gets a conservative density upper bound from the field's global
minimum and Lipschitz bound. A float64 cumulative distribution chooses a cell;
a prefix maximum repairs roundoff on zero-mass plateaus without accepting lost
positive increments. Uniform coordinates are drawn, rounded to float32, and checked for material
membership and sizing-density acceptance. Each site owns a separate block of `2**32` successive SplitMix64 states,
using the [public-domain reference mixer](https://prng.di.unimi.it/splitmix.c).
The bounded sampler consumes fewer than `2**23` states per site, preventing
stream overlap within a cloud even when rejection counts differ. Each state
supplies a 53-bit uniform variate.

This replaces the paper's two-pass error diffusion with parallel rejection;
it targets the same density, with finite floating-point and random-number
resolution. It does not guarantee that every small component receives a site
at a finite budget. Increase the budget or provide component-specific sites when
that guarantee is required. Grid resolution is in `[1,128]`; larger grids tighten
proposal bounds at greater preprocessing/storage cost. Very thin material or
strong grading can need more attempts. No unbounded retry or host decision is
hidden in replay.

| Sampling error bit | Meaning |
|---:|---|
| 1 | Invalid domain |
| 2 | Invalid sizing field |
| 4 | Invalid/empty proposal or unrepresentable positive CDF increment |
| 8 | A site exhausted its attempt budget |
| 16 | Duplicate output float32 sites |
| 32 | A sizing lower-bound check failed |

Successful runs publish `count == n` with unique sites. Any error publishes
`count == 0`; point storage outside that active count must not be consumed.
It can contain partial or previous results. Errors clear on the next replay.
The attempt budget is an integer in `[1,1000000]`. Extremely coarse float32
coordinate resolution can prevent unique sampling; translating/scaling the
input or reducing its site budget is then necessary.

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


## Automatic sizing and initialization measurements

The initialization benchmark separates control-mesh/pole refits from site
sampling. All three cases below generate 100,000 unique interior sites, with
zero device errors and no observed CPU quota throttling on the NVIDIA L40.
The combined graph includes a changing-boundary producer, domain refit,
automatic sizing, sampling, and a device consumer. Setup is reported separately
in each record; the field and output buffers remain resident.

| Boundary | Controls | Automatic refit | Sampling | Combined changing-input graph |
|---|---:|---:|---:|---:|
| [Thin plate](../results/validation/vtm/initialization-thin-plate.json) | 44 | 39.07 ms | 2.62 ms | 51.25 ms |
| [Sphere](../results/validation/vtm/initialization-sphere.json) | 64 | 3.45 ms | 14.74 ms | 19.75 ms |
| [Torus](../results/validation/vtm/initialization-torus.json) | 128 | 2423.29 ms | 16.58 ms | 2572.14 ms |

Thin plate and sphere use five timed replays; torus uses three. The combined
column is not the sum of the preceding medians: it also changes the boundary,
which changes numerical reconstruction work. These are initialization costs,
not final tetrahedral-meshing costs or CPU speedup evidence. The structured
torus control triangulation remains a performance limitation tracked in #62.

Reproduce each case with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.initialization \
  --case thin-plate --n 100000 --repeats 5 \
  --output results/local/initialization-thin-plate.json
# Use --case sphere, or --case torus --repeats 3, for the other cases.
```

The [source audit](../results/validation/vtm/initialization-isolation.json)
confirms that all 30 preexisting runtime modules except lazy exports match
`471f5cd` byte-for-byte. Default imports do not load the new optional modules.
Standalone diagram, Delaunay, regular-mesh, and uniform-optimizer computation
therefore uses the same implementation validated in the preceding milestone.
