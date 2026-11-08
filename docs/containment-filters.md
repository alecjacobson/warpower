# Exact containment filters

Material-boundary recovery can add an excluded tetrahedron when its original
binary32 geometry proves it lies in closed material. Every overlapping input
triangle plane must support the entire tetrahedron on the material side. A
vertex strictly off all overlapping planes supplies a winding witness. This
contract and the existing selection/topology transaction are unchanged.

The plane predicates use the existing certified `orient3d` filter. For a ray
from a strict witness `p` along `d`, set `a=q0-p`, `b=q1-p`, `c=q2-p` as exact
expressions on the input coordinates. Five values determine each triangle event:

- `n=det(a,b,c)`;
- `u=det(d,b,c)`, `v=det(d,c,a)`, `w=det(d,a,b)`;
- `den=u+v+w`.

The signs determine a forward crossing, its barycentric inclusion, and its
orientation. An exact edge hit or coplanar ray retains the original ambiguous
result and retry policy. Certified binary64 `Ball` operations enclose all five
values. Only when all five signs are certified does the filter decide the
event. Otherwise the original bounded dyadic-integer arithmetic computes the
same expressions. No tolerance turns an uncertain value into a zero.

Finite binary32 inputs convert exactly to binary64. Differences have magnitude
below `2^129`; degree-three determinant values and error bounds remain below
`2^400`. Nonzero input values and differences lie on the `2^-149` lattice, so
nonzero value products are at least `2^-447`. Required error products are at
least `2^-603`, above binary64 underflow. The existing padded interval operations
cover rounding of the error calculations. Calling kernels disable fast math and
floating-point contraction. Nonfinite values cannot bypass the integer fallback.

This adds no device buffers, host decisions, or graph nodes. Independent
power-diagram and Delaunay paths do not call these recovery predicates. Validation
and performance measurements for this change are tracked in
[issue #209](https://github.com/alecjacobson/warpower/issues/209).

## Measured complete solves

The NVIDIA L40 measurements compare runtime `795af2f` with `96027fb`, using
Warp 1.19.0. Each gallery result is the median of three additional fresh complete
resident solves, including domain refit, sampling, growth, fitting, quality
stages and extraction. Setup, upload, export and independent validation are
outside replay. All first and timed windows have zero observed CPU quota
throttling. The unchanged settings use 1,000 sites and 256 initial sites, except
the torus and graded Armadillo use 4,096 sites. Quality has four rounds;
coarse cavity/disconnected use 4/4 quadrature, and hand uses 20/20.

| Complete solve | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| Hand seed 2 | 2896.201 | 2109.867 | 1.37× |
| Hand seed 7 | 1897.661 | 1443.620 | 1.31× |
| Hand seed 7, four local retries | 1746.249 | 1378.670 | 1.27× |
| Coarse cavity | 1937.080 | 543.386 | 3.56× |
| Coarse disconnected | 1940.340 | 748.069 | 2.59× |

All 12 successful final pairs retain roles, recorded rounds, steps, moves and
rejection counts, canonical full/material connectivity and oriented boundary
triangles. Small near-zero coordinate
variations do not change these final cell sets. The other healthy solves—hand
seeds 0, 1, 42 and 2026, plate, torus and graded Armadillo—change by −1.04% to
0.00% in time. These small changes do not establish a filter benefit on paths
that perform no containment work. Coarse hand remains explicitly invalid in
both versions, with error 64 and zero public counts; its timing is excluded
from valid-mesh speed comparisons.

| Standalone routine | Before ms | After ms | Change |
|---|---:|---:|---:|
| Delaunay, 1,000 sites | 9.1940 | 9.1829 | -0.12% |
| Delaunay, 100,000 sites | 182.2579 | 183.2659 | +0.55% |
| Delaunay, 4³ grid | 2.2026 | 2.2108 | +0.37% |
| Delaunay, fitted hand | 13.3827 | 13.3799 | -0.02% |
| Diagram, 100,000 unweighted sites | 18.1608 | 18.1204 | -0.22% |

Standalone Delaunay uses nine repeats; the diagram uses 30 after five warmups.
Inputs, counts, status and diagnostics agree in all five pairs. The largest
observed slowdown is 0.56%, with no material regression in this grid. These
independent runtime paths are unchanged.

## Validation and retained variation

All 24 valid final meshes pass independent full-hull, exact empty-sphere and
orientation checks; material incidence and vertex-link checks; physical
quality and two-sided sampled distance measurements; exact VTU round trips;
and boundary self-intersection checks. Hand and plate also include sampled
feature-chain measurements. These are not Hausdorff or exact constrained-edge
conformity certificates.

The complete suite passes 1,100 tests. New primitive tests check all five
intervals and ray events against 1,384 arbitrary-precision rational cases,
including subnormal and extreme inputs, degenerate triangles and exact boundary
hits. Fifteen nonfinite cases retain an inconclusive result. Complete-trajectory
instrumentation computes old and new containment decisions side by side on
1,636 eligible cells across four recovered solves. All decisions agree,
including all 504 positive cavity decisions. Instrumented times are excluded
from performance comparisons.

An earlier clean pilot has four different exterior hull cells in hand seed 7
after coordinate differences of about `3.5e-18`. Both full hulls are independently
valid, and material cells, boundary, work, per-cell qualities and fit agree.
The strict checker failure, exact changed-cell volumes and all snapshots are
retained. The final grid was declared and run separately; its 12 valid pairs
match canonical connectivity. Neither comparison establishes bitwise
reproducibility or identifies the causal kernel; see
[issue #208](https://github.com/alecjacobson/warpower/issues/208).

The first primitive test harness used unsupported indexed tuple stores and was
corrected. An initial timing harness reused a candidate directory because an
export-loop variable shadowed the case name. It was stopped; its first candidate
geometry was overwritten and cannot serve as evidence. Remaining files and logs
are retained with this limitation. Fresh pilot and final runs use unique paths
and overwrite guards; their outputs and timings are independently checked.

The wide comparable-quality native CPU speedup goal remains unmet. These are
GPU-to-GPU improvements to exact recovery. The README retains its Paragram
flagship result.

Eight isolated Compute Sanitizer jobs pass with zero errors. The 23 primitive
and basic recovery tests, the device-selected quality guard, a complete recovered
hand solve with four local retries, and the four-phase captured failure/recovery
loop each pass padded and pool-disabled allocation-lifetime checking. The
instrumentation uses the existing 131,072-byte diagnostic stack allowance;
ordinary timings and native traces use the normal runtime configuration.
No tests are excluded and no job reaches its 1,200-second limit.

Native first-launch traces of the complete solve and the four-phase loop show
only device-to-device graph copies, with no host callback or allocation/free
nodes and no allocation/free/memcpy API calls in the execution range. Both
record graph instantiation/upload and one external final synchronization.
The loop reports errors `[0,1,0,0]`, validity `[1,0,1,1]` and recovery enable
flags `[1,0,0,1]`; invalid input produces zero counts and the final frame
recovers. Nsight can omit individual graph-copy activities, so every DOT copy
node is independently checked. The new full-hand profile attributes 7.04 ms
to 88 containment calls; profiling totals are separate from clean timings.

The [evidence manifest](../results/validation/vtm/containment-filters/manifest.json)
links sources, inputs, failed attempts, snapshots, timing and validation records,
memory logs and native traces. The finite-oracle array contains 1,384 cases
(512 ordinary, 512 random finite-bit and 360 structured cases); its retained
array and hashes correct an earlier progress-note arithmetic error that reported
1,744. The tested data and passing results did not change.
