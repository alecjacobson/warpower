# Conservative geometry in symbolic construction

The symbolic Delaunay fallback keeps its topology as triples of incident plane
IDs and uses the original indexed sites and symbolic lifting rule. Nearby BVH
nodes are visited first, using squared point-to-box distance solely to choose
visit order. The additional geometric filters described here accept only strict
signs; uncertain cases and exact ties retain the existing filtered/exact
predicates. Every resulting mesh still passes the full original-coordinate
Delaunay certificate.

## Cached vertex intervals

A vertex is the intersection of three site or clipping-box planes. The existing
halfspace code supplies float64 balls enclosing the exact, unweighted plane
coefficients. Cramer's rule and the [outward quotient bound](sphere-bounds.md)
enclose the Cartesian intersection. Negative determinant orientations are
normalized by exact negation. If the determinant interval contains zero, a
quotient is nonfinite, or a coordinate cannot be bounded, that coordinate keeps
the complete initial clipping-box interval. A valid clipped vertex's limit lies
inside that box, so interval intersection with the box is safe.

Each coordinate interval `[lo, hi]` is cached as a midpoint and an error radius.
The radius includes the half-width, `16*eps64*(abs(lo)+abs(hi))`, positive-arithmetic
padding, and `16*min_normal64`. These allowances enclose midpoint/radius rounding
and underflow; they are arithmetic error bounds, not geometric tolerances.
Initial box corners are exactly representable promoted float32 values, with zero
error. Subsequent vertices compute their intervals only when they are created.

The cache uses the existing `Scratch64.positions` and `cartesian` arrays as
values and errors during symbolic construction. Every vertex swap moves both
arrays. Plane-table compaction changes IDs without moving the corresponding
geometric vertex. Each constructor initializes its own active scratch before
use, so sharing these arrays with the ordinary floating-point constructor does
not retain stale geometry or add allocations.

For each candidate plane, its unweighted coefficients are computed once. A
cached vertex interval evaluates the linear halfspace expression with Ball
arithmetic. A finite, strictly nonzero result determines inside/outside without
recomputing an in-sphere determinant. An absolute normal-number allowance covers
underflow in these interval products and sums. Otherwise the original site or
site/box symbolic predicate resolves the classification. In particular, a
numerically small value never becomes a tie, and a zero sign never bypasses
symbolic tie-breaking.

## Why unweighted limits suffice

The symbolic weights perturb halfspace constants, not normals. A valid vertex
with nonzero normal determinant is therefore a continuous function of those
infinitesimals and tends to the unweighted Cramer intersection. A strictly
positive or negative halfspace value at that limit has the same sign for all
sufficiently small positive infinitesimals. Only a zero limiting sign requires
higher symbolic coefficients. There are finitely many vertices and predicates
in a bounded construction attempt, so all strict decisions admit a common
sufficiently small perturbation. No finite perturbation is applied to output
positions.

## Directional cell and node rejection

The union of the cached vertex intervals encloses the limiting convex cell.
The cell bounds are updated after each plane that actually removes vertices;
an unchanged vertex count alone would not detect a changed polyhedron.
Let `x` range over the cell box relative to its owner, and let `delta` range over
a BVH node's site positions relative to the same owner. A site cannot clip the
cell if the following strict inequality holds throughout those intervals:

```
2*x dot delta - |delta|^2 < 0.
```

For one coordinate, with `x in [l,h]` and `delta in [a,b]`, the maximum is the
larger of the two values obtained by taking `x=l` and `x=h` and setting
`delta=clamp(x,a,b)`. This follows because the expression is linear in `x`, and
for fixed `x` it is a concave quadratic in `delta`. Summing three coordinate
maxima bounds the complete expression. The implementation encloses input
subtractions, endpoint arithmetic, quadratic evaluation, and the final sum;
only a finite, strictly negative upper bound rejects a node or individual site.
Large offsets, poorly conditioned vertices and nonfinite arithmetic can weaken
or disable rejection, but cannot silently choose a predicate tie.

All relevant kernels disable fast math and fused multiply-add. Interval code
explicitly includes absolute underflow allowances rather than assuming that
rational Cartesian centers obey the normal-product assumptions of the original
float32 polynomial filters.

## Validation

`tests/test_symbolic_bounds.py` uses exact rational arithmetic to check mixed
site/box Cramer intersections across scales and translations, nearly flat
configurations, the directional box maximum, touching/one-ULP cases, and cached
plane signs. Exact repeated-site and cospherical ties must defer to the original
predicate. Independent symbolic-construction tests also cover complete geometry,
changing-input capture, capacity failures, clipping-box growth, invalid input
and recovery. The [369-test suite](../results/validation/vtm/symbolic-bounds-full-tests.txt)
passes. Two [additional integration tests](../results/validation/vtm/symbolic-bounds-extra-tests.txt)
then pass on the same numerical source: 1,000 forced-symbolic random sites match
SciPy, and small scratch chunks correctly reset cached geometry across duplicate
sites and changing-input replays.

Seven [padded pooled-memory cases](../results/validation/vtm/symbolic-bounds-memcheck.txt)
and one [synchronous-allocation lifetime case](../results/validation/vtm/symbolic-bounds-lifetime.txt)
pass with zero Compute Sanitizer errors. Both use a diagnostic 32 KiB stack;
production allocation and stack settings are unchanged. The
[native changing-input trace](../results/validation/vtm/symbolic-bounds-device-audit.json)
records 669 GPU kernel activities across three producer/domain/automatic-sizing/
sampling/consumer replays. Its graph has 504 kernel, 99 memset, 59 conditional
and 28 device-copy nodes, with no host, allocation or free nodes. All observed
copies and graph copy nodes are device-to-device; one final synchronization is
outside geometry. [Isolation hashes](../results/validation/vtm/symbolic-bounds-isolation.json)
record unchanged ordinary builders and final certificate.

## Measured scope

CUDA-event medians on an NVIDIA L40, Warp 1.19.0, with setup/JIT/upload excluded
and no observed CPU-quota throttling:

| Workload | Before | Cached intervals | Speedup |
|---|---:|---:|---:|
| Forced symbolic, 1,000 random sites | 1,529.74 ms | 124.90 ms | 12.25× |
| Symbolic fallback, 128 structured torus sites | 2,415.97 ms | 1,221.20 ms | 1.98× |
| Automatic sizing, 224 controls, static refit | 771.23 ms | 440.08 ms | 1.75× |
| Automatic sizing, 224 controls, changing domain | 798.03 ms | 497.51 ms | 1.60× |
| Automatic sizing, 896 controls, static refit | 3,364.40 ms | 538.67 ms | 6.25× |
| Automatic sizing, 896 controls, changing domain | 3,423.13 ms | 648.65 ms | 5.28× |
| Complete default 1,000-site torus mesher, including domain refit | 5,768.70 ms | 5,726.57 ms | 1.01× |

The [random baseline](../results/validation/vtm/symbolic-before-random1000.json),
[optimized random case](../results/validation/vtm/symbolic-cached-random1000.json),
[torus baseline](../results/validation/vtm/sphere-box-performance.json), and
[optimized torus case](../results/validation/vtm/symbolic-cached-torus128.json)
record stage timing and strategy. First replay can be slower while the clipping
box grows; subsequent replays reuse its size, rebuilding and certifying current
geometry. Forced-symbolic timings do not describe ordinary-constructor throughput.
The [10,000-site symbolic case](../results/validation/vtm/symbolic-cached-random10000.json)
still takes 2,159.67 ms; there is no before/after claim at that size.

The enriched control clouds add three quadrature sites per input triangle to
8×4 and 16×8 torus vertices. Records for
[224 before](../results/validation/vtm/symbolic-controls-before-224.json),
[224 after](../results/validation/vtm/symbolic-controls-after-224.json),
[896 before](../results/validation/vtm/symbolic-controls-before-896.json), and
[896 after](../results/validation/vtm/symbolic-controls-after-896.json)
include independent hull checks. The 224-site case additionally verifies the
39 cells differing from SciPy using rational predicates.
[Equivalence checks](../results/validation/vtm/symbolic-bounds-equivalence.json)
confirm bit-identical control positions, canonical tetrahedra and local feature
sizes at both sizes. The 896-site case does not claim an all-cell rational audit.

The complete mesher's [before](../results/validation/vtm/symbolic-before-full-torus1000.json)
and [after](../results/validation/vtm/symbolic-cached-full-torus1000.json) results
have identical positions, canonical material cells, counts and quality. Its
ordinary reconstruction path receives no meaningful overall acceleration here.
The goal of outperforming native CPU meshing at comparable quality remains open.

Reproduce the enriched comparison with `python -m benchmarks.symbolic_controls
--major 16 --minor 8 --subdivisions 1 --repeats 3 --output result.json` on each
revision. See `--help` for the smaller case and optional rational reference check.
