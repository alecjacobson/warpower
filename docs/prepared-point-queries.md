# Prepared point bounds for VTM seed queries

VTM's private previous-cell filter now evaluates a point-specific sphere
enclosure and uses either certified strict physical sign. This removes general
box midpoint/radius arithmetic and avoids a second sphere predicate for strictly
inside queries. Uncertain signs and physical ties retain the original complete
symbolic predicate. The change adds no storage or launches.

| Query stage | Previous prepared checker | Point-specific checker |
| --- | --- | --- |
| Geometry preparation | Current sphere cofactors and box, once per candidate | Same preparation and full precision |
| Cheap exclusion | Conservative sphere box against the query point | Same box test |
| Polynomial enclosure | General `interval(p, p)` in each coordinate | Exact float32-to-float64 promotion, then bounded differences |
| Strict negative sign | Skip the outside query | Same decision |
| Strict positive sign | Evaluate the original sphere predicate again | Reject the candidate directly |
| Uncertain or tied sign | Original physical, hash-weight and site-ID predicates | Same complete fallback |
| Queries after rejection | Continue scanning | Continue scanning |
| Workspace and launches | 108 bytes per candidate; prepare and check kernels | Unchanged |

Only `PreparedSeedQueries` in the VTM reuse path changes. Ordinary diagram and
Delaunay constructors retain their code and allocation paths. Every invocation
refreshes the prepared geometry; no proof survives a changed input without the
existing current-input checks. Canonical ownership, positive orientation, seed
compaction, full hull certification and ordinary-constructor fallback remain.

## Enclosure and sign policy

For positive defining tetrahedron `(a,b,c,d)`, preparation encloses its orientation
determinant `D` and sphere cofactors `(nx,ny,nz)` with the existing Ball64 arithmetic.
With `x = p-a`, the query polynomial is

```text
x.x*nx + x.y*ny + x.z*nz - D*(x.x*x.x + x.y*x.y + x.z*x.z).
```

This is the original-coordinate unweighted in-sphere determinant in Warpower's
convention. A positive sign means inside the sphere for a positive defining cell.
An individual float32 point promotes exactly to float64, so it needs no generic
midpoint uncertainty. Each subtraction still has its full rounding bound, and
all cofactor uncertainty and subsequent Ball64 arithmetic remain. Calling kernels
disable fast math and fused arithmetic, as required by those existing bounds.

The checker requires finite value and error and `abs(value) > error`. Equality,
zero enclosures, uncertainty and nonfinite results fall through to the original
`symbolic_sphere`; none becomes a tolerance-based tie. Exact physical ties still
use hash weights followed by ascending-site-ID infinitesimals.

The finite-input degree-five range argument is unchanged: float32 coordinates
are bounded by 2^128 and lie on the 2^-149 lattice; their promoted differences
and the geometric products fit the existing normal binary64 range. Full-enclosure
tests use independent rational Bareiss determinants, including finite float32 bit
patterns, subnormals and extremes, translations, thin cells, ties and one-ulp
changes. They check the entire value/error interval, not only signs that pass.
Additional nonfinite cases must fail the strict gate. These tests support the
arithmetic proof and its compiler/input assumptions.

## Complete solves and validation

Runtime `6a82244` is compared with `6469ab6` on NVIDIA L40 / Warp 1.19.0.
Each complete resident replay includes current-domain refit, fresh sampling,
growth, fitting, quality stages, extraction and diagnostics. Setup, compilation,
transfer and offline validation are excluded. Torus1k uses five measured replays;
the other cases use three. All initial and measured CPU quota windows are clean.

| Complete solve | Before (ms) | After (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 737.038 | 716.928 | 1.028× |
| Torus, 4,096 sites | 2,824.362 | 2,639.847 | 1.070× |
| Graded torus, 4,096 sites | 3,164.903 | 2,975.950 | 1.063× |
| Thin hand, 1,000 sites | 2,334.691 | 2,322.354 | 1.005× |
| Cavity, 1,000 sites | 916.279 | 905.709 | 1.012× |
| Disconnected, 1,000 sites | 638.196 | 631.352 | 1.011× |
| Armadillo uniform, 4,096 sites | 3,737.830 | 3,607.867 | 1.036× |
| Armadillo graded, 4,096 sites | 7,292.357 | 7,175.277 | 1.016× |

The largest complete-solve time reduction is 6.5%; the hand's 0.5% difference is
small. All eight pairs preserve input arrays, roles, full/material tetrahedron
sets, oriented boundaries, parameters and iteration work. Five preserve
coordinates bitwise. The hand changes three coordinates by at most `6.10e-19`;
the cavity and disconnected meshes each change one coordinate by one float32 ulp
(`2.33e-10` and `4.66e-10`, respectively). Every changed value is recorded without
relaxing the connectivity checks or claiming bitwise identity for these cases.

All eight final full hulls pass independent CGAL 6.2.1 exact orientation, global
empty-sphere, face-pairing and hull-support checks. NumPy/AABB material-quality
checks and single-cycle boundary vertex links pass. All six gallery VTUs
round-trip exactly and have zero CGAL boundary self-intersection pairs.
The [final comparison](../results/validation/vtm/prepared-point-queries/prepared-query-final-independent.json)
includes all coordinates, work, quality, setup and first-replay comparisons.

The persistent workspace pools remain 273,862,732 bytes at 1k and 1,743,411,108
bytes at 4k. Warm-cache setup is 1.516→1.551 s and 2.924→2.927 s; no setup gain
is claimed. First replays are 961.406→942.224 ms and 3,581.980→3,432.491 ms.
These pool measurements are not total process-memory peaks, and these setup
measurements do not include cold compilation.

Standalone comparisons use nine Delaunay and thirty diagram replays:

| Independently callable operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Certified Delaunay, 1,000 sites | 9.193856 | 9.158944 |
| Certified Delaunay, 100,000 sites | 182.851196 | 182.839066 |
| Certified Delaunay, grid ties | 2.184640 | 2.186880 |
| Certified Delaunay, fitted-hand cloud | 13.418912 | 13.403200 |
| Diagram, 100,000 sites | 18.093796 | 18.115757 |

All differences are below 0.4% and are treated as timing variation. Inputs,
output counts and diagnostics match. An initial diagram pair was
18.762877→18.130306 ms, with larger sample variability despite clean CPU quotas.
The table uses a fresh pair with identical parameters; both pairs and every raw
sample are retained. No performance improvement is attributed to the unchanged
standalone code.

The integrated focused suite passes **111 tests**; the complete suite passes
**736 tests** in 292.55 seconds. The 42 new tests include complete rational error
enclosures and captured producer/filter/consumer comparisons against the previous
checker. Six device-produced frames change geometry, canonical ownership and
candidate counts, including empty and restored prefixes, with poisoned prior
live flags. Cube, octahedron, thin, random and fitted 4k clouds cover scales
2^-60 through 2^60 at both 32- and 256-thread blocks. Small cases additionally
match independent rational physical and symbolic masks.

All 42 new tests pass unfiltered 32-byte-padded Compute Sanitizer memcheck and
API checking with full stream-ordered allocation/lifetime tracking, with zero
errors. The changing-input 2,050-slot routed integration test also passes padded
memcheck and API/use-after-free tracking. These diagnostic runs use a 128 KiB
CUDA stack; lifetime runs disable the memory pool. Runtime defaults do not change.
The existing [conditional native-sort allocation-order limitation](cuda-validation.md#conditional-native-radix-sort-scratch)
is retained: this work does not claim that the routed graph passes that broader
mode, and does not relabel the prior failures as successes.

The native full-solve trace records 60,862 kernel activities. Its graph contains
33,222 kernel nodes and 2,932 device-to-device copy nodes, matching the preceding
implementation's structure. Prepared seed checking executes 208 times and costs
554.292 ms versus 736.293 ms in the preceding trace; total kernel activity is
2,839.922 ms versus 3,031.179 ms. These diagnostic durations explain the clean
complete-solve measurements; they are not additional end-to-end timing claims.

A separate graph executes eight device-controlled producer/routed-constructor/
consumer phases, including changed inputs, a smaller prefix, invalid count,
nonfinite geometry and recovery. All six valid outputs match independent SciPy
meshes; both invalid phases publish empty output. It records 1,345 kernel
activities, with 2,389 kernel and 228 device-to-device copy nodes in the graph.
This trace executes the changed seed checker three times.

Both audits observe one graph launch and one final synchronization. All observed
copies and all DOT copy nodes are device-to-device, with no host callbacks or
allocation/free nodes. First-use graph instantiation/upload APIs are present;
setup, final readback and offline validation are outside the geometric pipeline.
The [full-solve audit](../results/validation/vtm/prepared-point-queries/prepared-query-full-native-audit.json)
and [outer-loop audit](../results/validation/vtm/prepared-point-queries/prepared-query-loop-native-audit.json)
retain the copy-trace limitations and exact counts.

The [manifest](../results/validation/vtm/prepared-point-queries/manifest.json)
records source revisions/hashes, private alternatives, commands, raw samples,
independent checks and validation artifacts. Text meshes and native traces are
losslessly compressed; decompression restores their recorded original hashes.
Prototype scripts target their recorded source revision and require the existing
offline reference binaries for independent checking.

```bash
python -m benchmarks.mesher --n 4096 --repeats 3 --save-mesh torus.npz --output torus.json
python -m pytest -q tests/test_prepared_point_bounds.py tests/test_prepared_query_masks.py
```

## Alternatives

Private pilots also evaluated outside-only point bounds, an analytic physical
sphere filter before the previous prepared enclosure, and stopping a candidate's
query lanes after a certified rejection. Each used fresh complete-solve controls.
The two-sign prepared expression gave the best measured arithmetic improvement.
Stopping added less than 1% beyond it in those torus pilots and is not adopted.
The integrated checker preserves exhaustive querying and the original diagnostic
policy. The private variants, raw samples and independent checks remain in the
evidence; they are not additional default implementations.

This work is tracked in [#170](https://github.com/alecjacobson/warpower/issues/170)
and [PR #171](https://github.com/alecjacobson/warpower/pull/171). The
[native CPU performance goal](cpu-mesher-baseline.md) remains unmet, and the
README retains the independently earned Paragram result.
