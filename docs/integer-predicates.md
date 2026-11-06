# Bounded integer predicate fallbacks

This change replaces expansion evaluation in the optional certified constructor
with exact signed integer arithmetic for stored finite float32 inputs. The
existing float64 interval filters, bounded-integer float64 shortcut and symbolic
hash/ID tie rule remain unchanged. Only uncertain cases reach this fallback.
No site is rounded, snapped or displaced, and no native extension or host
predicate is called. The original diagram and `DiagramWorkspace` construction
paths do not import these predicates.

## Representation and arithmetic

A finite float32 value is a signed integer mantissa times a power of two. Exact
power-of-two normalization in float64 recovers that representation, including
float32 subnormals. A common exponent `e` is the minimum of zero, coordinate
exponents and half the weight exponents rounded down. Mixed box rows also
include half their constant exponent. Coordinates are integers in units `2^e`;
weights and lifted constants are integers in units `2^(2e)`.

`_integer.Big` stores a sign, an active word count, validity and 48 little-endian
uint32 words. Addition/subtraction propagate carries/borrows; multiplication
uses uint64 products and sums. For radix `B=2^32`, a schoolbook multiply step is
at most `(B-1)^2 + (B-1) + (B-1) = B^2-1`, which fits uint64. Trimming gives a
unique zero sign and minimal word count. Operations propagate invalid operands
and explicitly reject any carry or product beyond storage.

Translated coordinate differences and lifted rows are formed with these exact
operations. Orientation uses a factored 3×3 determinant; power and mixed
site/box predicates use factored 4×4 determinants. The leading three words,
scaled back by the appropriate power of two, give the approximate float64
`estimate`. The sign comes from the full integer, never from that estimate.

## Capacity argument

Every finite float32 coordinate or weight has magnitude less than `2^128` and
is a multiple of `2^-149`. Thus `-149 <= e <= 0`. The following are conservative
strict magnitude bounds, including all intermediate determinant sums:

- A scaled coordinate is less than `2^277`; a translated difference is less
  than `2^278`. The orientation determinant is less than
  `6 * 2^(3*278) < 2^837`.
- A translated power lift is less than `2^261` in physical units, including
  the weight difference, and less than `2^559` in scaled integer units. Its
  determinant is less than `24 * 2^(3*278+559) < 2^1398`.
- A doubled site-bisector normal is less than `2^279` in integer units. The
  constant `|q|²-|p|²+w_p-w_q` is less than `2^259` in physical units and less
  than `2^557` in integer units. Box normals/constants and the constant-one
  normal-determinant row fit the same bounds. The mixed determinant is less
  than `24 * 2^(3*279+557) < 2^1399`.

Factored subdeterminants and their partial sums obey these term-sum bounds too.
All required results therefore fit well within 1,536 bits. Overflow checks still
fail closed rather than relying on callers to interpret a truncated result.
This proof concerns finite float32 inputs and the stated row constructions;
it is not an arbitrary-precision or float64-input API.

The smallest nonzero determinant unit is at least `2^-745`. The largest physical
mixed determinant is below `2^654`. Both are normal finite float64 values, as
are the powers used to recover the estimate. Calling kernels retain
`fast_math=False` and `fuse_fp=False` for the interval-filter arithmetic contract.

## Validation and measured effect

The integrated implementation passes **410 tests**, including 16 targeted
predicate/arithmetic tests. Independent rational/integer oracles cover raw
float32 bit patterns across the exponent range, subnormals, exact ties, weighted
queries, one-ULP changes, mixed box/site planes, invalid inputs, long
carries/borrows, explicit capacity overflow and all internal powers from −800
through 800. All 16 targeted tests pass padded CUDA memcheck with zero errors.
The direct exact-predicate producer/consumer graph also passes full allocation
and lifetime tracking, using the documented diagnostic 32 KiB stack and
synchronous allocations. These diagnostic settings are not runtime changes.

A native trace of four warmed Delaunay producer/consumer replays exercises local
certification, global fallback, invalid input and recovery. It records 514 GPU
kernels, four graph launches and one final synchronization; all 21 graph copy
nodes are device-to-device, with no host or allocation/free nodes. Setup, graph
instantiation, warmup and final readback are outside the trace. The direct exact
predicate graph is separately checked by the test and lifetime run; the native
Delaunay trace is not a count of integer-fallback invocations.

All 671,908 oriented output cells on the 100,000-site random fixture agree in
canonical connectivity with the saved native CGAL 6.2.1 reference and preceding
constructor. A forced global certificate also passes. Full torus VTM points,
roles, input arrays and oriented full/material/boundary connectivity are
unchanged. The 384-site hand result also matches the preceding constructor.

The complete hand hull passes an independent global certificate using the
preceding expansion arithmetic. Against native CGAL it has 33 different cells
on each side: Python-integer queries against every original site establish
empty open spheres and one or two additional exactly cospherical sites for
every differing cell. These are alternate tie choices, not an assertion of
identical CGAL connectivity on this fixture.

### Performance

NVIDIA L40, Warp 1.19.0; resident-device operation times exclude setup/JIT and
readback, but include every attempted recovery strategy. Five samples per
standalone case, three complete VTM replays, two cached refined-hand replays.
The first refined-hand run is reported separately because it discovers the
successful strategy and box extent.

| Complete operation | Expansion baseline | Integer fallback | Ratio |
|---|---:|---:|---:|
| Random 1,000-site Delaunay | 9.157 ms | 9.186 ms | 1.00× |
| Random 10,000-site Delaunay | 40.696 ms | 40.676 ms | 1.00× |
| Random 100,000-site Delaunay, initial five samples | 181.190 ms | 184.013 ms | 0.98× |
| Random 100,000-site Delaunay, ABBA follow-up | 182.939 ms | 182.041 ms | 1.00× |
| Cube | 0.525 ms | 0.526 ms | 1.00× |
| Grid | 2.193 ms | 2.197 ms | 1.00× |
| Corner cloud | 8.066 ms | 6.820 ms | 1.18× |
| Torus boundary, 128 sites | 1,143.839 ms | 228.214 ms | 5.01× |
| Fitted hand, 256 sites | 15.893 ms | 13.447 ms | 1.18× |
| Refined hand, 384 sites, first run | 195,746.500 ms | 38,786.0 ms | 5.05× |
| Refined hand, 384 sites, cached repeats | 35,624.156 ms | 6,494.056 ms | 5.49× |
| Full torus VTM, 1,000 sites | 3,199.471 ms | 3,207.572 ms | 1.00× |

The initial 1.56% random-100k difference did not persist in alternating version
batches: 30 samples per version give medians shown above and means 182.793 /
182.427 ms. Small cases and the full torus remain within about 0.3% in this
matrix; this does not establish literally zero overhead on every input. All
full-operation timing windows were free of CPU quota throttling. One follow-up
batch's separate auxiliary/certificate stage measurements were throttled;
those raw records are retained and excluded from clean stage comparisons.

The refined-hand benchmark uses the normal automatic recovery and initial box
factor one. Both versions produce 2,164 cells, try six auxiliary candidates,
and enlarge the symbolic box four times to factor 4,096. There is no diagnostic
strategy override, site snapping or jitter in these measurements. This remains
a standalone Delaunay fixture, **not a complete hand VTM solve or a CPU
speedup**. The wide CPU performance target remains open.

```bash
python -m benchmarks.refined_hand --repeats 3 --output refined-hand.json
python -m benchmarks.delaunay_recovery --case torus-boundary --repeats 5 --output torus.json
```

[Timing summary and source hashes](../results/validation/vtm/integer-performance.json),
[410-test record](../results/validation/vtm/integer-integrated-all-tests.txt),
[padded memcheck](../results/validation/vtm/integer-memcheck.txt),
[lifetime check](../results/validation/vtm/integer-lifetime.txt),
[native trace audit](../results/validation/vtm/integer-device-audit.json),
[100k CGAL comparison](../results/validation/vtm/integer-cgal.json),
[oriented output equivalence](../results/validation/vtm/integer-output-equivalence.json),
[independent hand validation](../results/validation/vtm/hand-integer-independent-validation.json),
and [source isolation](../results/validation/vtm/integer-isolation.json) retain
the evidence and scope. Existing diagram paths and README are unchanged;
filtered-predicate executable code and symbolic tie rules are unchanged.

[Issue #102](https://github.com/alecjacobson/warpower/issues/102) tracks this
adoption; the complete thin-hand example remains open in
[issue #93](https://github.com/alecjacobson/warpower/issues/93). Filling certified
partial meshes instead of repeating whole-cell clipping is tracked in
[issue #104](https://github.com/alecjacobson/warpower/issues/104).
