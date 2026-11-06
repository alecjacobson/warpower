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

## Validation in progress

The integrated implementation passes 16 targeted tests: the existing exact,
filtered and mixed-halfspace rational-oracle tests, plus signed integer
arithmetic, long carries/borrows, explicit overflow and all power-of-two
exponents from −800 through 800. Random raw float32 bit patterns exercise the
full exponent range. Invalid-input and capacity failures remain explicit.

The private prototype reduced the same 384-site hand diagnostic from 41.033 to
10.718 seconds, both with one symbolic box and the same unsuccessful output.
Starting the symbolic strategy with a box factor of 4,096 produced a complete
2,164-tetrahedron hull in 6.678 seconds. This is a diagnostic configuration, not
a new default, a complete hand VTM solve, or a CPU speedup result. All three
measurement windows observed zero CPU quota throttling; setup/compilation is
excluded. The successful mesh passes the preceding expansion-based global
symbolic certificate. It differs from native CGAL in 33 cells on each side;
independent Python-integer queries against every site confirm that every such
cell has an empty open sphere and one or two additional exactly cospherical
sites. The differences are tie choices, not a connectivity-equivalence claim.

Full integration, regression timing, native graph and memory checks are still
pending. [Issue #102](https://github.com/alecjacobson/warpower/issues/102) tracks
adoption; the complete thin-hand example remains open in
[issue #93](https://github.com/alecjacobson/warpower/issues/93).
