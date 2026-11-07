# Bounded binary32 predicate proof

The VTM first stage uses the determinant/permanent grouping in
`warpower/_analytic_predicates.py`, adapted from Shewchuk's public-domain
[May 18, 1996 reference](https://www.cs.cmu.edu/~quake/robust.html). The source copy
and URL/SHA256 are retained with the validation artifacts.

The source predicate formulas operate on original binary32 coordinates. They
enter the new arithmetic only when every relevant coordinate is finite and has
magnitude at most 16. Other inputs return an unavailable filter result. Only a
strict nonzero sign is accepted; uncertainty retains the existing binary64,
Ball64, integer and symbolic path. No floor or error bound defines a geometric
tie. Calling kernels disable fast math and fused multiply-add; the expression
tree is part of the specification.

## Relative arithmetic

Let u=2^-23, conservatively twice binary32 round-to-nearest unit roundoff, and
let each relative rounding factor lie in [1-u,1+u]. Treat each exact coordinate
difference as a leaf followed by its subtraction rounding. Expand the grouped
determinant into signed monomials of these exact differences. There are six
orientation monomials and 72 sphere monomials. The grouped permanent contains
exactly their absolute monomials; the proof script verifies this polynomial
identity without sampled geometry.

A multiplication adds its operands' rounding-factor counts and one; an addition
or subtraction takes their maximum and adds one. Negation and absolute value
do not add a relative factor. The maximum determinant counts N are 8 and 16.
The maximum permanent counts M are 8 and 17. Shared factors and repeated factors
do not invalidate a bound on the total number along a monomial.

For the corresponding relative-only calculation D_r, P_r, each determinant
monomial's coefficient differs from one by at most (1+u)^N-1. Each nonnegative
permanent monomial has a coefficient at least (1-u)^M. Consequently

    |D_r - D_exact| <= A P_r,
    A = ((1+u)^N - 1)/(1-u)^M.

This deliberately uses a generic expression-tree bound; it does not assume that
an artificial relative-only calculation retains special exact-cancellation
properties of a physical floating-point machine. The first-stage reference's
more specific constants are not needed for this argument.

## Absolute underflow allowance

Let eta=2^-125, twice the smallest positive normal binary32 value. Each operation
may flush a subnormal operand, and its rounded result may incur an absolute
underflow/flush error. Such an effect has magnitude below eta. This model covers
both gradual underflow and operand/result flushing; normal operations need only
their relative factor. The bounded range prevents overflow.

Compare the actual computation with a relative-only computation using the same
relative factors. For each intermediate, bound the latter's magnitude by M_x and
their absolute difference by E_x. Original coordinate leaves have M_x=16, E_x=0.
For addition and subtraction use

    M_z = (M_x + M_y)(1+u),
    E_z = (E_x + E_y + 2 eta)(1+u) + eta.

For multiplication, set a=E_x+eta, b=E_y+eta and use

    M_z = M_x M_y (1+u),
    E_z = (M_x b + M_y a + a b)(1+u) + eta.

Absolute value and negation keep M_x and conservatively increase E_x by eta.
These recurrences allow operand flushing before the operation and output
underflow afterward. The exact-rational script evaluates them on the source
expression tree. It bounds every actual intermediate by M_x+E_x, below 2^18
for orientation and 2^32 for the sphere calculation, far below binary32 overflow.

Let E_D and E_P be the final absolute differences between actual and relative-only
determinant and permanent. Since the actual permanent P is nonnegative,

    |D - D_exact| <= A P + E_D + A E_P.

The script proves E_D+A E_P < 2^-108 for orientation and < 2^-94 for sphere.
These are global bounds throughout the admitted coordinate range, including
subnormal coordinates and cancellation. They are not empirical maxima.

## Rounded threshold and consumption

Use C=2^-19 for orientation, C=2^-17 for sphere, and F=2^-64 for both. The runtime
computes the binary32 threshold B=fl(fl(C P)+F), then promotes B and the computed
determinant exactly to binary64 for the existing filter interface. A conservative
lower bound, including threshold operand/output flushing, is

    B >= C(1-u)^2 P + F(1-u) - 8 eta.

The exact-rational script verifies both strict inequalities

    C(1-u)^2 > A,
    F(1-u) - 8 eta > E_D + A E_P.

Thus B encloses the entire determinant error, including its own computation.
Finite |D|>B proves a nonzero exact sign; equality falls back. An accepted D is
normal because B is at least the normal F, so the comparison cannot accept a
flushed subnormal sign. The power-of-two constants are exactly representable.

## Checks

`benchmarks/bounded_predicate_proof.py` reads the current formula grouping, verifies the
6/72-monomial permanent identity, counts the 8/8 and 16/17 factors, and checks all
inequalities using Python Fraction. Its JSON records the source hash and exact
fractions. The binary32 implementation preserves that grouping while changing
input arithmetic type, the range gate, threshold and final exact promotion.

The arithmetic tests use independent rational original-coordinate determinants.
They evaluate both ordinary binary32 arithmetic and a test-only implementation
that explicitly flushes every subnormal operand/result at each operation. Cases
cover random bounded inputs, finite bit patterns, tiny/large scales, range
boundaries, exact ties, one-ULP changes, permutations, near-singular shears and
nonfinite values. Unsupported ranges are checked for unavailable results. The
algebraic argument establishes the bound; these tests check implementation and
exercise its difficult cases.

The executable checker `python -m benchmarks.bounded_predicate_proof` verifies
that the runtime formula trees, finite range gates, constants and consumer
compiler options match this argument. It then checks the monomial identity and
all stated rational inequalities. `tests/test_bounded_predicate_proof.py` runs
that check as part of the ordinary suite.
