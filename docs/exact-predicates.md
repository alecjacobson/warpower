# Exact predicate foundation

The private `warpower._predicates` module provides exact orientation and weighted
in-sphere signs for **finite float32 coordinates and weights**. It is the
certification foundation for [Delaunay recovery](https://github.com/alecjacobson/warpower/issues/32),
and is now used by the optional [certified constructor](delaunay-recovery.md).
The original diagram and tetrahedralization paths do not import it.

`orient3d(a,b,c,d)` uses the sign convention of `det(b-a,c-a,d-a)`.
`power_sphere(a,b,c,d,q,wa,wb,wc,wd,wq)` is positive when `q` lies inside
the power sphere of a positively oriented defining tetrahedron. Swapping two
defining vertices reverses the sign. `insphere` supplies zero weights.

Each function returns `Predicate(sign, estimate, valid)`. When `valid` is one,
`sign` is exactly -1, 0, or 1 for the stored inputs. Zero denotes an exact tie,
not a numerical tolerance band. `estimate` approximates the determinant magnitude in float64. Invalid input,
nonfinite arithmetic, or bounded-capacity failure returns `valid=0` and must
not be interpreted as a coplanar/cospherical result.

## Arithmetic and compilation contract

The exact fallback uses bounded signed integer arithmetic for dyadic float32
inputs. It forms translated differences and lifted coordinates exactly, then
evaluates factored determinants. The [representation and capacity argument](integer-predicates.md)
shows that the 48-word storage covers the finite float32 input range, while
explicit overflow checks retain fail-closed behavior. No native extension,
custom CUDA code or CPU predicate fallback is involved.

**Calling kernels must compile with `fast_math=False` and `fuse_fp=False`.**
Warp functions compile into their caller's module; setting these options on the
function's defining module alone does not protect another kernel's compilation.
The recovery kernels set these options explicitly for the interval filters.

The unchanged `_filtered_predicates` module first evaluates a float64
center/error ball for each arithmetic operation, accepting a sign only when
the ball excludes zero (or is exactly zero). It falls back to exact integers
otherwise. A rounded normal operation's absolute error is bounded by
`2^-52 * abs(result)`; positive error sums/products receive additional outward
padding. The finite float32 input and degree-five arithmetic restriction keeps
intermediate values in float64 range. Tests independently check that the
rational determinant lies in the reported ball, including extreme bit patterns
and cancellation. This argument does not extend to arbitrary float64 input sites.

For uncertain filters, a second fast path handles exact bounded integer
differences. Error-free `TwoDiff` must establish that each original translated
coordinate difference is an integer with absolute value at most 512. Merely
rounding to an integer is insufficient: a lost tiny operand disqualifies this
path. Power weights must also be exact integers with absolute value at most
262144. Every intermediate translated determinant product and sum then fits
exactly in float64: the power determinant is bounded by
`24*M^3*(3*M^2 + 2*W) = 120*512^5 < 2^52`; orientation is bounded by
`6*512^3 < 2^30`. The existing float64 determinant value is therefore exact,
including zero, despite a conservative ball that may overlap zero. This avoids
the integer fallback for common lattice ties without introducing a tolerance. Large common
translations, fractional offsets, near-integer differences, and lost low bits
are checked against the independent rational oracle.

## Validation

Tests compare both sign and determinant estimate against an independent Python
rational/integer Bareiss determinant. Cases include exact coplanarity,
cospherical and weighted ties, one-ULP displacements, permutation sign changes,
translated and scaled nearly flat clouds, random float32 bit patterns, minimum
subnormals, and maximum finite float32 coordinates. Invalid input and bounded
capacity failure have separate checks.

A captured producer changes query points; a predicate kernel and downstream
reduction consume them without allocation or host work. Its native graph has no
host or allocation/free nodes. The preceding expansion implementation passed six predicate tests under CUDA
memcheck with 32-byte allocation padding and no exclusions
([record](../results/validation/vtm/predicates-memcheck.txt)). That historical full suite passed
170 tests ([record](../results/validation/vtm/predicates-tests.txt)). Current integer-fallback
validation is tracked [separately](integer-predicates.md); those historical
records are not validation of a changed implementation.

```bash
python -m pytest -q tests/test_predicates.py
compute-sanitizer --tool memcheck --padding 32 --error-exitcode 99 \
  python -m benchmarks.memcheck -- -q tests/test_predicates.py
```

The [optional constructor](delaunay-recovery.md) adds finite candidate recovery,
original-coordinate certification, and consistent symbolic lifting for ordinary
Delaunay meshes. Weighted regular recovery remains outside its scope. These
components do not by themselves establish the overall VTM goal.

## Original-coordinate cell clipping

The optional symbolic constructor also needs to classify vertices incident to
clipping-box planes. `_halfspace_predicates` represents a site bisector for owner
`p` and neighbor `q` as `(-2(q-p), |q|²-|p|²+w_p-w_q)`, with the interior on the
nonnegative side. A box plane has an axis-aligned normal and its original
float32 bound as its constant. For three defining rows `A` and a query row `Q`,
`det4(A,Q) = det3(A.xyz) * Q(vertex)`. Opposite determinant signs identify an
outside vertex without rounding the intersection coordinates.

Float64 balls filter those signs. The exact fallback constructs the four
halfspace rows directly with signed integer differences, squares and weights.
Coordinates and normals use units `2^e`, while constants use `2^(2e)`; every
4×4 determinant therefore has the same scale factor `2^(5e)`, including mixed
box rows and the constant-one row for the normal determinant. No rounded plane
coefficient is promoted to an exact value. Products have degree at most five.

Zero query determinants use the certificate's infinitesimal hash weights and
then successively smaller positive unit weights in site-index order. This rule
also applies at box vertices. Singular defining normals return an explicit
invalid predicate; they never become accepted finite vertices. Rational-oracle
tests cover mixed box/site rows, extreme finite float32 bit patterns, row
permutations, and symbolic ties.
