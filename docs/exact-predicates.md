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
not a numerical tolerance band. `estimate` is the leading float64 component of
the exact determinant expansion; it is approximate in magnitude. Invalid input,
nonfinite arithmetic, or expansion-capacity failure returns `valid=0` and must
not be interpreted as a coplanar/cospherical result.

## Arithmetic and compilation contract

These Warp functions use float64 expansion arithmetic: error-free sums, split
products, zero elimination, and two-pass compression. This follows the expansion
approach described by
[Shewchuk](https://www.cs.cmu.edu/~quake/robust.html). Warp 1.19 does not expose an
FMA builtin, so products use the splitter `2^27+1`. No native extension, custom
CUDA code, or CPU predicate fallback is involved.

**Calling kernels must compile with `fast_math=False` and `fuse_fp=False`.**
Warp functions compile into their caller's module; setting these options on the
function's defining module alone does not protect another kernel's compilation.
The recovery kernels set these options explicitly. Enabling reassociation
or contraction invalidates the error-free arithmetic assumptions.

Homogeneous determinants use original coordinates directly. They do not first
round a translated coordinate difference or a summed lifted coordinate. The
power lift `x*x + y*y + z*z - weight` is itself an expansion. Orientation products
have degree three; power predicates have degree at most five. The exponent range
of finite float32 inputs keeps these products and splitting intermediates within
float64 range. This argument does not extend to arbitrary float64 input sites.

Expansions reserve 64 float64 components and compress longer intermediate sums.
Every append checks capacity. The implementation deliberately reports a capacity
failure rather than relying on a claim that every possible input must fit. The
exact fallback evaluates all determinant terms. The optional
`_filtered_predicates` module first evaluates a float64 center/error ball for
each arithmetic operation, accepting a sign only when the ball excludes zero
(or is exactly zero). It falls back to these expansions otherwise. A rounded
normal operation's absolute error is bounded by `2^-52 * abs(result)`; positive
error sums/products receive additional outward padding. The finite float32
input and degree-five arithmetic restriction keeps intermediate values in
float64 range. Tests independently check that the rational determinant lies in
the reported ball, including extreme bit patterns and cancellation.

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
expansions for common lattice ties without introducing a tolerance. Large common
translations, fractional offsets, near-integer differences, and lost low bits
are checked against the independent rational oracle.

## Validation

Tests compare both sign and determinant estimate against an independent Python
rational/integer Bareiss determinant. Cases include exact coplanarity,
cospherical and weighted ties, one-ULP displacements, permutation sign changes,
translated and scaled nearly flat clouds, random float32 bit patterns, minimum
subnormals, and maximum finite float32 coordinates. Invalid input and bounded
expansion failure have separate checks.

A captured producer changes query points; a predicate kernel and downstream
reduction consume them without allocation or host work. Its native graph has no
host or allocation/free nodes. Six predicate tests pass CUDA memcheck with
32-byte allocation padding and no exclusions
([record](../results/validation/vtm/predicates-memcheck.txt)). The full suite passes
170 tests ([record](../results/validation/vtm/predicates-tests.txt)).

```bash
python -m pytest -q tests/test_predicates.py
compute-sanitizer --tool memcheck --padding 32 --error-exitcode 99 \
  python -m benchmarks.memcheck -- -q tests/test_predicates.py
```

The [optional constructor](delaunay-recovery.md) adds finite candidate recovery,
original-coordinate certification, and consistent symbolic lifting for ordinary
Delaunay meshes. Weighted regular recovery remains outside its scope. These
components do not by themselves establish the overall VTM goal.
