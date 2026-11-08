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
