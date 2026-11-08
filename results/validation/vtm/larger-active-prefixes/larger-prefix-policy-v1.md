# Larger active prefixes: prototype and proposed ownership

Baseline: df74493c97633324147cb39ef4c41a7d34c7b90a (PR236).

The proposed production change keeps ActiveDelaunayWorkspace's current direct
private default (1024, 2048). VTM supplies an explicit increasing tuple from
(1024, 2048, 4096, 8192), filtered below its reserved capacity. The router's
validation upper bound becomes 8192. This keeps the public standalone diagram
and Delaunay paths unchanged and makes the VTM owner responsible for paying
for additional child storage. Large reservations retain the full constructor.

The private prototype copies only the original setup method, expands its
validation bound, and supplies this default policy before VTM setup. All device
kernels and publish, reset, conditional routing, fallback and run methods are
inherited from the current implementation. It adds no geometric shortcut or
weaker certificate. All prefix views and constructors are allocated in setup.

Taking a fitting prefix preserves every active original site and original ID.
Inactive sites have already been prepared as exact copies of point zero, so
omitting their later duplicate IDs cannot change the canonical first owner.
Each selected constructor still certifies the current full active convex hull.
A failed child keeps the existing full-reservation fallback; public outputs
stay in the full constructor's buffers. Fresh solve resets clear all children.

Qualification starts with captured 4095/4096/4097 and 8191/8192/8193 transitions,
growth/shrinkage, unrelated input, invalid counts/nonfinite/flat inputs and
recovery, duplicate first owners, exact ties and extreme original scales.
Then fresh interleaved complete solves measure target4K/reserved16K, progressive
8K, fixed16K and graded16K. Preserve every raw result, including any changed
trajectory. Independent validation and strict bitwise/work comparisons follow
only after clean timings finish. Record persistent bytes, setup and first replay.

This is a private experiment, not yet a production change or speedup claim.
Scratch sharing remains separate issue122. A capacity-cost reduction alone does
not establish the overall comparable-quality CPU performance goal.
