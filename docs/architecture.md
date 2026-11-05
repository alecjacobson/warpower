# Construction

Diagrams are built by independent GPU threads. Each thread clips a
bounded convex cell against candidate bisectors, then emits its active neighbor
planes. The final pass produces symmetric CSR adjacency. All project GPU code is
written as NVIDIA Warp kernels; sorting and scans use Warp's supplied primitives.

The clipper and directional bounds follow
[Paragram](https://arxiv.org/abs/2605.06408). Plane coordinates are relative to the
seed, avoiding subtraction of large squared world coordinates. Signed power
distances are handled before squaring. Failed cells produce explicit status codes
and are retried with larger storage; traversal never silently discards a valid
candidate to fit a queue.

## Spatial hierarchy

Points are sorted by 63-bit Morton codes. A
[binary radix tree](https://research.nvidia.com/publication/2012-06_maximizing-parallelism-construction-bvhs-octrees-and-k-d-trees)
provides spatially aligned subtrees, which are packed into small leaves. Fixed-size
runs of Morton-sorted points can cross large spatial discontinuities and are poor
substitutes for these subtrees.

A temporary segment tree supplies exact range bounds and maximum weights for
radix nodes. This uses ordered Warp kernel launches, avoiding an inter-block
publication protocol requiring custom CUDA memory fences. The additional arrays
are temporary and construction remains linear in storage.

## Tighter candidate rejection

Directional cell-box bounds are cheap but can remain loose for cells between
widely separated clusters. Before revisiting a queued node, an additional test
uses the current cell vertices.

Let `x` be a vertex relative to the seed, `q` a candidate site relative to the seed,
and `delta = seed_weight - node_max_weight`. The candidate cannot remove this
vertex if

```
|q|² - 2 x·q + delta > 0.
```

Over the node's axis-aligned box, the minimizing `q` is `clamp(x, lower, upper)`.
If the minimum is positive for every cell vertex, every site in that node leaves
the whole convex cell unchanged. The implementation evaluates this expression
directly to avoid subtracting two large squared distances, and applies a relative
roundoff margin. This is a conservative geometric bound in exact arithmetic;
the implementation uses floating-point arithmetic and does not claim exact
predicates.

## Validation and remaining work

Small diagrams are compared against an independent double-precision all-halfspace
oracle. Hierarchy tests verify exact subtree bounds, maximum weights, permutation,
leaf coverage, and coincident Morton keys. Queue retries, chunking, strong weights,
and separated clusters are exercised separately. CUDA memory checks cover the
clipper and complete pipeline.

Performance and API completion are tracked in
[issue 5](https://github.com/alecjacobson/warpower/issues/5) and
[issue 4](https://github.com/alecjacobson/warpower/issues/4). Floating-point boundary
failures can still occur; callers should inspect `status`.
