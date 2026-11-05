# API

```python
import warpower

diagram = warpower.voronoi_diagram(points)
diagram = warpower.power_diagram(points, weights)
```

`points` accepts a CUDA Warp `wp.vec3` array, a float32 CUDA PyTorch tensor of shape
`(N, 3)`, or a float32 NumPy array of that shape. NumPy inputs are uploaded to
`cuda:0`. Weights must be float32, have shape `(N,)` or `(N, 1)`, and be on the
points device. Inputs must be finite. Noncontiguous inputs are copied as needed.

The immutable result contains `adjacency`, `offsets`, `status`, and `diagnostics`.
The first three are int32 CUDA Warp arrays, or PyTorch tensors when the points
input is a tensor. PyTorch calls honor the current PyTorch CUDA stream. Adjacency
uses symmetric CSR, with `N+1` offsets; order within a row is unspecified. Empty
inputs produce empty adjacency/status and offsets `[0]`.

Exactly coincident sites have one deterministic owner: greatest weight, then
smallest input index. Other coincident sites are empty and never occur as
neighbors. The bounding domain is the input bounding box expanded by `1.00001`
in each direction.

| Status | Meaning |
| --- | --- |
| 0 | Construction succeeded |
| 1 | Plane capacity exhausted |
| 2 | Vertex capacity exhausted |
| 3 | Empty or non-owning coincident cell |
| 4 | Traversal queue capacity exhausted |
| 5 | Inconsistent floating-point clipping boundary |

Failed cells are retried with larger capacity and, when needed, double-precision
clipping before return. Remaining failures
are explicit; they produce no partial neighbor list. Inspect `status` when using
difficult geometry. Floating-point predicates are not exact.
One-sided facets also trigger double-precision recovery of both cells before
symmetrization. `diagnostics["consistency_retried_cells"]` counts these recovery
jobs, including repeated jobs, and `diagnostics["asymmetric_cells_remaining"]`
reports unresolved consistency after up to three passes.

Keyword arguments match Paragram: `debug`, `knn`, `leaf_size`, `initial_guesses`,
and `initial_guesses_offsets`. Initial guesses are flat int32 CSR hints; both
arrays are required together. Indices must be valid sites or `-1` to skip a hint.
Offsets must partition the entire hint array. Hints influence clipping order;
the complete hierarchy traversal still runs.

Like Paragram's L40 path, construction starts with the nearest leaf and disables
the separate KNN pass. `knn` is accepted as a nonnegative compatibility argument;
`diagnostics` reports its requested and effective values. `leaf_size=-1` selects
the tuned default of 4. An explicit positive leaf size is supported.

Warpower adds `chunk_size` for an explicit maximum number of concurrent cells.
Alternatively set `WARPOWER_NUM_CHUNKS` or `WARPOWER_MEM_FRAC` (default `0.5`). The
corresponding `PARAGRAM_*` environment variables are recognized for compatibility.
Chunking limits temporary clipping storage; inputs, output, and the hierarchy
remain resident. Topology is discrete and no differentiation is provided.
