# API

For loops captured in CUDA graphs, use the preallocated
[`DiagramWorkspace` API](graphs.md), which keeps counts and control flow on the GPU.

```python
import warpower

diagram = warpower.voronoi_diagram(points)
diagram = warpower.power_diagram(points, weights)
```

`points` accepts a CUDA Warp `wp.vec3` array, a float32 CUDA PyTorch tensor of shape
`(N, 3)`, or a float32 NumPy array of that shape. NumPy inputs are uploaded to
`cuda:0`. Weights must be float32, have shape `(N,)` or `(N, 1)`, and be on the
points device. Inputs must be finite. Noncontiguous inputs are copied as needed.

The immutable result contains `adjacency`, `offsets`, `status`, `diagnostics`,
and an optional `tetrahedra` field (default `None`).
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
| 6 | Uncertain clipping decision awaiting precision recovery |

Plane, vertex, and traversal capacity failures grow storage at the current
precision. They do not trigger FP64 merely because the 512-slot limit is reached.
Numerical failures require FP64 and retain the storage size already reached;
subsequent capacity overflow cannot send that cell back to FP32. These rules
apply to ordinary calls and captured execution, and reset for each construction.

A cancellation screen sends suspicious clipping decisions to a separate kernel
that checks the original site coordinates. Interior decisions use filtered/exact
predicates; unresolved box-plane decisions and mismatches request FP64 recovery.
Up to 512 eligible cells per pass use a 128-slot cooperative workspace, with 32
threads per cell. Cells that already require more storage bypass it. Each cell
gets at most one cooperative attempt, then uses the serial FP64 fallback when
needed. A numerical failure in that fallback remains explicit; it does not grow
buffers to try to resolve a precision problem. Output-row overflow widens the
row without unnecessarily increasing clipping scratch capacity.
Remaining failures are explicit and produce no partial neighbor list. The screen
is heuristic and recovery uses floating-point geometry: successful statuses do
not certify an exact diagram.

`diagnostics["cooperative_precision_attempts"]` counts attempts in those bounded
GPU recovery passes, at most one per site per construction.
`retried_cells` counts the cells entering subsequent host-dispatched retries;
`double_precision_cells` counts cells queued for that later serial FP64 phase.
These counters describe different stages and are not a count of
distinct cells that used FP64.

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

## Optional full tetrahedral mesh

```python
diagram = warpower.voronoi_diagram(points, return_tetrahedra=True)
tets = diagram.tetrahedra  # int32 CUDA array/tensor, shape (T, 4)

# Weights select the dual regular (weighted Delaunay) tetrahedralization.
weighted = warpower.power_diagram(points, weights, return_tetrahedra=True)
```

Each row indexes the original input points and has positive signed volume under
`dot(p1-p0, cross(p2-p0, p3-p0))`. Row order is unspecified. Coincident sites use
the same deterministic ownership rule as the diagram. Weighted sites whose
unbounded power cells are empty are absent from the mesh.

On success, the tetrahedra cover the **complete convex hull of the sites**.
Finite dual vertices can lie far outside the ordinary diagram's clipping box,
so this option performs a separate float64 Warp construction with adaptive domain
expansion. Artificial box vertices are excluded. Before return, GPU checks verify
positive orientation, empty power spheres, paired interior faces with opposite
orientations, and supporting convex-hull boundary faces. Diagnostics include
`tetrahedra_count`, `tetrahedra_expansions`, `tetrahedra_padding`, and a description
of the validation performed. Inputs with fewer than four sites, or all sites at
one position, produce an empty `(0, 4)` array.

The existing `adjacency`, `offsets`, and `status` still describe the original
**bounded diagram**, unchanged by this option. In particular, a site whose cell
is empty inside that box can still occur in the complete weighted mesh, and the
mesh can contain boundary edges absent from the bounded diagram's adjacency.

The optional path uses floating-point predicates, not exact arithmetic or
symbolic perturbation. It raises `ValueError` if construction or validation
cannot establish a complete mesh, including ambiguous cospherical/copower
configurations, near-coplanar tetrahedra, unsupported lower-dimensional inputs,
and exhausted capacity/domain-expansion limits. It never returns a partial mesh
as a complete tetrahedralization. A cube's eight cospherical corners, for example,
require a degeneracy tie-breaking policy and are currently rejected.

`return_tetrahedra=False` is the default. The optional module is imported only
when requested, and no tetrahedron kernels, buffers, or validation passes run on
the default path. The existing adjacency construction kernels are unchanged.
## Standalone tetrahedralization

`warpower.delaunay_tetrahedra(points, *, chunk_size=None, leaf_size=4)` returns
positively oriented int32 CUDA tetrahedra of shape `(T, 4)` covering the full
convex hull. `warpower.regular_tetrahedra(points, weights, ...)` is its weighted
counterpart. Both accept the same input types as the diagram functions, return
Warp arrays (or PyTorch tensors for PyTorch input), and skip bounded adjacency.
They share the optional mesh's numerical limitations and raise on unresolved
ambiguity. These exact-size convenience calls allocate and synchronize; for
captured device loops use
`DiagramWorkspace(points, weights, adjacency=False, return_tetrahedra=True)`.
