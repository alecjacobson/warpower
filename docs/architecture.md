# Construction

Diagrams are built by independent GPU threads. Each thread clips a
bounded convex cell against candidate bisectors, then emits its active neighbor
planes. The final pass produces symmetric CSR adjacency. All project GPU code is
written as NVIDIA Warp kernels; sorting and scans use Warp's supplied primitives.

## Comparison with Paragram

This comparison describes the code we benchmarked: Paragram
[`4f36b1f`](https://github.com/zenseact/paragram/tree/4f36b1f2c8738912c5efc056a06023d4c1d2da07)
and Warpower's implementation at
[`d38b80c`](https://github.com/alecjacobson/warpower/tree/d38b80c).
GPU-dependent defaults below refer to the NVIDIA L40 path. Links in the table
point to the corresponding source files.

| Aspect | Paragram | Warpower |
| --- | --- | --- |
| GPU implementation and dependencies | Handwritten C++/CUDA extension loaded through PyTorch; cuBQL supplies the BVH builder. Source builds use NVCC. [Backend][p-backend] | Python-defined Warp kernels; Warp supplies compilation, radix sorting, and scans. No project C++/CUDA extension; PyTorch is optional. [Package dependencies](../pyproject.toml), [hierarchy](../warpower/_bvh.py) |
| Shared construction method | One GPU thread clips each bounded convex cell against candidate bisectors, represents vertices by triples of incident planes, and exports active neighbor planes. Scratch storage is interleaved across 32 cells for coalesced access. [Clipper][p-cell] | Retains that construction method and 32-cell storage layout. One generic clipper serves Voronoi/power construction and float32/float64 recovery. Plane triples use three 10-bit indices packed into a `uint32`, allowing larger retry capacities. [Clipper](../warpower/_cell.py) |
| Plane coordinates | Normalized plane equations in world coordinates, with the offset formed from the sum and difference of the two site positions. [Bisectors][p-cell] | Also normalizes planes, but expresses them relative to the seed, using position differences and weight differences. This reduces sensitivity to large world-coordinate offsets. [Clipping](../warpower/_cell.py) |
| Spatial hierarchy | cuBQL's default spatial-median GPU builder; power diagrams add node maximum weights through bottom-up propagation with atomic arrival counters. [Builder][p-builder], [cuBQL configuration][p-cubql] | 63-bit Morton sorting followed by a Karras binary radix tree. A temporary segment tree supplies subtree bounds and maximum weights through ordered kernel launches. This adds temporary arrays and avoids a custom inter-block publication protocol. [Builder](../warpower/_bvh.py) |
| Initialization and tuning | Default leaf threshold 17 on L40; the separate KNN pass is disabled on this path. Traversal first descends to a nearby leaf. CSR initial guesses are supported. [Entry point][p-entry], [traversal][p-traversal] | Default leaf threshold 4; starts with a nearby leaf and supports CSR initial guesses. Accepts `knn` for compatibility but keeps the separate KNN pass disabled on all devices; reports requested/effective values in diagnostics. [API](../warpower/_api.py), [traversal](../warpower/_construct.py) |
| Candidate rejection | Directional cell-box bounds and distance-to-bisector tests; weighted traversal also uses each node's maximum weight. [Traversal][p-power-traversal], [clipper][p-cell] | Uses directional bounds plus an extra test minimizing the bisector expression over a node box for every current cell vertex. Weighted rejection checks the signed numerator before squaring. [Bounds and traversal](../warpower/_construct.py), [clipper](../warpower/_cell.py) |
| Traversal queue overflow | Best-first traversal uses 64 entries. When full, a nearer incoming entry replaces the farthest queued entry; otherwise the incoming entry is discarded. [Voronoi][p-traversal], [power][p-power-traversal] | Starts with 128 entries. Compacts entries rejected by the updated cell bounds; if still full, records failure and retries with capacity up to 512. Does not evict a potentially valid candidate to fit the queue. [Traversal](../warpower/_construct.py), [retry schedule](../warpower/_api.py) |
| Capacity and precision recovery | Float32 clipping with fixed limits of 64 planes and 64 vertices. The public construction call returns statuses without an automatic larger-capacity or float64 retry. [Clipper][p-cell], [entry point][p-entry] | Starts at 64, retries failures at 128 in float32, then at 128/256/512 in float64. Retains successful rows and reuses failed candidates as retry seeds. Sparse retries use one active cell per GPU warp; float64 Cartesian vertex positions are cached. [Retry schedule](../warpower/_api.py), [scratch storage](../warpower/_cell.py) |
| Failed-cell output | Exports active neighbor planes even when a cell's final status is unsuccessful; callers must inspect status. [Power entry point][p-entry], [Voronoi entry point][p-voronoi-entry] | Unresolved failed cells have explicit statuses and empty CSR rows. Candidate IDs retained for retries are never emitted as partial failed-cell adjacency. [Emission](../warpower/_cell.py) |
| Adjacency consistency and CSR | Gathers and reorders raw rows, scans counts, then checks reverse connections and inserts missing reverse edges with atomics. Does not reclip cells to resolve one-sided facets. [Assembly][p-adjacency], [entry point][p-entry] | Checks raw facet reciprocity first and retries both endpoints of one-sided facets in float64, for up to three passes. If reciprocal, scans counts and copies rows directly; otherwise uses the general symmetric compactor and reports unresolved asymmetry. [Assembly](../warpower/_adjacency.py), [recovery](../warpower/_api.py) |
| Exactly coincident sites | No global duplicate-ownership pass in the construction pipeline. [Entry point][p-entry] | GPU hash pass selects the greatest weight, then the lowest input index. Other coincident sites are empty and excluded from neighbors; signed zeros are treated as the same position. [Ownership](../warpower/_canonical.py) |
| Python interface | Float32 CUDA PyTorch inputs; immutable result with int32 `adjacency`, `offsets`, and `status` tensors. [API][p-api] | Also accepts native CUDA Warp arrays and NumPy inputs. Preserves the three result fields for PyTorch callers, adds diagnostics and `chunk_size`, and honors the current PyTorch CUDA stream. [API](../warpower/_api.py), [compatibility details](api.md) |
| Chunking and allocator policy | Chunks temporary clipping storage using `PARAGRAM_NUM_CHUNKS` or `PARAGRAM_MEM_FRAC` (default 0.5); tensor allocations use PyTorch's cache. [Chunking][p-chunking] | Adds `chunk_size` and `WARPOWER_*` equivalents, with fallback to the Paragram environment variables. The published benchmarks configure 32 GiB of Warp allocator retention; the library leaves that global policy to the application. Inputs, hierarchy, and output stay resident. [API](../warpower/_api.py), [benchmark policy](benchmarks.md) |

The diagram construction paths compared here use floating-point predicates;
the optional certified Delaunay constructor is described separately below.
Neither this comparison nor
successful statuses establish exact geometric equivalence. The
[numerical audit](numerical-audit.md) records observed adjacency differences and
the scope of independent checks. The [full benchmark report](../results/summary/report.md)
records timings and memory costs, including slower cases; it measures the whole
pipeline and does not attribute the aggregate speedup to any single change above.

[p-backend]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/_backend.py
[p-cell]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/faster_convex_cell.cuh
[p-builder]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/pwr_bvh.cu
[p-cubql]: https://github.com/NVIDIA/cuBQL/blob/d18c5fa1a5c98665d13484841ae65774da7751e8/cuBQL/bvh.h
[p-entry]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/laguerre_ultra.cu
[p-voronoi-entry]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/voronoi_ultra.cu
[p-traversal]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/voronoi_bvh.cuh
[p-power-traversal]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/pwr_bvh.cuh
[p-adjacency]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/adjacency.cuh
[p-api]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/_api.py
[p-chunking]: https://github.com/zenseact/paragram/blob/4f36b1f2c8738912c5efc056a06023d4c1d2da07/paragram/csrc/laguerre/chunking.h

## Cell construction and recovery

The clipper and directional bounds follow
[Paragram](https://arxiv.org/abs/2605.06408). Plane coordinates are relative to the
seed, avoiding subtraction of large squared world coordinates. Signed power
distances are handled before squaring. Failed cells produce explicit status codes
and are retried with larger storage; traversal never silently discards a valid
candidate to fit a queue.

Recovery first increases single-precision capacity, then retries unresolved cells
with double-precision planes, homogeneous vertex coordinates, and clipping tests.
The same generic Warp clipping algorithm serves both precisions. Valid candidate
IDs from a failed attempt seed the retry; failed cells still emit no partial CSR
row. Completed retry rows are retained. Sparse retry jobs use separate GPU warps
so a few difficult cells can run across multiple multiprocessors. Double-precision
Cartesian coordinates are cached when a vertex is created, avoiding repeated
division during bounds updates and candidate rejection.

After explicit failures are recovered, a GPU pass checks whether each directed
facet is present in the opposite cell. One-sided facets identify possible
numerical errors even when both cells report success. Both endpoints are retried
in double precision before CSR symmetrization. Up to three consistency passes
handle newly exposed neighbors; diagnostics report any remaining asymmetric
cells. This improves floating-point consistency without claiming exact geometry.
When all directed facets are reciprocal, CSR output scans the existing row
counts and copies each row directly. It avoids repeating reverse-neighbor
searches and atomic symmetrization. Unresolved cases use the general symmetric
compactor.

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

## Validation

Small diagrams are compared against an independent double-precision all-halfspace
oracle. Hierarchy tests verify exact subtree bounds, maximum weights, permutation,
leaf coverage, and coincident Morton keys. Queue retries, chunking, strong weights,
and separated clusters are exercised separately. CUDA memory checks cover the
clipper and complete pipeline.

The [full benchmark report](../results/summary/report.md) covers the paper's
synthetic sizes and nine real scenes on the L40. The [API guide](api.md) documents
inputs, outputs, statuses, and memory controls. Floating-point predicates are
not exact; callers should inspect `status` and consistency diagnostics.

## Optional tetrahedral dual

`return_tetrahedra=True` adds a complete Delaunay mesh for unweighted sites or a
regular tetrahedral mesh for weighted sites. This is a later addition to the
implementation versions compared above. It is isolated in
[`_tetrahedra.py`](../warpower/_tetrahedra.py); the default adjacency kernels and
their launch path are unchanged, and the module is imported lazily.

Cell adjacency alone records dual edges, not which four cells meet at a diagram
vertex. The clipper's vertex records already contain triples of incident planes.
Together with the seed index, three real site planes identify a tetrahedron.
Vertices incident to artificial box planes are excluded; minimum-site ownership
and GPU hashing remove duplicate tetrahedra, and a float64 orientation test gives
positive-volume connectivity.

Simply extracting the vertices inside the default finite box would miss valid
tetrahedra near the convex hull. The optional path therefore rebuilds cells in
float64 using a separate Warp BVH traversal, enlarging the box by a factor of
eight until the candidate mesh passes validation. All-site empty-power-sphere
queries and convex-hull support tests use the BVH; a face hash checks incidence
and opposing orientations. A boundary face that does not support the input hull
signals missing tetrahedra and triggers another domain expansion.

This validation accepts full-dimensional, numerically resolved simplicial duals.
Cospherical/copower ambiguity and other unresolved numerical or topology failures
raise an error rather than expose a partial mesh. It does not implement symbolic
perturbation. The [API guide](api.md#optional-full-tetrahedral-mesh) describes the
output, limitations, and distinction between the original bounded adjacency and
the complete tetrahedral dual.

The [validation report](tetrahedra-validation.md) records independent mesh
comparisons and paired default-performance checks.

The newer optional [`DelaunayWorkspace`](delaunay-recovery.md) adds exact
original-coordinate certification, consistent symbolic lifting for ties, and
finite normalized/weighted candidate recovery, and an original-coordinate
symbolic cell fallback. That fallback defines topology with incident plane IDs
and the certificate's exact tie rule. Conservative cached Cartesian intervals
prove strict signs and directional bounds reject irrelevant sites; uncertain
cases retain the exact site/box predicates. See [symbolic geometry](symbolic-bounds.md). It is an independent API used
by the VTM core. Large inputs can first use
[float32 interior candidates with float64 cell repair](fast-candidates.md),
followed by the same exact certificate and full original-constructor fallback.
This optimization owns separate scratch and leaves the basic construction
kernels intact. The original diagram and tetrahedralization paths
above are unchanged; certification and recovery costs apply only to this API.

For difficult small/medium buffers, [certified fragment completion](facet-completion.md)
can filter valid original-coordinate tetrahedra from failed auxiliary candidates
and extend their exposed faces. Exact symbolic sphere ordering selects each new
opposite vertex, and a full global certificate gates the final result. It avoids
repeated large-box clipping on the refined-hand fixture while retaining finite
budgets and the existing fallback. Published cells and their matching face index
remain available to downstream automatic sizing and meshing on device.

## Captured device execution

The optional [`DiagramWorkspace`](graphs.md) API owns persistent arrays and
captures construction with CUDA conditional nodes. Scalar counts, input errors,
retry decisions, bounding-box expansion, and mesh validation outcomes remain on
the device. Output capacities are fixed at setup; GPU counts delimit their valid
prefixes. Both counts and CSR offsets are cleared before consumers run if an
error occurs.

This path reuses the eager hierarchy, clipper, and geometric predicates, with
device-count traversal and validation variants. Failed cells and inconsistent
adjacency trigger conditional recovery graphs. A preallocated hierarchical Warp
tile scan replaces the native allocating scan in conditional branches. Optional
tetrahedralization retains its last successful domain scale as a starting point
and validates the current mesh on every replay. The original eager construction
kernels are unchanged.

## Optional variational optimization

The lazy [`ODTWorkspace`](odt.md) layer consumes the independently callable
full-hull mesh. It accumulates uniform interpolation energy and stable
volume-weighted circumcenter displacements in float64, holds the initial hull
fixed, and tests parallel updates after rebuilding Delaunay connectivity.
Uniform and graded accumulation use a [filtered exact volume fallback](stable-volumes.md)
for nearly cancelling determinants of the original float32 sites.
Per-site displacement limits account for the hull and float32 storage; captured
conditional loops handle energy rejection, convergence, and complete rollback.
The optimizer is a separate layer; standalone geometry calls retain their default execution path.

`SurfaceDomain` supplies an independent, fixed-buffer triangle hierarchy for
boundary geometry. It refits triangle bounds in pure Warp and provides float64
closest-point and oriented-ray queries, manifold/intersection validation, and
surface/crease/corner quadrature. `DomainQueryWorkspace` composes these queries
with upstream and downstream device kernels. See [domain API](domains.md).

The independent [`BoundaryWorkspace`](boundary.md) rebuilds a site hierarchy,
assigns surface/crease quadrature, and accumulates weighted centroid targets.
Device conflict resolution reserves distinct corner sites; crease and surface
membership evolves with the positions. An internal VTM core combines these
targets with interior ODT, reconstruction, and complete rollback. Boundary-first
acceptance is a heuristic because fitting changes the optimization domain.
Unassigned interior sites additionally use a [material-domain trust region](vtm-domain.md),
and unassigned exterior sites move toward the surface, preventing missed boundary
samples from letting unconstrained ODT updates drive sites away from the shape.

[`ExtractionWorkspace`](extraction.md) independently selects tetrahedra using the
paper's circumcenter thickening rule. Fixed-buffer face/edge tables expose the
oriented boundary and check local manifoldness; device union operations measure
components, shell Euler characteristics, and cavity volumes. Quality and directed
surface-error reductions are available to downstream device processes.

The independent [`SizingField` and `SizingQueryWorkspace`](sizing.md) evaluate
sampled positive sizing constraints with controlled gradation. A device hierarchy
prunes cones using minimum sample values and distance bounds; refit and queries
compose in capture. The internal VTM core optionally uses centroid-evaluated
inverse-cubed sizing for interior contributions and shares the field with its
surface and crease quadrature. The uniform standalone ODT kernel is unchanged.

[`AutomaticSizingWorkspace`](sizing.md#automatic-feature-size-estimates) derives
sampled feature sizes from certified control-site Delaunay cells and finite
Voronoi poles. [`InitialSamplingWorkspace`](sizing.md#device-initial-sampling)
uses a conservative grid proposal and bounded rejection to generate exactly the
requested number of unique interior sites at uniform or inverse-cubed sizing
density. Both are separate lazy APIs with fixed storage and captured refits/runs.

The internal [refinement workspace](refinement.md) ranks material-interior
Voronoi centers by circumradius over sizing, certifies their
[float32 rounding](stable-centers.md), removes coincident positions, and appends
a bounded batch to the device active prefix. Reconstruction and
statistics either commit together or restore the prior state entirely on device.
It uses separate fixed buffers and adds no work to existing geometry calls.

Material extraction, retained-cell selection and bounded dual volumes use a
[wide geometry fallback](extraction-geometry.md) when a tetrahedron's float64
determinant is numerically uncertain. Integer original-coordinate determinants
and circumcenter numerators preserve very flat positive cells through metric
evaluation. Centers and local offsets remain float64; the existing
material-selection and topology checks decide which cells belong in the output.

The core's optional [selective mode](selective.md) classifies retained tetrahedra,
marks poor-element vertices, and expands one neighbor ring in separate device
passes. Masked updates and an additional retained-quality guard share the existing
reconstruction/backtracking mechanism; default kernels and graph work are unchanged.

Extraction, retained-cell classification and boundary-distance queries use
[32-thread launch blocks](query-launches.md) to distribute small active workloads
across the GPU. This changes scheduling without changing geometric calculations,
buffers or the standalone diagram and Delaunay paths.
VTM-owned facet completion and seed filtering, boundary projection and refinement
also use measured [32-thread launches](reconstruction-launches.md). Shared
reconstruction classes retain the standalone constructor's 256-thread defaults;
the private VTM class selects its policy before graph capture.

The optional [boundary-jitter workspace](jitter.md) searches a tangent disk at a
surface vertex of a poor tetrahedron. Fixed buffers hold the original state and
device-seeded proposals. Each trial is reconstructed and certified; retained
quality, boundary fitting, topology counts, and volume guards determine whether
it commits or restores the original mesh. Corners and crease sites stay fixed.
This separate stage adds no work to the standalone geometry or default VTM paths.

The optional [interior improvement workspace](interior-improvement.md) moves a
dense interior site to a poor interior tetrahedron's centroid, or inserts one
when the device target allows growth. A fresh full-hull face table identifies
bounded cells; signed circumcentric flags provide actual Voronoi volumes for
sizing-normalized donor selection. It shares the quality transaction and
restores the active prefix and all geometry if the candidate fails its guards.

The public [`VariationalMeshingWorkspace`](mesher.md) owns these stages and a
single reusable optimization core. Device budgets and conditional loops control
initialization, growth, fitting, extraction recovery, and quality rounds.
Input/output topology counts gate publication; optional quality-stage checkpoints
restore the last valid mesh on failure. The output counts and diagnostics stay
on device for downstream captured consumers. This lazy API adds no work to the
independent geometry routines.

Optional [containment recovery](containment-recovery.md) can restore omitted,
wholly material tetrahedra around nonmanifold boundary edges. Exact containment
witnesses gate additions, and every bounded pass reruns topology checks. The full
mesher enables this extension and carries its selection into quality guards;
the independently callable extractor retains its original default rule.

At capacities of 256–2,048 sites, the VTM core first attempts to
[reuse Delaunay fragments](delaunay-reuse.md) from the preceding reconstruction.
It checks them against current original-coordinate sites, completes surviving
facets with a four-round budget, and requires the full final certificate. The
ordinary constructor handles any failed attempt. Fresh meshing initialization
clears the fragment; standalone diagram and Delaunay APIs keep their existing paths.

For an active prefix in a larger reserved workspace, the core now
[selects a reconstruction capacity on device](active-reconstruction.md).
Preallocated 1,024- and 2,048-site constructors handle the smallest fitting prefix;
the ordinary full constructor handles larger inputs and prefix failures. Inactive
scratch sites are exact duplicates of site zero, so the prefix contains every
distinct site and retains original IDs. Each branch still certifies the complete
active convex hull. Stable output buffers let downstream kernels consume either
branch without allocation, host decisions or graph recapture.

Broader example validation, integrated time-to-quality comparisons and CPU
performance work remain tracked in the [full VTM implementation plan](vtm.md).
An energy decrease alone does not establish good worst-element quality or a
CPU performance advantage.
