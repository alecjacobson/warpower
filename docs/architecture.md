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
Its [spatial radix topology](surface-radix.md) groups four Morton-sorted triangles
per leaf; device refits gather exact range bounds from a segment tree. Precomputed
escape links support stackless ray, contact and sampler-box traversal while
closest queries retain near-first child ordering.

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
witnesses gate additions, and every bounded pass reruns topology checks.
[Certified predicate filters](containment-filters.md) avoid integer work when
binary64 bounds prove the signs, retaining the exact fallback and selection.
The full
mesher enables this extension and carries its selection into quality guards;
the independently callable extractor retains its original default rule.
If bad edges remain after the additional-fitting budget, [exterior-band recovery](exterior-band-recovery.md) can
transactionally remove exterior-center tolerance cells. Referenced vertices,
feature-incident cells, a one-percent volume limit and final topology checks
guard this approximation; unsuccessful attempts restore the original selection.
The final [proven-exterior fallback](proven-exterior-recovery.md) runs only if
that policy still fails. One ordinary fitting transaction uses a midpoint-aware
boundary proposal, then a plane-separation and rational-centroid winding proof
can remove wholly exterior cells at edge or vertex contacts. Certified binary64
bounds filter the exact integer signs. [Hierarchy exclusion](exterior-proof-hierarchy.md)
uses current exact surface boxes and conservative rational-centroid slab bounds
to avoid unrelated triangles without rounding the centroid. All selection, rollback, and subsequent
quality checks remain in the captured device graph. This optional layer adds no
work or storage to standalone power-diagram or Delaunay construction.

At capacities of 256–2,048 sites, the VTM core first attempts to
[reuse Delaunay fragments](delaunay-reuse.md) from the preceding reconstruction.
It checks them against current original-coordinate sites, completes surviving
facets with a four-round budget, and requires the full final certificate. The
ordinary constructor handles any failed attempt. Fresh meshing initialization
clears the fragment; standalone diagram and Delaunay APIs keep their existing paths.
The private seed filter [prepares current sphere geometry once per cell](shared-seed-geometry.md)
and shares it across its query lanes. Every call refreshes these fixed setup
buffers before querying; ordinary construction retains its original default path.
The shared query now uses [point-specific Ball64 bounds](prepared-point-queries.md)
for both strict physical signs. Uncertain and tied queries keep the original
symbolic fallback. The private VTM checker now [groups current Morton-sorted
points](grouped-seed-queries.md) in batches of 32, with freshly computed exact
group boxes. Strict sphere-box separation can skip an entire group; unresolved
groups retain the same individual point tests. This adds fixed group storage and
one preparation launch while keeping standalone construction paths unchanged.
At capacities 4,097–16,384, the VTM filter instead [traverses the existing
current segment hierarchy](hierarchical-seed-queries.md) down to those 32-site
groups. This skips excluded subtrees and removes the separate group-box buffers
and preparation launch, while retaining the same point predicates and certificate.
On these larger routes, [strict and exact queries run separately](cached-exact-seed-queries.md).
Each lane caches one unresolved point from the current invocation; overflow
retains the complete rescan. Fixed setup storage and device-only dispatch preserve
the existing exact policy and standalone API paths.
Within one unchanged-input completion, [exact cell proofs](completion-proofs.md)
now follow retained cells through deduplication and compaction. New cells keep
their full sphere checks; all topology, coverage and hull-support checks remain.
Proofs are re-established against current input on the next invocation.
On the larger private routes, [new-cell sphere geometry](prepared-new-cell-queries.md)
is prepared once and queried through the current segment hierarchy. Consumed seed
Sphere/Box arrays supply the scratch, with no extra device storage; the original
point predicates and query/error/proof-counter definitions remain unchanged.
The private hull-support pass also proves [exact coordinate-plane faces](axis-hull-support.md)
directly against current original-site bounds. A freshly certified positive cell
supplies the interior direction; all other faces retain the existing traversal.
This adds no buffers or graph nodes and leaves standalone construction unchanged.
The remaining traversal uses [certified maximum-corner box signs](bounded-hull-box.md):
the existing normal intervals select the plane's exact maximizing box corner,
then the bounded orientation filter certifies containment or avoids a redundant
interval test. Uncertain signs retain the original box test and exact site checks;
a box corner outside the halfspace does not itself reject face support.
A private [fresh complete-mesh certificate experiment](previous-complete-certificate.md)
tries to avoid fragment filtering when the entire previous topology survives.
Its ordinary schedule is slower; earlier rejection is effectively neutral at 4k
and slower at 1k. Both variants remain unadopted.
A further [bounded local-flip experiment](local-flip-experiments.md) repairs six
saved torus/Armadillo inputs with strict 2→3/3→2 cavities and full final
certification. Both complete VTM schedules regress performance and remain
unadopted; retained coplanar hand cavities motivate a separate 4→4 investigation.
After an absent or unsuccessful fragment, these private routes now try
[shared wall reconstruction](shared-wall-reconstruction.md). The bounded
[Local DeWall candidates](local-dewall-candidates.md) borrow the consumed fragment
buffer and share its exact seed filter and completion workspace. Acceptance
publishes the current certificate and matching face/hull metadata together,
avoiding a second certificate; any failure uses ordinary construction. The
new per-site wall state adds about 0.6 MB to the 4k torus workspace. Independently
callable diagram and Delaunay APIs retain their source and defaults.
The immediately preceding [hull-support result](hull-support-reuse.md) also
lets proven hull faces skip candidate search. Each new boundary ordering receives
a fresh global support check before that mask can be consumed. The private
check now [stops a face traversal after an exact exterior witness](hull-witness-stop.md),
sharing that negative result across partitions while retaining every later
strided face slot and exhaustive checking for positive support.
Its leaf point queries now use the existing
[bounded and analytic orientation filters](bounded-hull-support.md) before the
original interval/exact fallback. Tree pruning, support lifetimes, storage and
launches retain their existing behavior.

The VTM strided face scan and retained-cell queries first try
[certified bounded float32 filters](bounded-predicate-filters.md) on finite
original coordinates with magnitude at most 16. Their error bounds cover
roundoff, underflow and the rounded acceptance threshold. Unsupported or
uncertain face queries retain the [analytic float64 bounds](analytic-predicate-bounds.md)
and original interval/exact fallback; retained-cell queries keep their prepared
Ball64 and symbolic fallback. Physical ties retain the original symbolic policy.
This adds no persistent arrays or launches. Standalone predicates, topology
checks and proof lifetimes remain unchanged.

The selective optimizer [reuses an accepted material evaluation](current-material-selection.md)
at the immediately following step. The current keep mask, quality, minimum and
count already describe that unchanged mesh, including material recovery when
enabled. One device flag permits this private intra-run reuse. Fresh solves,
trial proposals, rollback, select-all and direct evaluation invalidate it;
every candidate quality check remains fresh. Standalone power-diagram and
Delaunay routines allocate no cache state and retain their existing paths.
For recovered material selection, a private extractor now
[checks surface-query validity with a finite-distance witness](selection-surface-status.md)
instead of computing unused boundary-distance summaries. An unavailable witness
uses the original full query. Geometry, recovery, topology and all error checks
remain; public extraction and final mesher diagnostics keep full distances.
Private recovery also [omits nearest traversal for proved inside circumcenters](selection-interior-status.md).
Its distance buffer provides only a nonpositive/positive sign for exterior-band
decisions. Outside, ambiguous and unsupported centers keep the complete query;
physical geometry/quality and all public distances remain unchanged.

A separate [cooperative surface-query experiment](cooperative-surface-probe.md)
measures shared ray traversal and fused/staged retained-cell classifiers. These
are private prototypes; their kernels and scratch are not used by production.
The nearest-surface traversal now [reuses child box distances](surface-bound-reuse.md)
within each query, preserving its original ordering, comparisons and triangle
arithmetic. A local stack holds the already computed float64 values; nothing is
retained across queries or refits. The
[operation counts and bounded-arithmetic probes](surface-bound-investigation.md)
explain this choice and record the separate experimental float32 filter.

The complete mesher also [reuses exact repeated surface queries](surface-query-cache.md)
in separate private consumer caches. Full center bits, nearest-cutoff bits and
a device surface generation must match; refits advance the generation, and
saturation permanently disables reuse. Current geometry, quality, thresholds
and validation still run. Preallocated snapshots and table construction remain
inside captured execution. Standalone diagram, Delaunay, extraction and core
VTM defaults retain their ordinary paths. The report records the memory cost,
clean speed comparisons and explicitly investigated baseline variability in
one hand's material triangulation.

A separate [boundary owner-seeding probe](boundary-owner-seed.md) tries a
previous owner's current distance as an initial nearest-site search bound.
It remains unadopted because its complete hand solve regresses and its strict
material comparison fails; production does not use the prototype.

A [grouped boundary reduction probe](boundary-global-reduce.md) combines energy,
weight and maximum-distance updates within existing Warp blocks. It also remains
unadopted: complete torus and hand timings show no benefit. Production keeps its
original accumulation order and kernel.

For an active prefix in a larger reserved workspace, the core now
[selects a reconstruction capacity on device](active-reconstruction.md).
Preallocated 1,024/2,048/4,096/8,192-site constructors below the reservation
handle the smallest fitting prefix; the full constructor handles larger inputs
and prefix failures. The [larger-prefix report](larger-active-prefixes.md) measures
the speed gains and additional persistent storage/setup cost. At reserved
capacities of 2,049–16,384 sites, that full route now
[uses certified fragment reuse](extended-fragment-reuse.md), retaining current-input
cell proofs throughout completion. Larger capacities retain ordinary full
construction. Each fresh solve clears all fragments. Inactive
scratch sites are exact duplicates of site zero, so the prefix contains every
distinct site and retains original IDs. Each branch still certifies the complete
active convex hull. Stable output buffers let downstream kernels consume either
branch without allocation, host decisions or graph recapture.

After ordinary extraction recovery fails, the full mesher can use a
[bounded tangential quality transaction](tangential-topology-recovery.md) at a
surface vertex identified from a rejected pinched-link boundary. Each of at most
32 proposals certifies the current complete hull and retains quality, topology,
reference and feature-role guards. Exhaustion restores points and reconstructs
private certificates. This scoped policy resolves the measured uniform/graded
8K Armadillo failures; standalone diagram/Delaunay paths and allocations are
unchanged. The report includes its additional setup/storage cost and bounded
failure semantics. A subsequent [constructor-only reduction](recovery-query-storage.md)
omits recovery's unused approximate-query cache and scratch while retaining the
candidate extractor's active cache, ordinary quality-move caches and checkpoints.
[Spatial ordering of large surface-query batches](ordered-surface-queries.md)
reduces measured full-solve replay time by 10.0% on the 16K torus and 5.7% on
the graded 8K scan, with unchanged standalone geometry paths.

The optional [native surface-query backend](native-surface-queries.md) replaces
VTM's surface distance and selection queries with Warp FP32 mesh queries while
retaining FP64 tetrahedron geometry and certification. The reference backend
remains the default. Native queries support nested captured meshing against a
fixed surface; native refit currently requires a top-level capture because Warp's
scan introduces device allocation nodes.

The [example gallery](examples.md) includes uniform/graded torus and scanned
Armadillo meshes, thin geometry, cavities, disconnected domains and a
[sharp-feature mechanical plate](mechanical-plate.md). Integrated time-to-quality
comparisons and CPU performance work remain tracked in the
[full VTM implementation plan](vtm.md).
An energy decrease alone does not establish good worst-element quality or a
CPU performance advantage.

An unadopted [representable-boundary experiment](representable-boundary-probe.md)
removes exact midpoint rounding stalls through ordinary accepted proposals.
Fresh hand solves expose a quality/fit tradeoff: the selective minimum guard
then reduces the shared step 1,024-fold. The recorded cell/step investigation
motivates localized backtracking; production behavior remains unchanged.

Optional [local quality backtracking](local-quality-backtracking.md) now provides
bounded partial rollback after fresh selective quality rejections. Every retry
retains complete reconstruction/certification and all acceptance checks. The
default zero keeps existing kernels and storage; standalone geometry APIs have
no dependency on this helper.

The complete mesher also uses [bounded selective step history](selective-step-history.md)
after quality-only rejections. It tries twice the previous accepted scale and
retries the full original bracket on non-quality failure, exhaustion or apparent
convergence. History resets within the captured graph at each core invocation;
44 device bytes hold state and counters. The measured 16K torus solve improves
from 4.39 to 2.91 seconds with identical final geometry. Standalone VTM cores,
power diagrams and Delaunay workspaces retain their existing defaults.
