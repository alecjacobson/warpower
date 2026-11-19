# Public API reference

This is the consolidated reference for every symbol exported by
`warpower.__all__`. Import these names from `warpower`. Underscore-prefixed
modules, constructor arguments, and attributes are implementation details.
There are 17 primary exports and four [deprecated compatibility names](#deprecated-names).
Detailed guides and recorded experiments are indexed in the
[documentation guide](README.md).

## Choose an entry point

| Need | Public API |
|---|---|
| Bounded Voronoi or power neighbors, ordinary call | [`voronoi_diagram`, `power_diagram`, `Diagram`](#diagrams) |
| Convex-hull Delaunay or regular tetrahedra, ordinary call | [`delaunay_tetrahedra`, `regular_tetrahedra`](#tetrahedron-convenience-functions) |
| Repeated bounded neighbors and/or tetrahedra in a GPU loop | [`UpdatableDiagram`](#updatablediagram) |
| Certified unweighted Delaunay construction with degeneracy recovery | [`UpdatableDelaunay`](#updatabledelaunay) |
| Triangle boundary, closest points and inside/outside queries | [`SurfaceDomain`, `DomainQueryWorkspace`](#surfacedomain) |
| Relative element sizes from samples | [`SizingField`, `SizingQueryWorkspace`](#sizingfield) |
| Pole-based automatic sizing | [`AutomaticSizingWorkspace`](#automaticsizingworkspace) |
| Initial interior sites | [`InitialSamplingWorkspace`](#initialsamplingworkspace) |
| Boundary and sharp-feature fitting | [`BoundaryFitter`](#boundaryfitter) |
| Uniform optimization with the initial convex hull fixed | [`ODTWorkspace`](#odtworkspace) |
| Select and inspect a material tetrahedral subcomplex | [`ExtractionWorkspace`](#extractionworkspace) |
| Initialize, grow, fit and extract a boundary-fitted mesh | [`VariationalMesher`](#variationalmesher) |

Bounded neighbors and convex-hull tetrahedra describe different domains.
Neighbors are dual to Delaunay edges, but clipping Voronoi cells can remove a
dual facet outside the clipping box. Do not derive complete convex-hull
connectivity from bounded neighbor CSR. Material tetrahedra are a selected
subcomplex approximating a supplied triangle domain.

## Updating geometry

`UpdatableDiagram` and `UpdatableDelaunay` retain their input arrays and
output storage so the same object can refresh geometry after positions or power
weights change:

```python
import warpower

# points and weights are contiguous CUDA arrays.
diagram = warpower.UpdatableDiagram(points, weights)
diagram.update()
# An upstream kernel can modify points or weights in place.
diagram.update()
```

Each `update()` reconstructs from the current inputs, reusing allocated
storage and prepared execution. The name does not promise incremental neighbor
repair. Input shapes and addresses remain fixed; changing the site count
requires a new object. `run()` remains a supported alias for `update()`
on these two classes. Other services retain their documented `run()`,
`refit()` and `apply()` methods.

## Common contracts

The implementation requires an NVIDIA CUDA GPU. Points are float32: a
one-dimensional CUDA Warp `wp.vec3` array, a CUDA PyTorch tensor of shape
`(N, 3)`, or a NumPy array of that shape. NumPy points upload to `cuda:0`.
Unless a parameter says otherwise, arrays belonging to one operation must use
the same device. Geometry must be finite. Connectivity and counts use int32.
These discrete geometry APIs do not provide automatic differentiation.

Convenience functions allocate and synchronize to return exact-size outputs.
They copy noncontiguous inputs as needed, return PyTorch tensors for PyTorch
points and Warp arrays otherwise, and honor the current PyTorch CUDA stream
when called with PyTorch points.

Workspaces require contiguous device inputs and retain their storage, so
producers can update it in place. NumPy inputs upload once; later NumPy edits
are not observed. Workspace outputs are always Warp arrays. Constructors may
allocate, compile, synchronize, and prepare captured graphs. The operations
`update(stream=None)`, `run(stream=None)` and `refit(stream=None)` enqueue device work and return
the object; `BoundaryFitter.apply(...)` does likewise. Operations can be
composed with Warp capture. Conditional graphs require CUDA 12.4+ support from
the Warp build and driver; use `wp.ScopedCapture` when conditional nodes are
involved. The native surface-refit exception is described [below](#surfacedomain).

Keep workspaces, their domains/fields, and shared inputs alive while using their
graphs or outputs. Serialize runs of the same workspace and order producers and
consumers using the same stream or explicit CUDA events. An optional `stream`
must be a Warp stream on the input device. Updating an existing array is
supported; replacing a Python attribute does not rebind an already captured graph.
Changing sizes, capacities, topology or setup options requires a new object and
recapturing any enclosing graph.

Workspace array shapes are reserves, not valid lengths. Check the documented
error/valid flag and device count before reading each output prefix. Unused
storage may be uninitialized or stale. For a host snapshot outside capture,
read the count first and copy only the valid device slice:

```python
# mesh is an UpdatableDiagram or UpdatableDelaunay that has already updated.
if int(mesh.error.numpy()[0]) != 0:
    raise ValueError("Tetrahedralization failed")
count = int(mesh.tetrahedra_count.numpy()[0])
tets = mesh.tetrahedra[:count].numpy()
```

Host readbacks synchronize and belong outside captured consumers. Inside a
graph, launch over capacity and guard each access on the device error and count.
See [graph execution](graphs.md) for a consumer kernel and stream examples.

## Diagrams

```python
voronoi_diagram(points, *, return_tetrahedra=False, **kwargs)
power_diagram(points, weights, *, return_tetrahedra=False, **kwargs)
Diagram(adjacency, offsets, status, diagnostics=..., tetrahedra=None)
```

`weights` are additive power weights, float32 with shape `(N,)` or
`(N, 1)`; NumPy weights upload to the points device. Power distance is squared
Euclidean distance minus weight. Zero weights give an ordinary Voronoi diagram.
The bounding domain is the input AABB expanded by `1.00001` in each direction.
Exactly coincident sites have one owner: greatest weight, then lowest input
index. Other coincident sites have empty cells and never occur as neighbors.

Supported public `kwargs`:

| Argument | Default | Contract |
|---|---:|---|
| `knn` | `8` | Nonnegative compatibility option; effective KNN count is zero, with nearest-leaf initialization |
| `debug` | `False` | Print elapsed call time and diagnostics |
| `initial_guesses` | `None` | Flat int32 CSR hints; valid site indices or `-1` to skip |
| `initial_guesses_offsets` | `None` | Int32 `(N+1,)` offsets partitioning the entire hint array; supply both hint arrays together |
| `leaf_size` | `-1` | `-1` selects 4; otherwise a positive integer |
| `chunk_size` | `None` | Positive maximum concurrent-cell count, or automatic scratch sizing |

Hints affect clipping order; the complete hierarchy traversal still runs.
Automatic chunking recognizes `WARPOWER_NUM_CHUNKS` and
`WARPOWER_MEM_FRAC` (default `0.5`, strictly between zero and one), with
`PARAGRAM_*` aliases. Chunking bounds temporary clipping storage; inputs,
hierarchy and outputs remain resident.

`Diagram` is a frozen dataclass holding the following fields; freezing the
container does not make its arrays immutable. Omitted `diagnostics` creates a
new empty dictionary (the `...` above denotes its default factory).

| Field | Meaning |
|---|---|
| `adjacency` | Int32 flat symmetric neighbor CSR, using original site indices |
| `offsets` | Int32 `(N+1,)`; neighbors of site `i` occupy `[offsets[i], offsets[i+1])` |
| `status` | Int32 `(N,)` cell outcomes |
| `diagnostics` | Host dictionary with construction and retry statistics |
| `tetrahedra` | `None`, or the optional int32 `(T,4)` output described below |

Row order is unspecified. Empty inputs return empty adjacency/status and
`offsets=[0]`. Failed cells publish no partial neighbor row; inspect statuses
before treating a result as a complete diagram.

| Cell status | Meaning |
|---:|---|
| 0 | Construction succeeded |
| 1 | Plane or neighbor-output capacity exhausted |
| 2 | Vertex capacity exhausted |
| 3 | Empty or non-owning coincident cell |
| 4 | Traversal queue capacity exhausted |
| 5 | Inconsistent floating-point clipping boundary |
| 6 | Uncertain clipping decision awaiting precision recovery |

Clipping capacity, traversal queue capacity and numerical recovery are separate
requirements. Capacity overflow alone does not require FP64. Numerical recovery
retains the storage already reached and cannot return to FP32 on later overflow.
FP32 retries use 32 cooperating threads for 128/256/512-slot cells. Eligible
numerically difficult cells get at most one bounded cooperative FP64 attempt,
then serial FP64 fallback when needed. One-sided facets trigger consistency
recovery before symmetrization. These screens and floating-point constructions
do **not** certify an exact diagram, even when every status is successful.

Diagnostic counters have distinct scopes:

| Key | Meaning |
|---|---|
| `cooperative_precision_attempts` | Bounded cooperative FP64 attempts, at most one per site; excludes cooperative FP32 retries |
| `retried_cells` | Cells entering subsequent host-dispatched retries |
| `double_precision_cells` | Cells queued for the later serial FP64 phase, not all cells using FP64 |
| `consistency_retried_cells` | Consistency-recovery jobs, including repeated jobs |
| `asymmetric_cells_remaining` | Unresolved consistency after up to three repair passes |
| `knn_requested`, `knn_effective` | Requested compatibility setting and effective value zero |

For stage-level investigation, run
`python -m benchmarks.replay.retry_diagnostics input.npz retry-stages.json`.
The input contains `points`; `--weighted` also reads `weights`. This
synchronous diagnostic belongs outside timing and graph capture.
See [construction and recovery](architecture.md) and [correctness evidence](numerical-audit.md).

<a id="optional-full-tetrahedral-mesh"></a>

### Optional tetrahedron output

`return_tetrahedra=True` adds positively oriented tetrahedra covering the
complete convex hull, with original input indices and unspecified row order.
Positive orientation means `dot(p1-p0, cross(p2-p0, p3-p0)) > 0`.
Power weights select the regular tetrahedralization; globally hidden weighted
sites are absent. Coincident ownership is the same as for the diagram.

This performs a separate FP64 construction with adaptive domain expansion.
The CSR still describes the original bounded diagram. GPU checks cover
orientation, empty power spheres, interior face pairing and supporting hull
faces. Diagnostics add `tetrahedra_count`, `tetrahedra_expansions`,
`tetrahedra_padding` and a description of validation.

This builder uses floating-point predicates without symbolic tie breaking.
Unresolved cospherical/copower cases (including the eight cube corners),
near-coplanar cells, unsupported lower-dimensional input, and exhausted
capacity/expansion budgets raise `ValueError`; no partial tetrahedralization
is returned as complete. Fewer than four sites or all-coincident sites return
an empty `(0,4)` array. For certified unweighted recovery, use
[`UpdatableDelaunay`](#updatabledelaunay).

The default `False` allocates and runs no tetrahedron construction.

## Tetrahedron convenience functions

```python
delaunay_tetrahedra(points, *, chunk_size=None, leaf_size=4)
regular_tetrahedra(points, weights, *, chunk_size=None, leaf_size=4)
```

Return exact-size int32 `(T,4)` arrays with the same input types, orientation,
ownership, convex-hull coverage and numerical limits as
[optional tetrahedron output](#optional-tetrahedron-output), while skipping
bounded adjacency. `regular_tetrahedra` requires weights. `chunk_size`
and `leaf_size` follow the diagram rules. These allocate and synchronize;
use a workspace for capture.

<a id="diagramworkspace"></a>

## UpdatableDiagram

```python
UpdatableDiagram(
    points, weights=None, *, return_tetrahedra=False, adjacency=True,
    adjacency_capacity=None, tetrahedra_capacity=None,
    tetrahedra_cell_capacity=512, chunk_size=None, leaf_size=4,
    max_expansions=12,
)
diagram.update(stream=None)
```

Reconstruct the requested bounded CSR and/or convex-hull tetrahedra from the
current fixed-size input on every run. At least one output must be requested.
Weights follow the diagram contract. This is fresh construction, with a cached
successful tetrahedron box scale; it is not incremental neighbor reuse.

| Option | Contract |
|---|---|
| `adjacency_capacity` | Default `max(1,48*N)` directed entries |
| `tetrahedra_capacity` | Default `max(16,8*N)` output/candidate tetrahedra |
| `tetrahedra_cell_capacity` | 512 or 1024 clipping slots per cell; 1024 requires tetrahedron output |
| `chunk_size` | Positive concurrent-cell limit or automatic choice; smaller values reduce scratch |
| `leaf_size` | Positive hierarchy leaf size |
| `max_expansions` | Positive bound on tetrahedron-domain expansion attempts |

Reserves are practical defaults, not theoretical output bounds.
`max(adjacency_capacity, 8*tetrahedra_capacity, 512*N) < 2**31` is required.
Increase an exhausted reserve by constructing a new workspace outside capture.

Outputs are `adjacency`, `offsets`, `status`,
`adjacency_count`, `tetrahedra`, `tetrahedra_count` and `error`.
Counts and error are device int32 `(1,)` arrays. When requested,
`tetrahedra_expansions` records the current expansion count. Unrequested
outputs have no result contract; `status` is initialized to -1 when adjacency
is disabled. On any error, both published counts and all CSR offsets are zero.
The floating-point tetrahedron limitations above still apply.

| Error bit | Meaning |
|---:|---|
| 1 | Nonfinite input |
| 2 | Unresolved cell capacity or numerical failure |
| 4 | Adjacency reserve or CSR indexing overflow |
| 8 | Tetrahedron candidate reserve exhausted |
| 16 | Ambiguous tetrahedron construction |
| 32 | Sphere or topology validation failed |
| 64 | Domain expansion budget exhausted |

See [graph execution](graphs.md) for capture, memory and stream details.

<a id="delaunayworkspace"></a>

## UpdatableDelaunay

```python
UpdatableDelaunay(
    points, *, tetrahedra_capacity=None, tetrahedra_cell_capacity=1024,
    chunk_size=None, max_attempts=6, cache_strategy=True,
    fast_candidates=True, symbolic_recovery=True,
    symbolic_max_expansions=16, facet_recovery=None, facet_max_rounds=64,
)
mesh.update(stream=None)
```

Construct and certify an unweighted Delaunay tetrahedralization against the
original float32 coordinates without moving input sites. Requires at least
four sites and a three-dimensional distinct-site set for success. Coincident
positions, including signed zeros, use the lowest index. Exact predicates and
deterministic symbolic lifting resolve ties. Every successful result covers the
convex hull and uses every distinct site. Candidate generation and recovery
have finite budgets, so this is not a guarantee of success for every input.
Additive weights are unsupported by this API.

Capacity and chunk options configure the underlying geometry workspace.
`max_attempts` is 1–12 auxiliary candidate attempts.
`cache_strategy=True` tries the last successful strategy first, but always
certifies against current coordinates. `fast_candidates=True` enables the
candidate path for sufficiently large inputs (32,768 distinct sites).
`symbolic_recovery` enables original-coordinate symbolic cell recovery, with
1–32 `symbolic_max_expansions`. `facet_recovery=None` enables fragment
completion for 256–2,048-site buffers when symbolic recovery is enabled;
explicit booleans override it. `facet_max_rounds` is 0–64.

Consume `tetrahedra[:tetrahedra_count[0]]` only when `error[0]==0`.
`error` is a code: 0 success, 1 invalid input, 2 exhausted recovery; failure
zeros the count. `certificate_error`, `base_error`, `candidate_error`,
`symbolic_error`, `facet_error` and `fast_error` describe their respective
stages and may describe rejected candidates before eventual success.
`attempts`, `recovered`, `base_attempted`, `recovery_first`,
`auxiliary_level`, and the `symbolic_*`, `facet_*`, `fast_*` stage
counters identify the route used. Exact diagnostic masks and recovery policy are
in [certified Delaunay construction](delaunay-recovery.md).

## SurfaceDomain

```python
SurfaceDomain(
    points, triangles, *, surface_subdivisions=1, crease_subdivisions=4,
    feature_angle=45.0, corner_angle=45.0,
    validate_geometry=True, query_backend="reference",
)
domain.refit(stream=None)
```

Represent a watertight, consistently oriented, intersection-free triangle
boundary with at least four vertices and faces. Outer shells face outward from
material and cavity shells inward. Nonconvex/disconnected material and
unreferenced vertices are supported. `triangles` is contiguous int32
`(F,3)` Warp/PyTorch data on the points device, or NumPy data uploaded there.
Connectivity is copied and fixed; positions remain shared for CUDA inputs.
The constructor builds initial state. Call `refit()` after position edits
before any dependent query, sampler, fitter or extractor.

Subdivision counts are positive integers. Surface quadrature has
`3*F*surface_subdivisions**2` samples. Creases use two Gaussian nodes per
subdivision of each detected feature edge. Feature/corner angles are strictly
between 0 and 180 degrees. Refits update features, corners, sample positions and
weights in fixed storage. `surface_points`/`surface_weights` and
`crease_points`/`crease_weights` expose quadrature; inactive crease slots
have zero weight. `feature_count` and `corner_count` delimit their active
feature/corner lists. See [domain buffers](domains.md).

`validate_geometry=False` skips intersection and material-side orientation
validation; index, finite-position, collapsed-face and manifold checks remain.
The complete domain validator is not an exact-predicate certificate.
`error` is a device bitmask: 1 invalid index, 2 nonfinite position, 4 collapsed
or unresolved triangle, 8 nonoriented/nonclosed manifold topology, 16 triangle
intersection, 32 wrong or unresolved material-side orientation. Invalid domains
publish zero quadrature weights and feature counts. Geometry edits can recover
on refit; fixed invalid topology requires constructing a new domain.

`query_backend="reference"` is the default. The opt-in `"native"` backend
uses Warp FP32 surface queries with stricter usable-triangle limits. Its
`refit()` can run in a top-level capture but cannot run inside a conditional
subgraph, because Warp 1.19 adds allocation/free nodes. Queries and meshing
against a fixed native surface remain capturable. Read the
[native backend contract](native-surface-queries.md) before selecting it.

## DomainQueryWorkspace

```python
DomainQueryWorkspace(domain, points, *, max_distance=1e30)
queries.run(stream=None)
```

Query current positions against a previously refitted `SurfaceDomain`.
`max_distance` must be finite and in `(0,1e30]`; a hit must be strictly
closer. Outputs per query are `closest`, outward `normal`,
`signed_distance` (negative in material), `inside`, `face`, and `valid`.
A distance-limited miss has `valid=0`, `face=-1`, and distance equal to the
limit; it is not an error or proof of being outside. Boundary membership is
undefined, and equal-distance ties may select either triangle.

Scalar `error` includes domain error bits plus 64 for a nonfinite query and
128 for unresolved classification. Check both global error and per-query
`valid` before consuming geometric results. See [surface queries](domains.md).

## SizingField

```python
SizingField(points, values, *, gradation=1.0, minimum=1e-6, maximum=1e30)
field.refit(stream=None)
```

Define positive relative edge size
`mu(x) = clamp(min_i(values[i] + gradation*distance(x,points[i])), minimum, maximum)`.
Samples are upper constraints, not interpolation values or power weights.
`values` must be a contiguous CUDA Warp float32 array with one entry per
sample, on the points device. There must be at least one sample and fewer than
`2**29`. Values must be finite and positive.
`gradation` is finite in `[0,1e30]`; zero gives a constant field.
Clamps satisfy `1e-30 <= minimum <= maximum <= 1e30`.

The constructor builds the field. After editing positions or values, call
`refit()`; consumers do not refit it implicitly. Device `error` uses bit
1 for nonfinite sample positions and bit 2 for invalid values.
See [sizing and sampling](sizing.md).

## SizingQueryWorkspace

```python
SizingQueryWorkspace(field, points)
queries.run(stream=None)
```

Evaluate a previously refitted `SizingField` at nonempty current query points
on its device. Output `values` is float32, one per query. Scalar `error`
uses bit 1 for an invalid field and bit 2 for a nonfinite query; affected
outputs are zero and must not be used as positive sizes.

## AutomaticSizingWorkspace

```python
AutomaticSizingWorkspace(
    domain, *, control_subdivisions=1, gradation=1.0,
    minimum=1e-6, maximum=1e30, tetrahedra_capacity=None,
    tetrahedra_cell_capacity=1024, chunk_size=None,
)
provider.refit(stream=None)
```

Estimate local feature size from poles of the domain's control-site Delaunay
mesh and expose a usable `field: SizingField`. Control sites are the original
boundary vertices plus `3*F*control_subdivisions**2` surface samples.
`control_subdivisions` is nonnegative; zero uses only vertices. The control
count must be less than `2**28`. Sizing parameters follow `SizingField`;
capacity options configure the control-site Delaunay builder.

The constructor initializes the provider. After changing the boundary, call
`domain.refit()` then `provider.refit()`. This is a sampled pole estimate,
not an exact medial axis or a guarantee of adequate feature resolution.
The independent provider defaults to one subdivision; the mesher-owned provider
defaults to zero.

Inspect `control_points`, FP64 `local_feature_size`, `poles`,
`pole_valid`, `pole_source_tets`, `pole_count`, and `mesh`.
There are two pole slots per control site; use `pole_valid`, not a prefix
of length `pole_count`. Error bits are 1 invalid domain, 2 control mesh
failure, 4 pole geometry/capacity failure, 8 missing finite pole or invalid
distance. Failure invalidates the field and pole validity; child mesh storage
may reflect an earlier run. See [automatic sizing](sizing.md#automatic-feature-size-estimates).

## InitialSamplingWorkspace

```python
InitialSamplingWorkspace(
    domain, n, *, sizing=None, resolution=32, seed=2026, max_attempts=4096,
)
sampler.run(stream=None)
```

Generate `n` distinct float32 interior sites, uniformly or with density
proportional to `mu**-3` for a supplied `SizingField`.
`n` is in `[1,2**29)`, proposal-grid `resolution` in `[1,128]`,
and per-site `max_attempts` in `[1,1000000]`. `seed` is a signed int32
integer or contiguous CUDA Warp int32 `(1,)` array on the domain device.
The caller owns seed advancement. Identical inputs and seed repeat the sample
sequence. A finite rejection budget need not reach every small component.

The constructor prepares storage; call `run()` to sample. Refit changed
domain/field inputs first. Outputs are `points`, scalar `count`,
per-site `attempts`, `seed` and `error`. Success publishes `count=n`;
failure publishes zero, never a usable partial sample.
Error bits: 1 domain, 2 sizing, 4 proposal CDF, 8 exhausted attempts,
16 duplicate samples, 32 invalid sizing bound.
See [initial sampling](sizing.md).

<a id="boundaryworkspace"></a>

## BoundaryFitter

```python
BoundaryFitter(
    domain, points, *, active_count=None, surface_sizing=None,
    crease_sizing=None, query_precision="float64",
)
fitter.run(stream=None)
fitter.apply(*, damping=1.0, max_step=1e30, stream=None)
```

Compute sampled boundary-fitting targets for nonempty sites on the domain
device. `active_count` optionally selects a nonempty prefix using a contiguous
CUDA Warp int32 `(1,)` array. `surface_sizing` and `crease_sizing` are
positive float32 values per quadrature slot; contiguous Warp/PyTorch arrays
share storage, NumPy arrays upload, and `None` uses ones.
`query_precision` accepts `"float64"`, `"float32"` or `"mixed"`;
all retain FP64 accumulation. See [precision choices](boundary-query-precision.md)
and [mixed queries](mixed-boundary-queries.md).

`run()` computes `targets` and roles without moving sites.
Roles are 0 interior, 1 surface, 2 crease, 3 corner; `role_counts` has four
entries. Corner matching is distinct and deterministic; nearest-site ties use
the lowest active index. `apply()` moves surface/crease sites toward their
targets with `0 < damping <= 1` and `0 < max_step <= 1e30`, anchors corners
exactly, and leaves interior sites unchanged. These parameters must be finite.
Recompute targets after changing sites or domain geometry.

Additional outputs are `corner_sites`, `corner_owners`,
`surface_owners`, `crease_owners`, `surface_mass`, `crease_mass`,
`corner_rounds`, and two-entry `energy`, `total_weight`,
`max_distance` (surface then crease).
Error bits: 1 domain, 2 active count/nonfinite site, 4 corner matching,
8 sizing, 16 reduction/target. On failure targets revert to input positions,
roles/counts clear, and `apply()` leaves sites unchanged.
See [boundary fitting](boundary.md).

## ODTWorkspace

```python
ODTWorkspace(
    points, *, fixed=None, max_iterations=20, damping=0.5,
    tolerance=1e-4, step_limit=0.5, max_backtracks=12,
    tetrahedra_capacity=None, tetrahedra_cell_capacity=1024, chunk_size=None,
)
optimizer.run(stream=None)
```

Optimize uniform ODT energy with the initial convex-hull vertices fixed.
Requires at least four sites. Each run starts from current positions, discovers
their hull and resets counters/history. Shared CUDA sites are modified in place.
`fixed` optionally supplies a contiguous CUDA Warp int32 mask pinning
additional sites. This component uses the floating-point diagram tetrahedron
builder and retains its degeneracy limitations.

`max_iterations` and `max_backtracks` are nonnegative; zero iterations
only constructs/evaluates. `damping` and `step_limit` are finite and
positive; `tolerance` is finite and nonnegative. Displacement is capped by
`step_limit` times the shortest incident edge; rejected trials halve damping.
Stopping tolerance is relative to that edge. Capacity options configure
reconstruction.

Outputs are current `points`, `tetrahedra`, `tetrahedra_count`,
effective `fixed`, `energy`, `volume`, `minimum_radius_ratio`,
`iterations`, `rejected_steps`, `energy_history`, `termination`,
`error` and `last_geometry_error`. History contains initial energy plus
accepted steps, with remaining entries NaN. Termination is 0 iteration limit,
1 displacement convergence, 2 exhausted backtracking returning the last
accepted mesh. Error includes underlying construction bits, 256 for invalid
arithmetic and 1024 for hull-plane capacity. A rejected trial can leave a
nonzero `last_geometry_error` after successful recovery.

ODT energy decrease does not guarantee improved worst-element quality.
For grading and boundary fitting use the mesher below. See [ODT](odt.md).

## ExtractionWorkspace

```python
ExtractionWorkspace(
    domain, points, tetrahedra, tetrahedra_count=None, *,
    source_error=None, tolerance=0.4, containment_recovery=False,
    exterior_recovery=False, feature_roles=None, exterior_enabled=None,
    proven_exterior_recovery=False, proven_exterior_enabled=None,
)
extractor.run(stream=None)
```

Select a material subcomplex without changing the input mesh. Supply a
nonoverlapping, positively oriented tetrahedral complex; for the variational
method, supply the complete Delaunay triangulation. `tetrahedra` is
contiguous int32 `(capacity,4)`, on the points device (NumPy uploads).
Capacity must be in `[1,2**25)`. `tetrahedra_count` and `source_error`
are optional contiguous int32 `(1,)` arrays. These and the other extractor
integer inputs accept Warp/PyTorch arrays on the domain device, or NumPy arrays
uploaded at setup. Omitted count means the whole buffer is active; omitted
source error means zero.

Keep a cell when its circumcenter is in material or its outside boundary
distance divided by radius is strictly less than `tolerance` (finite,
`0 <= tolerance < 1`). This approximates the boundary and can retain cells
extending outside it. Extraction does not independently certify empty spheres
or arbitrary tetrahedron intersections.

Optional containment recovery adds proven wholly material cells; exterior
recovery transactionally removes exterior-center tolerance cells; proven
exterior recovery removes cells whose open interiors are proved exterior.
All default to `False` here. `feature_roles` is an optional device int32
site-role array for feature guards. `exterior_enabled` and
`proven_exterior_enabled` are optional device int32 `(1,)` runtime gates
for policies enabled at setup. See [selection and recovery](extraction.md).

Outputs include `tetrahedra`, `tetrahedra_count`, `source_indices`,
`boundary_faces`, `boundary_count`, `boundary_owners`, `radius_ratio`,
and per-input `selected`, `circumcenters`, `circumradii`,
`signed_distance`. Statistics include `volume`, `domain_volume`,
`volume_difference`, `minimum_radius_ratio`, `boundary_area`,
`surface_rms_distance`, and `surface_max_sample_distance`.
Topology outputs are `component_labels`, `component_count`,
`shell_labels`, `shell_count`, `cavity_count`, `shell_euler`,
`shell_volume`. Labels are root indices; shell values live at root indices.

Fatal `error` bits are 1 domain, 2 input count/upstream error, 4 index or
nonfinite position, 8 nonpositive determinant/invalid circumgeometry, 16 query.
Fatal errors zero both output counts. Separate `diagnostics` bits report
1 invalid face incidence/orientation, 2 invalid boundary-edge incidence/orientation,
4 invalid boundary-vertex link. Topology failures retain the subcomplex for
inspection. `valid=1` requires both masks to be zero. Empty output can be
valid; check the count if a nonempty mesh is required.
Validity does not certify equality with input topology, exact boundary
preservation, a minimum radius ratio, or a Hausdorff bound.

<a id="variationalmeshingworkspace"></a>

## VariationalMesher

```python
VariationalMesher(
    domain, capacity, *, target_count=None, initial_count=None,
    sizing=None, automatic_sizing=False, automatic_control_subdivisions=0,
    seed=2026, growth_iterations=5, polish_iterations=20,
    selective_iterations=10, max_growth_rounds=None, max_recovery_rounds=3,
    quality_rounds=4, refinement_fraction=0.5, quality_threshold=0.3,
    local_quality_retries=0, local_topology_retries=0, jitter_attempts=8,
    extraction_tolerance=0.4, containment_recovery=True, exterior_recovery=True,
    proven_exterior_recovery=None, tangential_recovery=None,
    sampling_resolution=32, sampling_max_attempts=4096,
    tetrahedra_capacity=None, tetrahedra_cell_capacity=1024, chunk_size=None,
    boundary_query_precision="float64",
)
mesher.run(stream=None)
```

Perform a complete fresh fit on every run: sample, grow, fit the boundary and
interior, improve quality, and extract material tetrahedra. A run does not
continue the previous fit. Refit a changed domain and any external sizing field
first; an owned automatic-sizing provider refits during the solve.

| Configuration | Meaning |
|---|---|
| `capacity` | Fixed site reserve, at least four; underlying geometry/index/memory limits also apply |
| `target_count` | `None` means capacity; otherwise integer or device int32 `(1,)` in `[4,capacity]` |
| `initial_count` | Minimum starting count, or an automatic minimum up to 256; raised for mandatory corners as described below |
| `sizing`, `automatic_sizing` | External `SizingField` or owned pole-based provider, mutually exclusive; omitting both selects uniform sizing |
| `automatic_control_subdivisions` | Nonnegative control-surface subdivision count; zero uses input vertices only |
| `seed`, `sampling_resolution`, `sampling_max_attempts` | Initial-sampler controls described above |
| `growth_iterations`, `polish_iterations` | Global fitting budgets before insertion batches and after reaching target/recovery |
| `refinement_fraction` | Maximum insertion batch / active count, in `(0,1]` |
| `max_growth_rounds` | Explicit budget or `ceil(log(capacity/initial)/log(1+fraction))+4` |
| `max_recovery_rounds` | Additional global fitting rounds after failed material extraction |
| `selective_iterations`, `quality_rounds` | Selective budget per round and maximum ordinary selective/jitter/interior rounds |
| `quality_threshold` | Radius-ratio threshold selecting poor elements, in `(0,1]` |
| `local_quality_retries` | Opt-in extra local quality rollback trials; requires positive quality rounds |
| `local_topology_retries` | Opt-in local topology rollback trials; requires quality rounds and material recovery, mutually exclusive with local quality retries |
| `jitter_attempts` | Boundary proposal budget per round, 1–1024 |
| `extraction_tolerance` | Circumcenter thickening tolerance in `[0,1)` |
| `containment_recovery`, `exterior_recovery` | Material-selection recovery policies, both enabled by default here |
| `proven_exterior_recovery` | `None` inherits exterior recovery; explicit boolean overrides |
| `tangential_recovery` | `None` inherits exterior recovery; enabling requires exterior recovery |
| `tetrahedra_capacity`, `tetrahedra_cell_capacity`, `chunk_size` | Reconstruction reserves, with 1024 clipping slots per cell by default |
| `boundary_query_precision` | `"float64"`, `"float32"` or `"mixed"` boundary assignment, retaining FP64 accumulation |

Iteration/retry budgets are nonnegative integers unless specified otherwise.
Each run resolves the initial count as
`min(target, max(minimum, corner_count + 4))`; an explicit minimum must be
at least four and no greater than the target. With no explicit minimum,
`min(256,target)` is used. `effective_initial_count` publishes the result.
Zero `quality_rounds` omits ordinary quality stages but does not disable
tangential topology recovery. Too few target sites for mandatory corners is an
explicit failure. The target counts sites, some of which may be unused by the
retained material tetrahedra.

| Published output | Valid prefix or meaning |
|---|---|
| `points` | `point_count[0]` sites |
| `full_tetrahedra` | `full_tetrahedra_count[0]` certified convex-hull tetrahedra |
| `tetrahedra` | `tetrahedra_count[0]` retained material tetrahedra |
| `boundary_faces` | `boundary_count[0]` outward material-boundary triangles |
| `radius_ratio` | `3*inradius/circumradius` per retained tetrahedron |
| `counts` | The four counts above, in that order |
| `topology` | Material components, boundary shells, cavities, aggregate shell Euler characteristic |
| `metrics` | Retained volume, domain volume, signed difference, minimum radius ratio, directed sampled boundary RMS |

Scalar views also expose `component_count`, `shell_count`,
`cavity_count`, `euler`, `volume`, `domain_volume`,
`volume_difference`, `minimum_radius_ratio`, `surface_rms_distance`.
Consume outputs only with `error==0` or `valid==1`. Failure zeros all
published counts/topology/metrics; buffers can still contain earlier data.
`active_count` is internal progress, not the published `point_count`.

`status` is 0 successful solve with ordinary quality stages disabled,
1 quality budget exhausted with progress, 2 quality stagnation, 3 error.
Statuses 1 and 2 still publish valid meshes. Error bits are 1 domain/input
topology/target/corners, 2 sizing, 4 sampling, 8 global fit/reconstruction,
16 fatal refinement/restoration, 32 growth budget, 64 failed nonempty manifold
extraction with matching topology, 128 failed quality-stage restoration.
Further diagnostics include `effective_initial_count`, `rounds`,
`steps`, `moves`, `refinement_rejections`, `quality_rejections`,
`last_errors`, `extraction_reason` and `last_bad_extraction`.

The boundary approximates the input; it does not preserve every input triangle.
Matching topology invariants is not a proof of homeomorphism. Validity does not
guarantee minimum element quality or a Hausdorff error bound. Read the
[mesher guide](mesher.md) for stopping, rollback and diagnostic details, and
[examples](examples.md) for complete inputs, exports and independent checks.

## Deprecated names

The following public names still resolve to the same classes and preserve
existing constructor signatures, output buffers and methods. Accessing them
through `warpower` emits `DeprecationWarning`; new code should use the
replacement names.

| Deprecated name | Replacement |
|---|---|
| `DiagramWorkspace` | `UpdatableDiagram` |
| `DelaunayWorkspace` | `UpdatableDelaunay` |
| `BoundaryWorkspace` | `BoundaryFitter` |
| `VariationalMeshingWorkspace` | `VariationalMesher` |

For diagram and Delaunay objects, prefer `update()`; existing `run()` calls
remain supported without a method-deprecation warning. The fitting classes
retain `run()`, and `BoundaryFitter` also retains `apply()`.
Historical experiment reports use the names available at their recorded
revisions.
