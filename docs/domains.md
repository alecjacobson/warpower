# Device surface domains

`SurfaceDomain` supplies the boundary queries and quadrature needed by the
variational mesher. It is independently useful for closest-surface queries,
inside/outside tests, and surface integration. This is a domain service, not a
complete tetrahedral mesher.

The default `query_backend="reference"` retains the contracts below. For good,
well-scaled input, an optional [native Warp query backend](native-surface-queries.md)
uses FP32 surface queries. Its native refit has a narrower capture contract;
query and validation options are independent.

```python
import numpy as np
import warp as wp
from warpower import SurfaceDomain, DomainQueryWorkspace

# vertices: float32 (V, 3); triangles: int32 (F, 3), consistently oriented.
# Surface positions and query positions can instead be contiguous CUDA Warp
# arrays or CUDA PyTorch tensors. Existing device point buffers are shared.
domain = SurfaceDomain(vertices, triangles, surface_subdivisions=2)
queries = DomainQueryWorkspace(domain, query_points)
with wp.ScopedCapture() as capture:
    # An upstream process may update domain.points and queries.points here.
    domain.refit()  # Omit when the boundary has not changed.
    queries.run()
    # A downstream kernel can read queries.closest, signed_distance, inside,
    # face, valid, and error without host readback.
wp.capture_launch(capture.graph)
```

Setup allocates fixed buffers, copies triangle topology, builds a Morton-ordered triangle hierarchy on CUDA, compiles kernels, and
synchronizes. `refit()` and
`run()` then perform device work with stable storage and no allocations or host
geometry decisions. They accept an optional Warp `stream`. Call them in the same
stream as the producer and consumer, or establish CUDA event dependencies.
Separate concurrent loops need separate workspaces. Keep the domain alive while
using its buffers or its `tree` structure in custom Warp kernels.

Topology is fixed after setup. Changes to source triangle arrays are not observed;
construct a new domain for different connectivity. Point positions remain shared
for contiguous CUDA inputs; NumPy points are uploaded once. After changing them,
call `refit()` before querying. Ordinary diagram and Delaunay calls do not import
or execute this layer.

## Domain contract and errors

Input is a manifold, watertight, intersection-free triangle boundary. Face
normals point out of the material: outer shells face outward and cavity shells
face inward. Nonconvex shapes, cavities, and disconnected components are
supported. Unreferenced vertices are permitted. Duplicate indexed faces,
open/nonmanifold edges, inconsistent edge orientation, and disconnected vertex
links are rejected on device. Geometry validation checks triangle intersections,
including excess contact between triangles sharing an edge or vertex, and tests
which side of each face contains material.

The complete domain validator is not an exact-predicate certificate. The
hierarchy, closest-point regions, and oriented ray counts use pure Warp kernels
with float64 geometric arithmetic. Shared-edge/vertex ray hits trigger two
alternative directions on device; unresolved classification is reported rather
than guessed. After bounding-box filtering, intersections between triangles
without shared vertex indices use filtered float64 orientation signs with exact
dyadic-integer fallback. Plane separation and segment/triangle tests handle crossing,
coplanar overlap, and unindexed touching without a geometric tolerance. Shared
indexed contact retains the separate float64 edge/cone tests.

The exact-sign path fixes eight false intersection reports in the CGAL-distributed
fandisk and bunny surfaces ([#92](https://github.com/alecjacobson/warpower/issues/92)).
Warp 1.19's native triangle test uses a fixed plane threshold, including its
float64 overload, so casting coordinates alone does not fix the issue. Independent
CGAL exact checks found all eight pairs disjoint and both complete surfaces free
of self-intersections using the same float32 coordinates. Both surfaces now pass
the normal device validator; no geometry checks are disabled. The reduced
fixtures are also checked against an independent arbitrary-precision integer
separating-axis oracle across all windings, pair orders, and power-of-two scales.

Faces with twice-area at most `1e-12` times the sum of squared edge lengths are
rejected before division. Material-side orientation is checked at each face
centroid by excluding that face from the ray and accounting for its expected
crossing, avoiding an offset that could jump a thin gap. Tests include a box only
`2e-7` thick and triangles skipped by Warp's native float32 mesh queries. Use
well-scaled coordinates: position storage is float32, and no predicate can
recover detail already lost during input rounding. No automatic surface repair
is performed.

The triangle ordering is established once during setup. A refit updates every
triangle and hierarchy bound without sorting or allocating. Large deformations
remain valid but can make this fixed ordering less efficient. Bounds enclose the
complete triangles, not only their centroids. A spatial radix tree groups four
consecutive Morton-sorted triangles per leaf. Ray, contact and sampler-box
queries follow precomputed escape links without a stack; near-first closest
queries use 128 entries. Morton keys and sorted-index tie breaking bound radix
depth below 96 under the int32 capacity contract. A balanced segment tree supplies
exact range bounds in one additional refit kernel. See the
[implementation and measurements](surface-radix.md).

The device `domain.error[0]` is a bit mask:

| Bit | Meaning |
|---|---|
| 1 | Triangle index outside the vertex array |
| 2 | Nonfinite vertex position |
| 4 | Collapsed or numerically unresolved triangle |
| 8 | Boundary is not a consistently oriented closed 2-manifold |
| 16 | Triangle intersection beyond legal indexed contact |
| 32 | Incorrect or unresolved material-side orientation |

Invalid geometry produces zero quadrature weights and zero feature/corner counts.
Indices and nonfinite coordinates are sanitized before hierarchy access, so an
invalid index does not become an out-of-bounds device read. Invalid topology
persists for the lifetime of the domain; a bad deformation can recover after
valid positions are supplied and `refit()` is called again.

`validate_geometry=False` skips the intersection and material-orientation checks
for an already validated, trusted boundary. Index, topology, finite-coordinate,
and degenerate-face checks remain active. This choice also affects subsequent
refits; the caller is responsible for preserving a valid embedded surface.

## Query outputs

`DomainQueryWorkspace(domain, points, max_distance=1e30)` reserves one result
per input point. A signed count of oriented ray crossings distinguishes material
from outside, cavities, and gaps between disconnected components.

| Buffer | Meaning |
|---|---|
| `closest` | Closest surface position, float32 `vec3` |
| `normal` | Unit normal of the selected triangle |
| `signed_distance` | Negative inside material, positive outside |
| `inside` | Int32 material-membership flag |
| `face` | Triangle index, or -1 when invalid/missing |
| `valid` | Int32 result-validity flag |
| `error` | Domain error bits, plus 64 for a nonfinite query and 128 for unresolved classification |

Always check `valid` before consuming a result. A face must be strictly closer
than `max_distance`; a distance-limited miss has `valid=0`, `face=-1`, and distance
set to the requested limit. It is not an error and does not establish outside
membership. Classification exactly on the boundary is not defined. Tied nearest
faces may select either face; the selected point and normal belong to that face.

## Quadrature and geometric features

Surface integration uses three degree-two Gaussian nodes on each of `s*s`
uniform subtriangles, where `s=surface_subdivisions`. `surface_points` contains
`3*F*s*s` samples and `surface_weights` their areas in float64. The weights sum to
the boundary's area. Increasing `s` resolves changes in nearest-site assignment
within large input triangles; coarse samples can underresolve boundary fitting.

An edge is a feature when adjacent outward normals differ by more than
`feature_angle` degrees (default 45). The stable compact prefix of `feature_edges`
contains `feature_count[0]` vertex-index pairs. Each feature edge has two
Gauss nodes on each of `crease_subdivisions` segments (default 4).
`crease_points` and `crease_weights` reserve space for the worst case; only the
first `2*crease_subdivisions*feature_count[0]` samples are active, and all inactive
weights are zero. Active weights sum to total feature length.

A feature vertex is a corner when its feature degree differs from two, or its
two incident feature directions turn by more than `corner_angle` degrees
(default 45). `corner_ids` and `corner_points` have a compact prefix of
`corner_count[0]`, and `corner_mask` flags the original vertices. These are
geometric candidates for the boundary-fitting stage, not preassigned mesh sites.

Samples and compact feature ordering are reproducible for identical inputs.
Refitting updates positions, weights, features, corners, and all device counts.
The sizing weights required by Alliez et al. (`dA/mu^4` and `dL/mu^3`) are applied
by the fitting layer; these base quadrature weights contain area/length only.

## Exact-contact validation record

On the NVIDIA L40 with Warp 1.19.0, the contact change passes
[388 repository tests](../results/validation/vtm/domain-contact-tests.txt),
[three padded memory cases](../results/validation/vtm/domain-contact-memcheck.txt)
and [a captured refit with full allocation/lifetime tracking](../results/validation/vtm/domain-contact-lifetime.txt)
using the documented diagnostic allocator/stack settings. A
[native producer/refit/query/consumer trace](../results/validation/vtm/domain-contact-device-audit.json)
contains 33 kernel, two memset and two device-to-device copy graph nodes, with
no host callback, allocation/free or host-transfer nodes.

The complete 1,000-site torus solve takes 3,940.59 ms (median of three additional
replays), versus 3,936.47 ms in the preceding gallery run. Both timing windows
were unthrottled; the difference is approximately 0.1%.
[Points, roles and all canonical connectivity are identical](../results/validation/vtm/domain-contact-torus-equivalence.json),
and independent Delaunay/topology checks pass. This is an internal regression
control, not a CPU comparison.

The stricter intersection check has a measurable cost: a deforming
9,216-triangle domain plus 1,000 queries takes 28.007 → 30.160 ms (ten-replay
medians). Queries alone remain 17.481 → 17.482 ms. These
[before](../results/validation/vtm/domain-contact-domain-before.json)/[after](../results/validation/vtm/domain-contact-domain-after.json)
measurements include no CPU throttling. The
[other runtime modules and flagship README remain unchanged](../results/validation/vtm/domain-contact-scope.json).
