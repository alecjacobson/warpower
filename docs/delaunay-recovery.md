# Certified Delaunay construction

`DelaunayWorkspace` is an optional, independently callable ordinary Delaunay
constructor. It certifies connectivity against the original float32 sites and
recovers some configurations rejected by the existing floating-point builder,
including cube corners and regular lattices. Input positions are never moved.
The existing `delaunay_tetrahedra`, `regular_tetrahedra`, diagram functions, and
`DiagramWorkspace` retain their implementations and performance characteristics.

```python
import warp as wp
from warpower import DelaunayWorkspace

mesh = DelaunayWorkspace(sites, tetrahedra_cell_capacity=1024)
with wp.ScopedCapture() as capture:
    produce_sites(mesh.points)  # Your Warp launches, writing the shared buffer.
    mesh.run()
    consume_mesh(mesh.points, mesh.tetrahedra, mesh.tetrahedra_count, mesh.error)
wp.capture_launch(capture.graph)
```

Consumer kernels must check `error[0] == 0` and use only the prefix delimited by
`tetrahedra_count[0]`. These are device arrays. The example's producer/consumer
functions stand for application kernel launches, not host copies. Setup allocates,
compiles, captures, and synchronizes; `run(stream=None)` and captured replay use
fixed storage and device decisions. Establish stream dependencies when sharing
inputs with other processes. Warp scoped capture is required for conditional
nodes; see [graph execution](graphs.md).

Inputs are at least four sites in a contiguous CUDA Warp/PyTorch float32 `(N,3)`
buffer, or NumPy data uploaded once. The site count is fixed at setup. All active
positions must be finite. Coincident positions, including signed zeros, share
the lowest site index. A full-dimensional result uses every distinct site.
This API does not accept additive power weights; the independent regular-mesh
APIs remain available with their existing numerical contract.

## Construction and certification

The first attempt uses the existing unweighted full-hull constructor. Both its
successful outputs and every recovery candidate undergo the new certificate.
Recovery runs the regular builder on a normalized copy with small auxiliary
weights. The default starts at `1e-5 * h(i)`, for a fixed float32 hash direction
`h`, and divides the amplitude by ten on each failed attempt. There are six
attempts by default (`max_attempts` accepts 1 through 12). Neither normalization
roundoff nor finite auxiliary weights are assumed to preserve the desired mesh.
Only the original-coordinate certificate can accept a candidate.

The certificate uses [exact predicates](exact-predicates.md) with conservative
float64 filters. Every positive-volume candidate tetrahedron must have an empty
sphere against all original sites. BVH nodes are skipped only when interval
bounds prove a strictly negative determinant. Uncertain leaves use exact
expansion arithmetic. Exact zero-volume candidates are discarded; negative
original orientations reject the candidate.

Cospherical ties use **one consistent symbolic lifting**. First evaluate the
ordinary determinant. At exact zero, evaluate its coefficient for infinitesimal
weights `epsilon * h(i)`. The determinant is affine in the weights, so this is
another exact weighted determinant. Remaining exact ties use successively smaller
positive infinitesimal weights in site-index order, evaluated using orientation
cofactors. This chooses connectivity without perturbing the output coordinates.
Connectivity is deterministic for fixed indexed inputs; output row order is not
specified. A different site ordering or a different library's symbolic rule can
produce another valid Delaunay triangulation of the same cospherical sites.

Each accepted tetrahedron is a strict lower facet of that common symbolic lifted
point set, so these tetrahedra form a nonoverlapping subcomplex. A face hash
requires opposite orientations at each paired interior face. Every unpaired face
must exactly support the original convex hull. Any missing volume would expose
an unsupported interior face, so a nonempty accepted subcomplex covers the full
hull. Duplicate tetrahedra fail incidence/orientation checks. Canonical-site usage
is checked explicitly. This argument uses global lower-facet checks, rather than
assuming that a topologically plausible candidate is geometrically embedded.
For the lifting interpretation, see the
[CGAL triangulations manual](https://doc.cgal.org/6.0/Triangulation_3/index.html).

## Errors and limits

`error[0]` is zero on success, one for invalid original input, and two for
exhausted construction/certification attempts. Failure always publishes a zero
output count. Corrected input can succeed on the next replay. **Finite retries
are not a universal exact constructor**: small perturbations can still fail to
resolve construction, capacity, or expansion problems. Entirely coplanar input
has no volumetric result and reports an error.

`attempts` counts normalized recovery attempts; `recovered` is one when such an
attempt supplied the output. `base_error` and `candidate_error` expose the
underlying builders' codes. A zero candidate code with zero attempts means no
candidate construction was needed. `certificate_error` describes the last
certificate, using the following bit mask:

| Bit | Meaning |
|---|---|
| 1 | Invalid count, index, repeated canonical vertex, or nonfinite vertex |
| 2 | Exact predicate could not certify a sign |
| 4 | Negative original tetrahedron orientation |
| 8 | Not a strict lower facet under the symbolic lifting rule |
| 16 | Traversal stack capacity exceeded |
| 32 | Invalid face multiplicity or orientation |
| 64 | Exposed face does not support the convex hull |
| 128 | Empty mesh or missing distinct site |

The default tetrahedron capacity is `8*N`; the worst-case Delaunay size can be
quadratic, so applications may need `tetrahedra_capacity`. Cell capacity is 1024
by default here, while the older `DiagramWorkspace` default remains 512. Two
construction workspaces plus certificate buffers are reserved at setup. This
robustness path has measurable extra cost. Recovery and certification optimization
is tracked in [#56](https://github.com/alecjacobson/warpower/issues/56); it is not
a claim of performance parity with the original constructor.

The developing internal `VTMWorkspace` uses this certified constructor by default.
Its `certified_delaunay=False` option selects the earlier floating-point builder
for explicit comparisons. Fixed-hull `ODTWorkspace` remains unchanged. Boundary
and extraction behavior is documented [separately](boundary.md).

## Validation

Tests compare nondegenerate results against SciPy, complete hull volume against
an independent convex hull, and small degenerate outputs against rational
in-sphere determinants. They exercise lattices, cospherical points, duplicates,
scale/translation, corrupt candidates, failure recovery, and a changing-input
producer/mesher/consumer capture. Different valid cospherical triangulations are
compared geometrically rather than requiring SciPy's choice of diagonals.

```bash
python -m pytest -q tests/test_delaunay_recovery.py tests/test_filtered_predicates.py
python -m benchmarks.delaunay_recovery --case random --n 1000 \
  --repeats 10 --output results/local/recovery.json
```

The [full suite](../results/validation/vtm/recovery-tests.txt) passes **189 tests**.
The [installed-wheel check](../results/validation/vtm/recovery-wheel.json) runs a
captured producer/recovery/consumer loop, with installed sources matching the
checkout. The [isolation record](../results/validation/vtm/recovery-standalone-isolation.json)
verifies unchanged standalone source modules and no recovery imports from fresh
ordinary Voronoi or Delaunay calls.

CUDA memcheck with 32-byte allocation padding reports zero memory errors in the
[initial 14-case batch](../results/validation/vtm/recovery-memcheck-initial.txt).
That run had one test assertion failure: it incorrectly required SciPy's choice
of boundary diagonals for a cospherical case. The corrected test instead checks
the alternative tetrahedra with rational predicates. That test, four additional
one-ULP/near-coplanar cases, and all six VTM integration tests then pass the
[11-case follow-up](../results/validation/vtm/recovery-memcheck-followup.txt),
also with zero memory errors and stream-ordered lifetime checking. Together the
runs cover all current recovery/filter/VTM tests. Both use a diagnostic 32 KiB
stack; the first uses normal pooled allocation and the follow-up uses the
diagnostic synchronous allocator described in [CUDA validation](cuda-validation.md).
No production allocator or stack setting is changed.

The [native trace](../results/validation/vtm/recovery-device-audit.json) records
three warmed recovery replays for the 200-site corner cloud: 624 GPU kernel
activities and device-to-device copies. The captured graph contains 164 kernel,
40 memset, 19 conditional, and ten device-copy nodes, with no host or
allocation/free nodes. The CPU submits three graphs and performs one final
synchronization outside geometry. Record with Nsight `--cuda-graph-trace=node`;
the audit now rejects traces that omit kernel activity.

Stage timings below use CUDA events, five warmed repetitions, an NVIDIA L40,
and no observed CPU quota throttling. Setup is excluded. Stages are measured
separately, so their medians need not sum to the combined median; the auxiliary
stage is not part of a successful base-only replay.

| Input | Complete certified replay | Base attempt | Auxiliary candidate | Certificate |
|---|---:|---:|---:|---:|
| [8 cube corners](../results/validation/vtm/recovery-cube.json) | 190.26 ms | 182.98 ms, fails | 0.30 ms | 6.97 ms |
| [64-site lattice](../results/validation/vtm/recovery-grid.json) | 1872.37 ms | 1822.31 ms, fails | 1.56 ms | 48.40 ms |
| [200-site corner cloud](../results/validation/vtm/recovery-corner-cloud.json) | 19.13 ms | 10.18 ms, fails | 5.33 ms | 2.64 ms |
| [1,000 random sites](../results/validation/vtm/recovery-random.json) | 14.62 ms | 8.83 ms | 8.86 ms, unused | 5.69 ms |

Repeated failed base construction dominates the small structured cases. These
measurements motivate [#56](https://github.com/alecjacobson/warpower/issues/56),
including reuse of a previously successful construction strategy with fresh
certification. They are not a CPU mesher comparison or a VTM speedup claim.
