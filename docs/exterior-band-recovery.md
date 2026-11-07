# Exterior-band boundary recovery

The thickened-domain rule in Alliez et al. can retain exterior-center cells
that make a material boundary nonmanifold. The graded Armadillo regression has
two edges with four incident boundary triangles. Its complete Delaunay hull is
valid; each bad edge has a retained exterior-center cell separated from an
interior-center cell around the edge. Containment additions cannot resolve it.

`ExtractionWorkspace(..., exterior_recovery=True)` enables a bounded device
transaction for this case. The independent extractor defaults to `False`.
`VariationalMeshingWorkspace` enables it by default as a fallback after its
existing containment and additional-global-fitting recovery budget. Previously
successful fitting trajectories retain their selection and quality. Pass both `containment_recovery=False`
and `exterior_recovery=False` to use only the original circumcenter extraction
and additional-fitting policy. Diagram and Delaunay functions remain independent.

## Selection and safeguards

Recovery starts only with no input/query error and topology diagnostic exactly
2 (bad boundary edges). Each pass marks selected cells incident to such an edge
whose signed circumcenter distance is strictly positive. Interior centers and
centers on the input surface remain selected. Optional `feature_roles` is a
contiguous int32 array with one value per site: roles at least two protect every
incident cell. The full mesher supplies its current crease/corner roles; standalone
callers must supply roles when they need this protection.

The transaction preserves all vertices referenced by the starting selection,
including ordinary surface and interior sites. It may reduce selected volume
by at most one percent of the starting selected volume. It runs at most four
passes and stops when no cell changes or an unsupported diagnostic appears.
Every changed selection undergoes the complete face, edge and vertex-link
topology check. An orphaned vertex, excess volume change, unresolved topology,
or error rejects the transaction and restores the entire starting selection.

This is explicitly a boundary-approximation policy. A circumcenter outside the
domain does **not** prove that its tetrahedron is wholly exterior. The policy
does not establish a distance bound, exact facet conformity, preservation of
every input feature, or a quality threshold. Input cells must already form a
nonoverlapping positively oriented complex. The full mesher additionally requires
the output's component, shell, cavity and aggregate Euler counts to match its
input. Unresolved failures retain the existing zero-public-count behavior.

## Captured state and quality guards

All allocations and conditional graphs are created during setup. Replay uses
device classification, edge-table lookups, reference flags, compaction and
topology checks. It leaves the sites and complete input tetrahedralization
unchanged. `selected` and `source_indices` describe the final material selection.
If containment recovery also runs, that selection can include both additions
and removals relative to the original circumcenter rule.

Optional `exterior_enabled` is a device int32 scalar. Zero disables attempts on
that replay without changing ordinary extraction; any nonzero value enables
them. Omitting it enables attempts whenever the topology prerequisite holds.
The full mesher owns a shared scalar, resets it for each fresh solve, and enables
it only if the existing initial-extraction recovery budget is exhausted. The
same scalar controls extraction inside all subsequent quality guards.

Device `removed_count`, `exterior_rounds` and `exterior_rejected` reset every
replay. `removed_count` is zero on rejection; rounds count attempted passes;
rejected distinguishes an attempted unsuccessful transaction from no attempt.
The original classification measurements remain available in the circumcenter,
circumradius and signed-distance arrays.

After either recovery policy activates, selective optimization, boundary jitter
and interior moves assess the recovered subcomplex. The full mesher's checkpoint
and minimum-quality guards remain active. A fresh solve resets this trajectory
state; healthy trajectories retain their cheaper ordinary quality evaluator.

## Regression and independent validation

The reduced eleven-cell edge star tests preservation of interior centers,
surface centers, crease/corner cells and referenced vertices. It also tests
rollback for excess volume change and a second unresolved bad edge. Captured
device producers and consumers cover success, protected rejection, empty input,
upstream failure and restoration. A saved real-input regression requires exactly
the independently identified two cell removals and checks that selective quality
uses the same recovered cells and an independent NumPy minimum radius ratio.

The full graded scanned example uses the same input, sizing field and budgets
as the preserved failure. Its native full-hull, self-intersection, topology,
quality and boundary-distance checks are separate from resident GPU timing.
See the [scanned example](scanned-armadillo.md) for published measurements.


## Performance and graph validation

At baseline `7c251ee` and runtime `058fa60`, three warm complete resident solves
per case give the following medians on L40 / Warp 1.19.0. Setup, compilation,
transfer, validation and export are outside these timings. Both versions use
identical inputs, seeds, iteration budgets and quadrature settings.

| Complete solve | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Torus, 1,000 sites | 1,189.819 | 1,190.939 |
| Torus, 4,096 sites | 7,798.504 | 7,793.697 |
| Graded torus, 1,000 sites | 1,967.408 | 1,967.825 |
| Thin hand, quadrature 20/20 | 2,612.638 | 2,615.201 |
| Cavity, quadrature 20/20 | 1,131.187 | 1,129.513 |
| Disconnected, quadrature 20/20 | 896.716 | 897.863 |

All six retain identical full/material cell sets, oriented boundaries, roles and
iteration counts. Five have bitwise-identical positions; the hand differs only
at tiny near-zero coordinates. Every timing shift is below 0.15%. The uniform
scan also remains bitwise identical. Coarse quadrature-4 cavity and disconnected
cases remain valid; the coarse hand still rejects its candidate (issue #112).
These are regression checks, not speedup claims.

Standalone Delaunay before/after medians are 9.165/9.204 ms (1,000 random sites),
182.761/182.843 ms (100,000 random sites), 2.194/2.192 ms (64-site cospherical
grid), and 13.405/13.389 ms (fitted hand). Their geometry paths are unchanged.
The 100,000-site diagram's first pair was 18.246/18.979 ms, prompting two
alternating rounds of 30 samples: 18.043/18.072 ms and 18.275/18.036 ms.
Pooled recheck medians are 18.178/18.055 ms. The original slowdown did not repeat;
all original and recheck samples remain in the
[comparison record](../results/validation/vtm/exterior-summary.json).
All measured replay quota windows are unthrottled.

The added full-mesher workspace pool delta is 144,116 bytes at capacity 1,000
and 589,940 bytes at capacity 4,096. Warm-cache setup changes from 1.397/2.609 s
to 1.453/2.756 s respectively. These are allocator deltas, not process memory
peaks. Independent diagram/Delaunay allocation paths are unchanged; the default
independent extractor allocates no exterior-recovery buffers.

The full suite passes 575 tests. After improving inactive-padding storage in the
replay fixture, ten reduced transaction cases pass two independent unfiltered
padded memcheck runs; the saved scanned case and all seven lifetime phases also
pass with zero errors. The lifetime run disables the allocation pool and enables
API memory-access and stream-ordered lifetime checks. No kernel is filtered or
error suppressed. These diagnostics use an explicit 128 KiB CUDA stack allowance:

```bash
compute-sanitizer --tool memcheck --padding 32 --error-exitcode 99 \
  python -m benchmarks.memcheck --stack-bytes 131072 -- -q tests/test_exterior_band.py
compute-sanitizer --tool memcheck --padding 32 --error-exitcode 99 \
  python -m benchmarks.memcheck --stack-bytes 131072 -- -q tests/test_exterior_scanned.py
```

Earlier combined runs at 32 KiB reported a sticky CUDA700 without identifying a
kernel memory access. A retained, previously executed ordinary graph followed by
the reduced transaction reproduces it; 128 KiB resolves it without runtime changes.
The individual transaction and the ordinary public extractor sequence pass at
32 KiB. Failed logs and the reproducer are retained. The underlying mechanism is
still tracked in [issue #147](https://github.com/alecjacobson/warpower/issues/147);
this observation is not a proven sanitizer defect. The runtime never sets a
CUDA context stack allowance.

The [native extraction audit](../results/validation/vtm/exterior-extraction-native-audit.json)
observes seven graph launches, 465 kernels and one final synchronization. Its graph
contains 113 kernel, 67 memset, six conditional and three device-to-device copy
nodes. Phases cover successful recovery, protected rejection, empty input,
upstream failure, restored input, disabling and re-enabling recovery. Device
consumers publish volume only for valid results. The existing five-phase full
mesher loop also passes deformation, invalid inputs and recovery.

The [full-mesher audit](../results/validation/vtm/exterior-full-native-audit.json)
observes three complete fresh torus solves, 98,982 kernels, three graph launches
and one final synchronization. Both audits contain no host callback, host/device
copy or allocation/free API during replay. All copy nodes in both DOT graphs are
device-to-device; Nsight can omit graph copy activity records, so DOT and native
API evidence are checked together. Setup and readback are excluded explicitly.

The [artifact manifest](../results/validation/vtm/exterior-provenance.json) pins
runtime, fixture revision, source hashes, timing records, meshes, independent
checks, DOT graphs and diagnostic logs. The six-case comparisons plus both scans
and two coarse cases pass independent CGAL full-hull and NumPy material checks.
Gallery meshes additionally pass self-intersection and exact VTU round-trip
checks. The main README remains unchanged; CPU outperformance is still unachieved.
