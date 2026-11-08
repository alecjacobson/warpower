# Bounded tangential topology recovery

The full VTM mesher can now recover the uniform and graded 8,192-site scanned
Armadillo failures tracked in [#228](https://github.com/alecjacobson/warpower/issues/228).
Both complete all four quality rounds and independently validate. This extends
the mesher's recovery policy; it does not change the independently callable
power-diagram or Delaunay algorithms.

## Policy and transaction

`VariationalMeshingWorkspace(..., tangential_recovery=None)` inherits the
`exterior_recovery` setting. Explicit `False` omits its workspace and graph work.
Explicit `True` requires `exterior_recovery=True`. `quality_rounds=0` disables the
ordinary quality schedule but retains this extraction recovery; disable the new
policy explicitly to omit it. It allocates and captures during construction;
all replay decisions and geometry remain Warp device work.

After ordinary fitting, containment, exterior-band and optional proven-exterior
recovery fail, the mesher can use the main extractor's saved witness: the smallest
eligible surface vertex with a disconnected boundary link in a rejected band transaction. It obtains the
witness before restoring the selection, using the current boundary's twin/degree
traversal. Feature and corner vertices are ineligible. No observed vertex ID is
hardcoded. Every actual band attempt clears its witness; recovery also requires
the CURRENT rejected flag, preventing an earlier witness from activating a
later skipped attempt. Only the mesher's main extractor owns this witness.

A candidate moves that vertex in the nearest input triangle's tangent plane.
The first 16 directions use at most 10% of its shortest full-mesh incident edge;
the next 16 use at most 20%. Actual binary32 displacement must fit the bound,
remain finite and nonzero, and not coincide with another active site. Every
candidate rebuilds and certifies the complete current Delaunay convex hull.
Acceptance additionally requires:

- A valid, nonempty material mesh matching the input component/shell/cavity/Euler
  counts, with manifold boundary edges and connected vertex links.
- Strictly reduced squared radius-ratio deficit below `quality_threshold`, with
  nondecreasing minimum radius ratio up to the existing `1e-10` relative slack.
- The existing quality-move 1% boundary-fit, retained-volume and RMS bounds,
  including their numerical slacks (RMS also allows `1e-7` of the domain diagonal).
- Every originally referenced material vertex still referenced, and every active
  site's surface/crease/corner role unchanged.

Candidate extraction retains containment and band transactions; it omits its
optional whole-cell exterior search. Final publication still runs the mesher's
configured extraction policy. The global fitting energy-decrease rule is
unchanged. These are quality recovery moves: a small fitting-energy increase
within the quality bound is permitted and is not a global-step energy decrease.

Exhaustion restores original points and the active count, reconstructs the full
hull, and reevaluates/refits. Merely restoring public cell rows would leave private
certificate/index state stale. Any restoration geometry error invalidates the
full count. The parent retains explicit failure and zero public output counts if
no valid material mesh is recovered. Each fresh solve resets recovery diagnostic
counts/metrics even if invalid input skips the search. The finite 32-candidate
search is not guaranteed to repair every input.

## Measured complete solves

Baseline `ceb5ea1420b4` and runtime `6178890a2a1f`, NVIDIA L40, Warp 1.19.0.
Each table entry uses three warmed fresh captured replays, including refit,
sampling, growth, fitting, quality and extraction. Setup/upload/readback/export
are excluded. Initial and timed replay CPU-quota windows are unthrottled.
The two formerly failing 8K rows have no successful-mesh speedup ratio.

| Case | Before, ms | After, ms | Change |
|---|---:|---:|---:|
| torus16384-fixed32 | 4889.893 | 4895.704 | +0.12% |
| torus1000 | 513.307 | 512.277 | -0.20% |
| thin-hand-seed7 | 1246.329 | 1252.573 | +0.50% |
| cavity | 357.008 | 357.590 | +0.16% |
| disconnected | 487.265 | 487.718 | +0.09% |
| plate | 921.662 | 920.356 | -0.14% |
| armadillo-graded4096 | 2581.819 | 2577.673 | -0.16% |
| armadillo-graded8192-recovery | Failed (error 64) | 4301.835 | New valid result |
| armadillo-uniform8192-recovery | Failed (error 64) | 3944.245 | New valid result |

The validated outputs can be inspected in the quality-colored
[uniform boundary](../results/validation/vtm/tangential-topology-recovery/tangential-production-uniform8192-boundary-v1.png),
[graded boundary](../results/validation/vtm/tangential-topology-recovery/tangential-production-graded8192-boundary-v1.png),
[uniform cutaway](../results/validation/vtm/tangential-topology-recovery/tangential-production-uniform8192-cutaway-v1.png), and
[graded cutaway](../results/validation/vtm/tangential-topology-recovery/tangential-production-graded8192-cutaway-v1.png).
These CPU renders use the saved timed meshes; rendering is outside meshing timing.

Both Armadillo cases use initial 256, seed 2026, surface/crease subdivisions 1,
feature angle 120°, corner angle 45°, and four quality rounds. Their recovered
meshes match the previously validated private prototype's connectivity, roles
and scheduler work. The recorded comparisons quantify any coordinate roundoff.
The Armadillo is an additional scanned benchmark, not a model from Alliez's paper;
[asset provenance and download](scanned-armadillo.md) remain separate from the
library's code license.

| Case | Selected witness | Accepted attempt | Full tets | Material tets | Boundary faces |
|---|---:|---:|---:|---:|---:|
| armadillo-graded8192-recovery | 6598 | 1 | 49,660 | 39,510 | 6,204 |
| armadillo-uniform8192-recovery | 3406 | 22 | 49,595 | 39,377 | 6,276 |

All 16 successful baseline/production meshes pass independent full-hull/global
empty-sphere validation, positive-volume and material-quality checks, topology
and boundary-link checks and exact boundary self-intersection tests. All 12
gallery VTU exports also round-trip exactly.
Previously successful cases preserve canonical cells, oriented boundary, roles
and scheduler work; coordinate differences are retained explicitly in the
[comparison report](../results/validation/vtm/tangential-topology-recovery/tangential-production-final-v1-independent.json).
Coordinate differences occur in `thin-hand-seed7` (one scalar, max 4.40457e-18)
and `cavity` (one scalar, max 2.32831e-10). Their independently checked meshes
and recorded quality bounds remain valid.

## Standalone performance and setup cost

The source audit pins 261 files, including 255 unchanged files. Existing mesher,
extraction and quality-move kernels retain their code; new recovery work is
scoped to the full mesher. Diagram/Delaunay source and allocations are unchanged.
Standalone inputs and diagnostics match in the paired measurements:

| Routine | Before, ms | After, ms | Change |
|---|---:|---:|---:|
| delaunay1000 | 9.171 | 9.191 | +0.21% |
| delaunay100000 | 182.758 | 181.396 | -0.75% |
| delaunay-grid | 2.203 | 2.184 | -0.88% |
| delaunay-fitted-hand | 13.721 | 13.380 | -2.48% |
| diagram100000 | 18.138 | 18.060 | -0.43% |

The optional recovery adds preallocated extraction, checkpoint and proposal
storage. Constructor allocator deltas below include owned workspace storage;
they are not process peak-memory measurements. [Issue #241](https://github.com/alecjacobson/warpower/issues/241)
tracks omitting inherited approximate-query scratch unused by this complete-extraction
recovery; that saving is not part of these measurements. Setup includes construction and
capture, using the available compilation cache, and is outside replay timing.

| Case | Before bytes | After bytes | Added bytes | Constructor seconds |
|---|---:|---:|---:|---:|
| thin-hand-seed7 | 280,165,171 | 291,997,380 | 11,832,209 | 1.307 → 1.426 |
| armadillo-graded4096 | 1,789,782,047 | 1,837,651,982 | 47,869,935 | 2.450 → 2.853 |
| armadillo-uniform8192-recovery | 2,942,037,928 | 3,037,767,575 | 95,729,647 | 2.855 → 3.314 |

## Capture, rollback and memory qualification

All 1,326 tests pass. Nine new focused tests cover captured witness selection,
feature/count/upstream-error changes, stale-witness rejection, reference/role
preservation, binary32 proposal bounds, restoration errors and policy/storage
scoping. A separate full uniform 8K nondefault-stream graph runs five phases:
one-attempt exhaustion, successful recovery, invalid target, collapsed domain,
then successful recovery again. It keeps output addresses stable and gates the
consumer with zero counts on failure. Both successful frames independently
validate and recover exactly; exhaustion restores bitwise points and the canonical
complete hull. The same sequence is checked in a native CUDA trace. A separate offline guard
check binds the restored baseline to the recorded original failed mesh and verifies
all 8,188 original material references and every role in all four accepted frames
across the ordinary/native loops. Only surface site 3406 moves, by 19.998043% of its
shortest original edge; the quality deficit falls 2.893661 → 2.860257, with unchanged
minimum ratio and retained-volume change within 1%. These are the initial recovered
meshes before the later ordinary quality rounds.

Native full-solve and five-phase loop traces contain device kernels, conditional
nodes, memsets and device-to-device copies, one graph launch and a final external
synchronization. No host transfers, callbacks or allocation/free operations occur
inside replay. First-launch instantiation/upload, where present, is reported by
the native audit. Two sanitizer modes cover eight unchanged focused kernel tests,
three isolated policy cases and the unchanged full loop: 32-byte padded memcheck,
and allocation-pool-disabled API/lifetime checks. All ten jobs pass with zero
errors. Diagnostic stack size is 131,072 bytes; periodic 60-second Python stack
dumps are observations, not process timeouts.
Both sanitizer loops also match the independently validated ordinary-loop
geometry, including bitwise coordinates and oriented simplex connectivity;
the [comparison](../results/validation/vtm/tangential-topology-recovery/tangential-production-memory-v3-comparison.json)
binds the saved frames and source hashes.

The original combined-contract run hit its 600-second limit while constructing a
second workspace. A fresh retry reached its 1,800-second limit at the third
workspace's final constructor synchronization. Neither produced a sanitizer
verdict; both incomplete runs are retained. Following the isolation used in
[issue #155](https://github.com/alecjacobson/warpower/issues/155), the three policy
constructors run in fresh processes. A source AST audit verifies that every
original statement and assertion is preserved, repeating only the common domain
setup. Isolation changes workspace coexistence during instrumentation; it does
not establish a combined-process sanitizer pass. The unchanged combined test
passes in the ordinary full suite. Isolated jobs have 600-second limits; the full
loop jobs retain 1,200-second limits. Diagnostic timings are not performance samples.

The [artifact manifest](../results/validation/vtm/tangential-topology-recovery/manifest.json)
pins sources, raw results, trace archives, input references and checksums. Earlier
prototype and pre-reset production loops remain retained with their own hashes.
The README's measured Paragram flagship remains unchanged. The wide
comparable-quality native CPU speedup goal remains open; these results establish
recovery and GPU revision comparisons, not that CPU performance target.
