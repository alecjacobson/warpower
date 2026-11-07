# Reconstruction sized to the active prefix

A growing mesh reserves space for its final site count. Previously, every VTM
reconstruction processed that full allocation, padding inactive sites with
duplicates. This also disabled small-cloud reconstruction strategies whenever the
reservation exceeded their size limits, even if few sites were active.

The internal VTM core now preallocates constructors for 1,024 and 2,048 sites
alongside the ordinary full constructor when a device active count is present
and reserved capacity exceeds 2,048. A device kernel selects the smallest fitting
prefix. Each small constructor can recertify and complete previous fragments;
larger inputs use ordinary reconstruction. A failed small attempt also falls
back to the full constructor, preserving its larger output and strategy budgets.
If that fails, publication remains an explicit error with zero tetrahedra.

## Geometry and graph contract

Before reconstruction, VTM fills its separate mesh scratch buffer with active
positions followed by exact copies of site zero. Consequently, every distinct
site lies in any prefix large enough to contain the active sites. The selected
constructor's complete original-coordinate certificate proves the active convex
hull. Original site indices, first-owner duplicate handling, and symbolic tie
ordering are preserved. Inactive input storage remains available to its producer;
it is neither interpreted as geometry nor overwritten by padding.

The router publishes connectivity, count, errors and scalar diagnostics into
fixed output arrays. It deliberately does not expose a certificate face index
from a different constructor. Internal consumers build their own required face
indices from the current output. Independent Delaunay workspaces retain their
existing certificate/index contract and implementation.

All constructors, conditions, prefix views and output storage exist before
capture. Growth, shrinkage, invalid-count handling, construction and fallback use
device conditional nodes. Fresh full-mesher sampling clears prefix fragments and
strategy hints. Rejected trials may leave fragments in scratch, but every later
use must recertify them against current sites. No validity is inherited merely
because a branch succeeded on an earlier input.

The standalone power-diagram, Delaunay and fixed-boundary ODT routines are
unchanged. VTM capacities at or below 2,048 and fixed-budget cores without a
device active count keep their preceding paths.

## Complete resident-solve measurements

Fresh measurements on NVIDIA L40 / Warp 1.19.0 compare main `3b3ca21` against
runtime `f81bbaf`. Each median contains three additional complete replays after
a first solve and warmup. Every replay refits the input surface and starts fresh
sampling; construction, fitting, refinement, quality trials, extraction and
diagnostics are included. All four cases use the same analytic torus, seed 2026,
256 initial sites and four quality rounds. No CPU quota throttling was observed.

| Reserved sites | Target sites | Before | Routed | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1,000 | 6.570 s | 2.463 s | 2.67× |
| 4,096 | 2,048 | 6.245 s | 3.865 s | 1.62× |
| 4,096 | 4,096 | 9.313 s | 8.614 s | 1.08× |
| 16,384 | 1,000 | 32.388 s | 2.884 s | 11.23× |

The full 4,096-site case makes 27 calls to the 1,024-site constructor, 14 to the
2,048-site constructor and 170 to the full constructor. Its smaller improvement
reflects how much reconstruction happens after both prefixes are outgrown.

These gains compare with preceding Warpower under the same reservation. They
are not CPU speedups, nor are they gains over a minimally sized 1,000-site
workspace. The quality-matched CPU performance target remains open in
[#40](https://github.com/alecjacobson/warpower/issues/40).

## Setup and memory tradeoff

Setup increases from approximately 0.85–0.87 seconds to 2.43–2.47 seconds in these
measurements. Setup, upload, compilation, output serialization and independent
checks are excluded from the replay times above. In particular, the smaller
resident-solve gain for a full 4,096-site target does not offset the extra setup
cost for a single solve.

Two prefix constructors require about 689 MB of additional device pool storage
with the default clipping capacities. Scratch stays allocated even when a branch
is inactive. Sharing compatible scratch across mutually exclusive branches is
tracked in [#122](https://github.com/alecjacobson/warpower/issues/122).
For capacity 4,096, the measured pool grows from 981,523,967 to 1,670,773,237 bytes;
for capacity 16,384, it grows from 1,391,336,527 to 2,080,588,869 bytes.

## Validation

The [full suite](../results/validation/vtm/active-prefix-suite.log) passes 512 tests.
The 11 focused routing tests pass
[padded memory checking](../results/validation/vtm/active-prefix-memcheck.txt)
with zero errors, including device producers/consumers, growth and shrinkage,
invalid counts, duplicate owners, coplanar/nonfinite inputs, exact cospherical
ties, one-ulp changes, exhausted output capacity, full fallback and VTM rollback.
A [complete nested mesher case](../results/validation/vtm/active-prefix-integrated-memcheck.txt)
with capacity 32,768 and target 128 also passes with zero errors.

The seven-phase resident audit passes
[stream-ordered allocation-lifetime checking](../results/validation/vtm/active-prefix-lifetime.txt).
Its [native trace](../results/validation/vtm/active-prefix-device-audit.json)
records 775 kernel activities, seven graph launches and one final synchronization.
The captured graph has 1,680 kernel, 412 memset, 230 conditional and 166
device-to-device copy nodes. No host copies, host callbacks, replay-time
allocation/free or host geometry decisions appear. Nsight does not report
activity records for every graph copy; the DOT graph independently establishes
the direction of every copy node.

All four paired meshes have bitwise-identical positions, matching site roles,
equal full/material tetrahedron sets and identical oriented boundary faces.
Each saved result also passes the offline CGAL exact original-coordinate hull
checker and independent NumPy quality/topology measurements. These CPU checks
are validation tools, never runtime fallbacks. The
[result summary](../results/validation/vtm/active-prefix-summary.json) contains
the per-case checks and raw-record links are covered by the
[provenance manifest](../results/validation/vtm/active-prefix-provenance.json).

The source audit finds 54 existing runtime modules unchanged; only `_vtm.py`
selects the new internal router. Standalone timing gates remain close: 1,000-site
Delaunay 9.180→9.177 ms, 100,000-site Delaunay 182.103→184.823 ms (+1.49%), and
100,000-site diagram adjacency 18.207→18.061 ms. These routines execute identical
source before and after; the small timing differences are not attributed to the
new VTM path. Raw samples and CPU quota observations are retained.

## Reproduction

```sh
python -m benchmarks.mesher --case torus --n 1000 --capacity 16384 --repeats 3 --reference --save-mesh results/local/prefix-mesh.npz --output results/local/prefix-timing.json
python -m benchmarks.active_delaunay --dot results/local/prefix.dot --output results/local/prefix-audit.json
pytest -q tests/test_active_delaunay.py
```

The benchmark reports setup time, pool memory, complete replay times and device
branch counts. Its independent CPU checks and serialization happen after timing.
