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

## Memory tradeoff

Two prefix constructors require about 689 MB of additional device pool storage
with the default clipping capacities. Scratch stays allocated even when a branch
is inactive. Sharing compatible scratch across mutually exclusive branches is
tracked in [#122](https://github.com/alecjacobson/warpower/issues/122).

## Reproduction

```sh
python -m benchmarks.mesher --case torus --n 1000 --capacity 16384 --repeats 3 --reference --save-mesh results/local/prefix-mesh.npz --output results/local/prefix-timing.json
python -m benchmarks.active_delaunay --dot results/local/prefix.dot --output results/local/prefix-audit.json
pytest -q tests/test_active_delaunay.py
```

The benchmark reports setup time, pool memory, complete replay times and device
branch counts. Its independent CPU checks and serialization happen after timing.
