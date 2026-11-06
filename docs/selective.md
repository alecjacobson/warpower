# Selective VTM updates

The internal `warpower._vtm.VTMWorkspace` accepts `selective_threshold=0.3` to
restrict positional optimization to vertices around poor retained tetrahedra.
This implements the selective-update stage discussed in
[Alliez et al., section 3.6](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Alliez05.pdf).
Activation is explicit; use it after ordinary global fitting has made the mesh
reasonably stable. The complete public mesher scheduler remains under development.

```python
from warpower._vtm import VTMWorkspace

local = VTMWorkspace(
    domain, fitted_points, max_iterations=10, selective_threshold=0.3,
    selective_tolerance=0.4, active_count=active_count, sizing=sizing,
)
local.run()  # Also records into an enclosing Warp ScopedCapture.
```

`selective_threshold` is in `(0, 1]`, and defaults to `None`. The default uses
the existing uniform/graded optimizer without selection buffers, classification
work, extra graph nodes, or additional kernels. Existing geometry, Delaunay, and
ODT routines remain independent.

## Selection and acceptance

Before each iteration, the optional helper classifies the current full-hull
tetrahedra with the same rule as [domain extraction](extraction.md). It retains
material-interior circumcenters and exterior centers whose distance to the
boundary divided by circumradius is below `selective_tolerance` (default 0.4,
valid range `[0, 1)`).
Use the same tolerance for subsequent extraction when interpreting the quality
guard. Radius ratio is `3 * inradius / circumradius`.

This stage only needs membership and quality. It skips nearest-distance searches
for established interior centers and limits exterior searches to the thickening
cutoff. Only the search radius is expanded for roundoff; the final `d / R`
comparison stays strict. Zero tolerance, ambiguous rays, and extreme circumradii
retain the full-query fallback, including exact boundary-center handling. The
independent extractor still computes and publishes its full distance outputs.

The helper first marks all vertices of retained tetrahedra whose radius ratio
is strictly below the threshold. A separate pass expands that immutable mark
through the edges of retained tetrahedra, producing exactly one additional
neighbor ring. Unselected positions remain fixed during the proposed update.
Corner anchors remain fixed; selected surface and crease sites retain their
usual fitting targets. Active prefixes and supplied grading fields are supported.

The usual reconstruction, exact Delaunay certification, and boundary/ODT
acceptance rule still apply. A candidate that passes them is classified again
and rejected if it reduces the minimum retained radius ratio beyond a relative
`1e-10` numerical tolerance. Device backtracking then reduces the step. Exhausting
the budget restores the old points, connectivity, and statistics. This additional
guard is an implementation choice, not a convergence theorem from the paper.
It can reject all directions or accept a tiny rounded step; it does not guarantee
that the threshold is reached or that selective updates are faster.

Selection uses fixed buffers and GPU reductions. It repeats on every iteration
as positions, connectivity, and boundary membership change. Capture replay does
not read masks/counts on the host, allocate storage, synchronize, or use a CPU
fallback. The whole Delaunay mesh is still reconstructed for each trial: reducing
the moving-site set does not reduce all reconstruction and validation work.

## Status and diagnostics

The core's published geometry remains `full_tetrahedra` and
`full_tetrahedra_count`, covering the convex hull. Extract the material-domain
mesh before using it as the final tetrahedralization.

- `termination=0`: iteration budget ended.
- `termination=1`: accepted motion fell below the core movement tolerance.
- `termination=2`: backtracking exhausted; the prior valid state was restored.
- `termination=3`: no selected movable vertices remain. This also occurs when
  all selected vertices are protected corners.

Core `error` bit **16** means the current state could not be classified for
selection, or its retained domain was empty. The full-hull output count becomes
zero. Corrected inputs can recover on the next call. Other core error bits keep
their existing meanings.

The private `core.selection` helper exposes `mask`, `bad_vertices`,
`selected_count`, `rejections`, and `last_error` for diagnostics. Masks describe
the last iteration's starting mesh, before its proposed move. `rejections`
counts candidates rejected by the additional quality guard during this run.
Its `last_error` bits are 1 for failed classification, 2 for an empty retained
candidate, and 4 for reduced minimum quality. These are search rejections,
distinct from a fatal core error. Classification scratch may describe the last
rejected candidate; use the independent extractor for final mesh statistics.

Tangential boundary jitter and worst-element relocation/insertion are separate
stages still tracked in [#70](https://github.com/alecjacobson/warpower/issues/70)
and [#71](https://github.com/alecjacobson/warpower/issues/71).

## Validation and measured tradeoffs

The complete suite passes **300 tests**, including 20 selective/retention cases
([output](../results/validation/vtm/selective-full-tests.txt)). Independent
checks cover exactly one neighbor ring, fixed unselected positions, retained
minimum quality, rollback, graded active prefixes, corner anchors, and captured
producers/consumers. Full-query comparisons cover nonconvex, cavity, disconnected,
exact-boundary, and adjacent-float strict-threshold cases.

Two classification cases pass fully instrumented pooled memcheck with padding
([output](../results/validation/vtm/selective-pooled-memcheck.txt)). Three
rollback/recovery/threshold cases pass full memory and allocation-order checking
([output](../results/validation/vtm/selective-lifetime.txt)). Both report zero
errors under the [documented diagnostic settings](cuda-validation.md).
The [native audit](../results/validation/vtm/selective-device-audit.json) records
9,852 kernel activities across three complete selective/extraction replays,
including accepted and rejected steps. All graph copies are device-to-device;
there are no host callbacks, host transfers, or allocation/free nodes.

On L40, starting from the same 1,000-site torus after 30 global iterations gives:

| Further optimization | Accepted / rejected steps | Worst radius ratio | Median radius ratio | Median GPU time |
|---|---:|---:|---:|---:|
| Global, 10 iterations | 10 / 0 | 0.1093 | 0.8864 | 297.49 ms |
| Guarded selective, retained-only queries | 4 / 14 | 0.1188 | 0.8831 | 606.36 ms |
| Guarded selective, full-query reference | 4 / 14 | 0.1188 | 0.8831 | 822.90 ms |

The starting worst ratio is 0.0343. The selective result retains 4,150 tetrahedra,
has one genus-one component, and has sampled directed boundary RMS 0.01805 model
units. Global continuation has lower RMS, 0.01776. All results pass independent
Delaunay and topology checks, with no CPU throttling
([selective](../results/validation/vtm/selective-torus-1000.json),
[global](../results/validation/vtm/selective-global-control-1000.json),
[full-query reference](../results/validation/vtm/selective-full-query-reference-1000.json)).
These timings include resident point reset, further optimization, and extraction;
initial fitting and other full-mesher stages are excluded. Selective mode offers
a different quality tradeoff here, not a speedup over global continuation.

Retained-only queries reduce this selective loop by **26.3%** without changing
its accepted/rejected counts or final quality. Classification falls from about
1,028 ms to 378 ms across three profiled replays; construction and certification
now dominate ([initial profile](../results/validation/vtm/selective-full-query-kernel-profile.json),
[current profile](../results/validation/vtm/selective-kernel-profile.json)).
Further performance work remains in #43, #48, and #58. This is not a native CPU
speedup measurement.

The [source audit](../results/validation/vtm/selective-isolation.json) finds all
35 other runtime modules and all eight preexisting VTM kernel/function ASTs
unchanged from `cf9e50d`. Default uniform VTM, 1,000 sites and 20 iterations,
takes [520.50 ms before](../results/validation/vtm/selective-default-before.json)
and [519.83 ms after](../results/validation/vtm/selective-default-after.json),
a 0.13% difference within measurement variation. Counts, roles, and accepted
steps match; energy histories agree within `2e-14` relative roundoff.

```bash
python -m benchmarks.selective --mode selective --reference \
  --output results/local/selective.json
python -m benchmarks.selective --mode global --reference \
  --output results/local/global-continuation.json
python -m benchmarks.selective --full-query-reference --reference \
  --output results/local/selective-full-query.json
```
