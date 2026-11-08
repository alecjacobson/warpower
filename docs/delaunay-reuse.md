# Reusing certified fragments during VTM

Nearby optimization steps often preserve most Delaunay tetrahedra. The internal
VTM core uses previous connectivity as a candidate fragment at capacities from
256 through 2,048 sites. The independently callable `DelaunayWorkspace`, power
diagram functions and fixed-boundary `ODTWorkspace` keep their existing paths.
For a small active prefix inside a larger
reserved VTM workspace, [device capacity routing](active-reconstruction.md) now
selects a smaller constructor that can use this strategy.
The active mesher's full route also [uses reuse at reserved capacities of
2,049–16,384 sites](extended-fragment-reuse.md), with current-input proof consumption
enabled throughout completion. Larger full routes retain ordinary construction;
the exhaustive seed filter costs O(sites × retained cells). The later report
records complete uniform/graded torus and Armadillo gains, storage costs and
validation of the extension. The measurements below describe the earlier change.

Every reconstruction rebuilds canonical duplicate owners and the current spatial
hierarchy on device. The seed filter removes malformed, duplicate, flat or
non-Delaunay previous cells using current original-coordinate exact predicates.
Survivors must be strict facets under the same symbolic tie policy used by
[facet completion](facet-completion.md). Previous validity alone never suffices.
[Same-input proof propagation](completion-proofs.md) now avoids repeating those
cell predicates inside the subsequent completion. [Current hull-support results](hull-support-reuse.md)
also let supported faces skip the following candidate search. This preserves the complete
certificate requirements and does not reuse proofs across input changes. Those
reports also contain newer timings and capture audits.

The filter now [shares current sphere geometry across its query lanes](shared-seed-geometry.md).
It prepares cofactors and conservative boxes once per unique cell on every call;
the exact/symbolic query semantics and full certificate remain unchanged. Only
VTM reuse enables the extra fixed buffers and preparation launch.

The fragment is extended for at most four rounds. A successful result must pass
the existing full certificate: physical orientation, site coverage, paired faces,
connectivity, convex-hull support and global empty-sphere conditions. If filtering,
completion or certification fails, the ordinary reconstruction/recovery pipeline
runs. A failure there still publishes an explicit error and zero tetrahedra.
Nonfinite input is rejected before hierarchy or predicate work. The matching
certificate face index is published along with accepted connectivity.

All fragment buffers, counters and completion scratch are allocated during setup.
Replay contains device kernels, device copies and conditional graph control.
Fresh full-mesher initialization clears the retained prefix; subsequent fitting,
refinement and quality trials can reuse it. Backtracking may leave a fragment
from a rejected proposal in scratch, but the next reconstruction must filter it
against the restored sites. Changing active prefixes and duplicate ownership use
the same current-input checks. There is no CPU geometry fallback.

Internal device diagnostics report whether reuse was attempted and accepted,
the retained seed count, completion rounds and failure bits. These do not replace
the final constructor error/count contract. Exhausting the four-round budget is
a request for ordinary reconstruction, not permission to publish a partial hull.

The focused tests exercise a captured producer/consumer loop with nearby and
unrelated inputs, duplicate owners, nonfinite input and restoration. Additional
cases cover corrupted fragments, zero completion budget, insufficient output
capacity, exact cospherical ties and one-ulp changes. Independent SciPy geometry
checks map duplicate sites to their canonical first indices; exact integer tests
check physical orientation and empty spheres for degenerate cases.

## Complete-solve measurements

Fresh paired measurements on NVIDIA L40 / Warp 1.19.0 compare `ea54594` with
runtime `8a721a7`. Each median uses three additional complete captured solves,
including domain refit, fresh sampling, growth, fitting, quality improvement and
extraction. Setup, first execution, raw samples and CPU quota observations are
reported separately in the linked records. All measured windows have zero
observed CPU quota throttling.

| Case | Before | With fragment reuse | Speedup |
| --- | ---: | ---: | ---: |
| Uniform torus, 256 sites | 1.694 s | 1.225 s | 1.38× |
| Uniform torus, 1,000 sites | 3.259 s | 2.224 s | 1.47× |
| Uniform torus, 2,048 sites | 4.736 s | 3.801 s | 1.25× |
| Prescribed-graded torus, 1,000 sites | 5.947 s | 3.715 s | 1.60× |
| Thin hand, 1,000 sites, quadrature 20/20 | 16.268 s | 3.366 s | 4.83× |

All five final results have identical material/full-hull connectivity sets,
oriented boundaries and roles. Torus positions agree bitwise, including the graded
case. Four hand coordinate components near zero differ by at most `1.08e-18`,
consistent with changed reduction order. Independent NumPy measurements verify
material quality and topology; CGAL exact predicates check every new full-hull
cell's physical orientation, global empty sphere, face pairing and hull support.
These CPU checks run offline after timing and are not used by the mesher.

Standalone Delaunay medians are 9.167→9.181 ms at 1,000 sites and
183.226→182.978 ms at 100,000 sites. The independent 100,000-site diagram has
18.960→18.246 ms medians over 15 samples. Its source is identical; this is no
observed regression, not a claimed diagram optimization. In total, 52 existing
runtime modules and the README are unchanged. Only the VTM core/scheduler select
the new internal constructor. The additional fixed scratch allocation and setup
cost are included in the raw complete-mesher records.

The [result summary](../results/validation/vtm/reuse-cells-summary.json) includes
all independent checks, source hashes and exact coordinate differences. The same
directory contains before/after timing JSON and the five new mesh-array exports.
The [provenance manifest](../results/validation/vtm/reuse-cells-provenance.json)
records their hashes and the validated revisions.
These are improvements over the preceding Warpower implementation. The
[quality-matched CPU performance goal](cpu-mesher-baseline.md) remains open;
neither these timings nor mean quality imply that it has been achieved.

## Capture and memory checks

The [full suite](../results/validation/vtm/reuse-cells-suite.log) passes 501 tests.
All seven new cases pass
[padded memory checking](../results/validation/vtm/reuse-cells-memcheck.txt), and
the [complete torus mesher](../results/validation/vtm/reuse-cells-integrated-memcheck.txt)
also passes padded checking with its nested optimization/refinement loops. The
six-phase producer/consumer audit passes
[allocation-lifetime checking](../results/validation/vtm/reuse-cells-lifetime.txt),
with zero errors. The phases cover fresh construction, unchanged topology,
three-round facet completion, duplicate-site fallback, invalid input and recovery.

The [native trace summary](../results/validation/vtm/reuse-cells-device-audit.json)
records 824 kernel activities, six graph launches and one final synchronization.
All observed transfers and all 38 copy nodes in the captured graph are
device-to-device. No replay host callbacks, CPU transfers, allocations or frees
are present. The [phase results](../results/validation/vtm/reuse-cells-native-run.json)
and [graph](../results/validation/vtm/reuse-cells-native.dot) are retained.
Instrumented checks use the [diagnostic stack settings](cuda-validation.md).

Reproduce representative measurements and the audit with:

```bash
python -m benchmarks.mesher --case torus --n 1000 --repeats 3 --reference --output results/local/reuse-torus.json
python -m examples.vtm_gallery --case thin-hand --n 1000 --repeats 3 --independent-quality --output results/local/reuse-hand
python -m benchmarks.reuse_delaunay --dot results/local/reuse.dot --output results/local/reuse-audit.json
```
