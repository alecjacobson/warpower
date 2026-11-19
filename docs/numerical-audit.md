# Independent adjacency audits

## Current recovery coverage

The later [claim audit](../experiments/benchmark-claims-audit/README.md) found
seven incorrect neighbor pairs across four scene checkpoints.
[Robust numerical recovery](../experiments/robust-neighbor-recovery/README.md)
fixed those cases. The latest merged
[cooperative-capacity report](../experiments/cell-capacity-recovery/README.md)
audits 3,762 directed facets across live Bonsai and four large scene inputs,
including all fourteen directions of those seven pairs, with zero errors.
The audit covers CGAL disagreements, baseline changes and the original wrong
pairs; it does not certify every shared edge. All five graphs match that
change's immediate main-branch baseline.

That report records 210 targeted tests and scoped sanitizer runs, plus broader
captured-workspace findings that remain open in
[issue #300](https://github.com/alecjacobson/warpower/issues/300).
Neither targeted tests nor successful statuses constitute a universal exactness
or sanitizer-cleanliness claim.

## Historical all-halfspace row audits

The audits below compare Paragram `4f36b1f` with Warpower `66be246` on seed-42
100,000-site inputs. Each disputed row is reconstructed independently with all
100,000 sites' float64 halfspaces, a linear-programming interior point, and Qhull
dual hull vertices. Neither GPU implementation is treated as ground truth.

| Distribution and mode | Audited rows | Paragram extra / missing edges | Warpower extra / missing edges |
| --- | ---: | ---: | ---: |
| Five clusters, Voronoi | 359 | 220 / 594 | 0 / 0 |
| Uniform, power | 241 | 442 / 0 | 0 / 0 |
| Five clusters, power | 708 | 894 / 550 | 0 / 0 |

Counts are directed edges within audited rows. Rows on which both implementations
agree are not audited; these results do not establish correctness of the entire
graph. The oracle itself uses floating-point arithmetic, not exact predicates.
Earlier Warpower `d9ea6df` audits found one extra undirected edge between sites
26333 and 78334 in both clustered modes. Only one cell reported the facet before
symmetrization. Detecting that inconsistency and recomputing both cells in double
precision removes the edge. The audited `66be246` implementation agrees with the oracle
on every audited row in all three cases.

Machine-readable reports include every disputed row, input and graph hashes,
implementation identifiers, the oracle source hash, and the SciPy version:

- [Five-cluster Voronoi](../results/warpower-l40/consistency-cluster5-100k-voronoi-audit.json)
- [Uniform power](../results/warpower-l40/consistency-uniform-100k-power-audit.json)
- [Five-cluster power](../results/warpower-l40/consistency-cluster5-100k-power-audit.json)

The subsequent direct-CSR optimization (`d38b80c`) produces identical full
directed edge sets and status arrays on all three audited inputs. The
[equivalence record](../results/warpower-l40/direct-csr-audit-equivalence.json)
matches their canonical graph hashes to the original audit reports.

## Historical recovery and validation

Single-precision clipping can lose a consistent polygon boundary even when its
storage is large enough. Warpower retries unresolved cells in double precision,
retaining completed rows and valid candidate IDs from previous attempts. The
100,000-site ten-cluster cases completed with no failed cells in either mode.
Empty weighted cells retain their distinct status.

The recorded implementation passed 51 tests, including duplicate-site ownership, strong
weights, CSR hints, stream handling, chunking, forced recovery, and independent
small-diagram oracle comparisons. Compute Sanitizer reports zero errors across
45 small tests and two targeted consistency regressions. The targeted run
excludes the two initial 100,000-site single-precision clipping kernels; recovery
and adjacency kernels remain instrumented. The two ten-cluster regressions run
outside the sanitizer, and two other tests are CPU-only oracle checks.

These counts describe that revision, not the current suite size.
Run the current full test suite with `python -m pytest -q` from the checkout.
The [validation record](../results/validation/validation.json) includes the
sanitizer commands, their instrumentation scope, and links to the test logs.

See [benchmark reproduction](benchmarks.md) and [the construction algorithm](architecture.md).
