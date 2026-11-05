# Independent adjacency audits

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
precision removes the edge. The current implementation agrees with the oracle
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

## Recovery and validation

Single-precision clipping can lose a consistent polygon boundary even when its
storage is large enough. Warpower retries unresolved cells in double precision,
retaining completed rows and valid candidate IDs from previous attempts. The
100,000-site ten-cluster cases now complete with no failed cells in either mode.
Empty weighted cells retain their distinct status.

The implementation passes 51 tests, including duplicate-site ownership, strong
weights, CSR hints, stream handling, chunking, forced recovery, and independent
small-diagram oracle comparisons. Compute Sanitizer reports zero errors across
45 small tests and two targeted consistency regressions. The targeted run
excludes the two initial 100,000-site single-precision clipping kernels; recovery
and adjacency kernels remain instrumented. The two ten-cluster regressions run
outside the sanitizer, and two other tests are CPU-only oracle checks.

Run the full test suite with `python -m pytest -q` from the checkout.
The [validation record](../results/validation/validation.json) includes the
sanitizer commands, their instrumentation scope, and links to the test logs.

See [benchmark reproduction](benchmarks.md) and [the construction algorithm](architecture.md).
