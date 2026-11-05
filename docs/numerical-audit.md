# Independent adjacency audits

The audits below compare Paragram `4f36b1f` with Warpower `d9ea6df` on seed-42
100,000-site inputs. Each disputed row is reconstructed independently with all
100,000 sites' float64 halfspaces, a linear-programming interior point, and Qhull
dual hull vertices. Neither GPU implementation is treated as ground truth.

| Distribution and mode | Audited rows | Paragram extra / missing edges | Warpower extra / missing edges |
| --- | ---: | ---: | ---: |
| Five clusters, Voronoi | 359 | 220 / 594 | 2 / 0 |
| Uniform, power | 241 | 442 / 0 | 0 / 0 |
| Five clusters, power | 708 | 894 / 550 | 2 / 0 |

Counts are directed edges within audited rows. Rows on which both implementations
agree are not audited; these results do not establish correctness of the entire
graph. The oracle itself uses floating-point arithmetic, not exact predicates.
Both clustered cases retain the same extra undirected edge, between sites 26333
and 78334. This discrepancy remains an explicit numerical limitation.

Machine-readable reports include every disputed row, input and graph hashes,
implementation identifiers, the oracle source hash, and the SciPy version:

- [Five-cluster Voronoi](../results/warpower-l40/robust-cluster5-100k-audit.json)
- [Uniform power](../results/warpower-l40/robust-uniform-100k-power-audit.json)
- [Five-cluster power](../results/warpower-l40/robust-cluster5-100k-power-audit.json)

## Recovery and validation

Single-precision clipping can lose a consistent polygon boundary even when its
storage is large enough. Warpower retries unresolved cells in double precision,
retaining completed rows and valid candidate IDs from previous attempts. The
100,000-site ten-cluster cases now complete with no failed cells in either mode.
Empty weighted cells retain their distinct status.

The implementation passes 49 tests, including duplicate-site ownership, strong
weights, CSR hints, stream handling, chunking, forced recovery, and independent
small-diagram oracle comparisons. Compute Sanitizer reports zero errors across
45 tests; the two 100,000-site cluster regression cases were excluded from that
memory-check run (the remaining two tests are in the CPU-only reference file).

See [benchmark reproduction](benchmarks.md) and [the construction algorithm](architecture.md).
