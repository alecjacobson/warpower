# Separating strict seed checks from exact fallback

A private split of the prepared seed checker reduces register and local-memory
requirements in its common kernel but makes complete solves about 1.3% slower
at both tested sizes. **The split is not adopted.** Production retains the
[combined point checker](prepared-point-queries.md).

The strict kernel uses the existing sphere box and point-specific Ball64
expression. It records a bit in a per-cell uint32 mask for each of the 32 query
lanes that encounters any uncertain or nonfinite result. After the entire strict
kernel finishes, a separate exact kernel repeats the original scan only for
flagged lanes. Every mask, including inactive capacity, resets before each call.
Both kernels continue checking rejected cells; rejection cannot hide a later
uncertain predicate. No variable queue, host decision, new precision rule or
symbolic tie policy is introduced. The final completeness certificate is unchanged.

## Complete timings and kernel properties

The paired controls use main `b0fb1d4`, runtime `6a82244`, NVIDIA L40 and Warp
1.19.0. Each replay begins with fresh sampling and includes domain refit, growth,
fitting, quality stages and extraction. The 1k experiment measures five replays;
the 4k experiment measures three. Every initial and timed quota window is clean.
Setup, compilation, transfers and offline checks are outside replay timing.

| Sites | Combined checker, complete ms | Split checker, complete ms | Change |
|---|---:|---:|---:|
| 1,000 | 716.827 | 726.020 | +1.28% |
| 4,096 | 2,640.651 | 2,675.762 | +1.33% |

At block size 32, the combined kernel has 255 registers and 15,936 bytes of local
storage per thread. The split strict kernel has 67 registers and zero local
storage; its exact fallback retains 255 registers and 15,936 local bytes.
These compiler properties do not establish which costs determine elapsed time.
The split adds a mask clear and an extra kernel per invocation. Its persistent
allocator-pool footprint increases by 32,000 bytes at 1k and 229,376 bytes at 4k.
There is no setup-speed or standalone-speed claim.

A separate instrumented 4k solve counts 624 prepared-query invocations and
12,882,912 cell instances over the benchmark's three graph executions. **No
cell or lane requires exact fallback on this trajectory.** The saved output is
bitwise equal to the clean split benchmark. Additional counter kernels make
these diagnostic timings unsuitable for performance claims. The absence of
fallback here does not generalize to exact ties or other inputs; those cases
are explicitly exercised by the tests below. Lower register use alone did not
produce a complete-solve gain in this experiment.

## Validation and retained evidence

The prototype passes 69 existing reconstruction tests and 36 captured-mask
comparisons against both previous and current exhaustive checkers. Those cases
change positions, duplicate owners and candidate prefixes across six device
frames, with empty/restored prefixes, stale live flags and poisoned pending
masks. Cube, octahedron, thin, random and fitted 4k clouds cover scales 2^-60,
1 and 2^60 and block sizes 32/256. Small cases also match independent exact
rational physical and symbolic masks.

Six further tests cover no fallback, all-lane fallback on an 84-site integer
cosphere, and an uncertain lane 31 after another lane has already rejected the
cell. They verify unsigned mask handling, empty-prefix reset and valid recovery.
Both complete benchmark pairs preserve bitwise positions, roles, full/material
tetrahedron sets, oriented boundaries, inputs, parameters and work counters.
Every candidate output passes independent CGAL exact global full-hull checks,
NumPy/AABB quality measurements and single-cycle boundary-vertex links.

There is no new broad-domain adoption benchmark, full-suite run, memory/lifetime
audit or native capture audit for this rejected prototype. No production code,
public tests, standalone API, default or README changes. The
[manifest](../results/validation/vtm/prepared-split-probe/manifest.json)
preserves kernels, drivers, all samples, compiler properties, test logs,
independent geometry checks and the instrumented counter run.

Tracked in [#175](https://github.com/alecjacobson/warpower/issues/175); the
[native CPU target](cpu-size-frontier.md) remains unmet.
