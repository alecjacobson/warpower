# Reusing exact cell proofs within facet completion

VTM's private fragment-reconstruction path now reuses empty-sphere checks within
one unchanged-input completion. It still filters previous cells against the
current sites on every invocation. This does not cache validity across moving
inputs or change the independently callable diagram and Delaunay defaults.

For a positively oriented tetrahedron, the existing certificate requires a
strictly negative symbolic sphere predicate against every other canonical site.
The symbolic lifting convention depends on site IDs as well as coordinates.
`CertifiedSeeds` already establishes this condition for every cell it emits.
The new private coordinator runs that filter immediately before consuming its
output, with the same points and canonical-owner buffers throughout completion.

## Proof provenance

The initial accepted seed prefix starts with valid cell-proof bits. Every round
then follows the existing facet search, candidate generation and full mesh
certificate, with two bookkeeping steps:

1. Deduplication writes a proof bit at the actual output index returned by its
   atomic counter. Only a winning source from the preceding certified prefix
   carries a proof. That prefix's live bits were set from its accepted count;
   unused capacity cannot supply a proof. New candidates start unproved. If a
   new duplicate wins before an identical old cell, it conservatively loses
   the optimization and receives a fresh check.
2. Positive-cell compaction moves proof bits with the same selected-prefix map
   as the tetrahedra. It cannot shift an old proof onto a different surviving
   cell when a flat input cell is removed.

A separate private point-parallel sphere kernel skips proved cells and performs
the original bound-filtered exact symbolic check on every unproved cell. The
ordinary global-certificate kernels remain unchanged. Large-cloud BVH checking
also retains its original full check; the optimization targets the small/medium
constructors used by VTM's previous-fragment strategy.

All index, canonical-owner, orientation, face-pairing, hull-support and coverage
checks still run. A fragment may continue after coverage/hull failures only when
its cell predicates have all passed; any geometric failure stops completion.
The next round can therefore retain those cell proofs. Newly exposed boundary
faces still require their ordinary global support checks.

There is no public skip-mask argument. The coordinator establishes the seed proof
it consumes, and each invocation reinitializes its proof state after filtering
current inputs. Capacity exhaustion, invalid counts and failed certificates keep
zero-output failure behavior. Same-input proof propagation is distinct from the
cross-frame coordinate/ownership snapshots tracked in
[issue #130](https://github.com/alecjacobson/warpower/issues/130).

## Validation and costs

The additional state consists of two fixed-size integer proof buffers and two
integer counters per private completion workspace. No replay allocation or host
readback is needed. The counters report reused and newly checked cells across
one completion, not all individual predicate evaluations. Diagnostic consumers
report zero when the current frame did not enter a nonempty completion.

The focused tests include independent rational determinants of the complete
symbolic lifting polynomial for one-cell and complete cospherical fragments,
three coordinate scales and duplicate sites. Further tests force old/new hash
winners, hash collisions, flat-cell compaction, rejected seed candidates,
nonempty new spheres beside valid retained proofs, exhausted budgets, invalid
and restored prefixes, and promoted/restored duplicate owners. Resulting hulls
also pass the original unmodified global certificate.


All 553 tests pass. The same 34 focused cases also pass unfiltered padded memory
checking in three independent processes: 24 proof cases, seven reconstruction
cases and three refinement cases, with zero errors. The groups take 147.24,
12.88 and 23.58 seconds respectively.

The original combined 34-case process was interrupted after approximately
20 minutes, while its final case had not reported a result. Its outcome remains
incomplete. Recent samples showed a busy main CPU thread and idle GPU; debugger
attachment was unavailable, so the precise cause was not established. The
[interrupted record](../results/validation/vtm/proved-completion-interrupted-audit.json)
and log are retained alongside the
[successful grouped runs](../results/validation/vtm/proved-completion-memory-summary.json).
No test or kernel was excluded from the reruns. These instrumentation timings
are separate from production performance.

A separate six-phase allocation/lifetime audit, with memory pooling disabled,
also passes with zero errors. Its
[device consumer records](../results/validation/vtm/proved-completion-replay-proof-counts.json)
show 305 reused cell proofs and zero new cell checks for the nearby frame's
completion. The moved frame accumulates 1,015 reused proofs and 123 full cell
checks across its rounds. The duplicate frame exhausts the bounded fragment and
successfully falls back; nonfinite input publishes zero output, and restoration
succeeds. The initial seed filtering still checks current input on every attempt.

## Complete-solve measurements

The [comparison record](../results/validation/vtm/proved-completion-summary.json)
and [artifact manifest](../results/validation/vtm/proved-completion-provenance.json)
compare baseline `2dc0343` with runtime `be9e71b` on NVIDIA L40 / Warp 1.19.0.
Each median uses three warm, complete fresh resident solves, including domain
refit, sampling, growth, fitting, four quality rounds and final extraction.
Setup, compilation, uploads, export and independent validation are outside replay
timing. All first/warm replay windows report zero CPU quota throttling.

| Complete resident solve | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 1,355.436 | 1,204.810 | 1.125× |
| Torus, 4,096 sites | 7,858.200 | 7,804.899 | 1.007× |
| Graded torus, 1,000 sites | 2,295.360 | 1,996.118 | 1.150× |
| Thin hand, 1,000 sites, quadrature 20/20 | 3,050.938 | 2,622.045 | 1.164× |
| Cavity, 1,000 sites, quadrature 20/20 | 1,275.556 | 1,139.671 | 1.119× |
| Disconnected, 1,000 sites, quadrature 20/20 | 1,121.361 | 898.191 | 1.248× |

The benefit at 4,096 sites is modest: only its smaller active-prefix constructors
use this strategy. All six pairs preserve full and material cell sets, oriented
boundaries, roles, optimizer steps and quality-move counts. Five have identical
point coordinates; the hand's maximum difference near zero is `1.83e-18`.
Aggregate floating-point reductions can differ with cell ordering. Both coarse
quadrature-4 recovery examples reproduce their preceding meshes bitwise, taking
2.656 s for the cavity and 2.528 s for disconnected components.

All eight saved full hulls pass independent CGAL 6.2.1 exact orientation, global
empty-sphere, face-pairing and hull-support checks; those checks allow cospherical
alternatives. Independent NumPy measurements verify material topology and quality.
All six gallery VTU round trips match the saved arrays.

The measured workspace pool delta increases by 64,008 bytes at capacity 1,000
and by 196,624 bytes at capacity 4,096, matching the new proof buffers/counters.
The resulting totals are 272.667 MB and 1,705.607 MB. Warm-cache setup changes
from 1.334 to 1.372 s and from 2.603 to 2.632 s, respectively. These allocator
deltas are not total process-memory peaks; cold compilation costs are separate.

To reproduce a complete measured case, using benchmark dependencies:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 PYTHONPATH=. python \
  -m benchmarks.mesher --case torus --n 1000 --repeats 3 --reference \
  --save-mesh torus1000.npz --output torus1000.json
```

Use `--n 4096` for the larger case; `examples.vtm_gallery` supplies the other
cases with the quadrature settings in the table. Run measurements separately
without competing GPU or heavy CPU work. The records retain all parameters,
first-run timings, setup, raw samples and CPU quota counters.

## Native trace and independent routines

Across three complete fresh torus1000 solves, the native profiles show:

| Work | Calls | Before (ms) | After (ms) |
| --- | ---: | ---: | ---: |
| Completion sphere checks | 951 | 521.252 | 60.834 |
| Additional proof-bit compaction | 951 | — | 1.968 |
| Global hull-face support checks | 966 | 579.242 | 580.188 |
| Facet candidate searches | 531 | 388.311 | 389.415 |

The [before](../results/validation/vtm/proved-completion-before-kernels.json) and
[after](../results/validation/vtm/proved-completion-kernels.json) records include
all kernels. Summed kernel time changes from 4,069.024 to 3,618.006 ms. Activity
count increases from 95,613 to 97,404 because proof bookkeeping adds launches;
the clean complete-solve timings include that overhead. The sphere and
deduplication kernel bodies are AST-identical to their original counterparts
after removing the new proof guard, counters and bit write. Mask provenance is
validated separately by the adversarial tests.

The [full native capture audit](../results/validation/vtm/proved-completion-device-audit.json)
records three graph launches and one final synchronization. The graph contains
10,288 kernel nodes, 3,485 memsets, 1,234 conditionals and 919 copies, all
device-to-device. No host transfers, callbacks, allocations or frees occur during
replay. Nsight observes only three copy activities, so every graph copy node is
also checked for its direction.

Standalone diagram and Delaunay APIs retain their preceding kernels and defaults.
Captured standalone recovery graphs contain none of the new proof kernels and
retain identical 256-thread facet launches. Paired warm gates give:

| Standalone operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Delaunay, 1,000 random sites | 9.184 | 9.170 |
| Delaunay, 100,000 random sites | 182.454 | 181.948 |
| Delaunay, 8³ structured grid | 3.959 | 3.965 |
| Delaunay, 256 fitted-hand sites | 13.375 | 13.431 |
| Diagram, 100,000 uniform sites | 18.817 | 18.874 |

The Delaunay rows use nine repeats and the diagram uses fifteen. All succeed
without quota throttling; differences are within 0.42% and are treated as timing
variation. No standalone performance gain is attributed to this private VTM
change. The full CPU outperformance target remains open.
