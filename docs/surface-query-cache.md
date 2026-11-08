# Exact repeated surface queries

Complete replay time improves **10.5% on torus4096** and **25.4% on graded
Armadillo** in the pinned production comparison below. These are GPU revision
comparisons; the wide native CPU outperformance target remains open.

The complete mesher reuses exact surface-query results in its private extraction,
selective update and quality-move consumers. Each consumer owns its own cache.
Standalone power-diagram, Delaunay, extraction and core VTM defaults retain their
ordinary paths. This work is tracked in [issue #218](https://github.com/alecjacobson/warpower/issues/218).

## Keys, validity and execution

The key contains all three binary64 circumcenter bit patterns. Nearest queries
also require the identical distance cutoff bits. Hashes only select a
probe sequence; lookup verifies the complete coordinate key. Signed zero is
kept distinct. A result can be reused only if the previous invocation actually
performed that query. Ray and nearest results, including the face, point and
distance, are stored without changing their bits. Misses use the original query.

`SurfaceDomain.refit()` advances an eight-byte device generation before updating
safe points, its hierarchy and quadrature. Surface positions may change only
with a refit, and topology is fixed at setup. Lookup requires the consumer's
saved generation to equal the current one. The counter saturates at INT64_MAX;
that value permanently disables reuse instead of wrapping onto an old cache.
No host input comparison or fresh-solve callback is required.

Current cell indices, coordinates, orientation, geometry, radius ratio,
retention thresholds and errors are evaluated on every invocation. Only the
surface-query result is reused. Consumers have separate caches so results do
not cross compiled calling contexts. Current results and previous snapshots
occupy separate buffers. After classification, device copies create the next
snapshot and a separate kernel rebuilds its hash table. Failed classifications
leave an empty table. A maximum of 128 linear probes bounds work: an uninserted
or unavailable key falls back to the original traversal. Duplicate centers use
the lowest stored row; a different cutoff can reduce reuse but cannot supply an
incorrect nearest result.

All arrays and kernels are prepared before the caller captures execution. Generation increments,
lookups, copies and table construction run on the caller's device stream.
Nothing is allocated or read back during execution. Seven consumers reserve
42,205,240 cache-array bytes at 4k sites, plus the shared eight-byte domain
generation. At 1k sites they reserve 10,325,560 bytes. These totals exclude
allocator overhead and existing mesher storage.

## Investigation

Instrumentation ran the original queries and recorded exact keyed repeats.
Across the first torus4096 solve and one fresh replay, ordinary retained
classification repeated 35.19% of 6,914,930 ray queries and 22.84% of 1,362,182
nearest queries with matching cutoffs. Full classification repeated 72.99% of
1,422,196 ray/nearest queries. Same-row repeats were scarce because cells are
reordered. Recovery selection repeated about 64–66% of ray queries on hand and
graded Armadillo. These diagnostic observations are not performance timings.

The first observer failed to compile a mutable loop flag and performed no solve.
Later observers passed; all outputs, sources and the original failure are
retained. Their hand trajectories include tiny coordinate/exterior differences,
so they are not claimed bitwise identical. Independent own-frame hull checks,
material topology/quality and explicit exterior audits establish their limited
equivalence.

The first cache prototype used reset callbacks. Its replacement uses the device
generation described above. Both passed original-kernel cold/warm comparisons
and independent complete meshes. Only the latter design was integrated.

## Production validation and observed variability

Runtime `d8c0cab8213eaf9b3bb0638fa97dbf7ebbf8cda3` passes all **1,113 tests**,
including 239 focused integration tests. Separate production comparisons cover
53 retained and 106 full/selection cases against the original kernels, cold and
warm, including extreme scales, strict cutoff ties, missing witnesses, malformed
input and inactive padding. Ten new tests include a 13-phase captured loop:
query permutations, surface deformation/refit, invalid input and recovery,
empty input, cutoff changes and overflow, followed by forced counter saturation.

All 24 valid final outputs independently pass exact CGAL original-coordinate
full-hull/orientation/empty-sphere checks, material quality/topology, boundary
vertex links, self-intersection checks and VTU round trips. The coarse hand
still returns error 64 and zero published counts on both revisions; it is an
error-path gate, not a valid-mesh speed result.

Seven hand pairs have small coordinate differences; the largest is
1.1641532182693481e-10 on seed42. Seed7 changes four exterior cells in each
direction, with unchanged material connectivity, boundary and work. Seed1
changes six material cells and two boundary faces, so the original strict
comparison **fails**. Its report and failed combined gate run remain in the
published evidence. The continued checker validates the remaining meshes
without silently accepting that mismatch.

[Issue #220](https://github.com/alecjacobson/warpower/issues/220) investigates the
seed1 result explicitly. The changed boundary is one quadrilateral diagonal
flip. Exact rational arithmetic proves a two-sided distance bound of
1.736417787783083e-18 between these two saved output boundaries and an exact
volume difference of approximately 4.44299e-21. This is a local saved-output
comparison, not an input-to-output Hausdorff guarantee. Sampled fit and feature
metrics match exactly. The 3,207 common material cells differ in radius ratio by
at most 2.22045e-16. Mean radius ratio changes from 0.8134324617780175 to
0.8133956439947354; the recorded minimum, maximum and 1st/5th/50th/95th percentiles
are identical. We do not claim identical material connectivity or mean quality.

Eight fresh baseline replays produce both alternatives (six of the first, two
of the second); eight candidate replays also produce both (two and six).
**All sixteen** saved outputs independently validate, have matching recorded
work, and retain their alternative's feature metrics. This establishes that
both alternatives already occur without the cache. It does not establish equal
output distributions or identify a unique source of floating-point variation.

A full seed1 trajectory audit evaluates each cached answer alongside a fresh
original query on the same current inputs. It observes **zero mismatches** over
1,155,336 ray and 662,598 nearest queries, including 812,578 ray hits and 541,084
nearest hits. Face IDs, distance bits and hit-point bits all match. Its complete
output independently validates. Removing the audit wrappers recovers the
production classifier AST. The final assessment accepts the explicitly
investigated baseline variability while retaining the failed strict claim.

## Clean complete-solve timings

Baseline `07aa4c78fc69fdeae468db457f455c953ca84c06` and the runtime above use the
same inputs, parameters and recorded work. Each value is the median of three
additional fresh complete graph replays on the L40. Every solve includes refit,
fresh sampling, fitting, growth, four quality rounds and extraction; setup,
compilation, upload, readback and export are outside replay. Refit invalidates
the cache before each fresh solve. All 36 timed jobs have clean quota windows;
independent CPU geometry checks ran afterward.

| Configuration | Baseline ms | Query reuse ms | Change |
|---|---:|---:|---:|
| Torus, 4,096 sites | 1,687.389 | 1,510.987 | −10.45% |
| Graded Armadillo, 4,096 sites | 3,460.057 | 2,579.852 | −25.44% |
| Counterbored plate, 1,000 sites | 966.711 | 920.418 | −4.79% |
| Hand seed7, 1,000 sites | 1,307.441 | 1,288.585 | −1.44% |
| Hand seed1, 1,000 sites | 1,035.037 | 1,036.078 | +0.10% |
| Cavity, 1,000 sites, coarse quadrature | 459.265 | 458.007 | −0.27% |
| Disconnected components, 1,000 sites, coarse quadrature | 729.268 | 730.469 | +0.16% |

The complete batch also includes hand seeds 0, 2, 42 and 2026 and seed7 with four
local retries. Torus, graded Armadillo, plate, cavity and disconnected outputs
have bitwise equal positions and equal canonical full/material/oriented-boundary
connectivity. Hand differences have the scope described above. The largest
measured healthy VTM slowdown is 0.165%; this batch does not establish statistical
significance for subpercent changes.

| Independent API | Baseline ms | Candidate ms | Change |
|---|---:|---:|---:|
| Delaunay, 1,000 sites | 9.1703 | 9.2045 | +0.373% |
| Delaunay, 100,000 sites | 182.3758 | 181.2706 | −0.606% |
| Delaunay, grid | 2.2073 | 2.1967 | −0.481% |
| Delaunay, fitted hand | 13.4075 | 13.4201 | +0.094% |
| Diagram, 100,000 sites | 18.1601 | 18.1220 | −0.210% |

Their kernels, outputs and diagnostics are unchanged. These control variations
are not attributed to the private cache. The main README retains the earned
Paragram adjacency flagship. The [native CPU comparison](cpu-size-frontier.md)
still shows a performance and quality gap.

## Memory, capture and reproducibility

Six memory jobs pass with zero errors: query-cache tests, a complete hand solve
with four local retries, and a producer/mesher/consumer failure-recovery loop,
each with padded allocations and with the pool disabled plus lifetime checks.
Native full and loop traces record 59,044 and 152,100 kernel activities. Each DOT
graph contains 1,826 copies, all device-to-device, with no HOST, MEM_ALLOC or
MEM_FREE nodes. The replay trace has no allocation/free/memcpy API calls and one
external final synchronization. Native activity records do not enumerate every
graph copy, so copy directions are also checked in DOT. Setup and the final
readback are outside the traced resident solve.

The loop reports errors `[0,1,0,0]`, validity `[1,0,1,1]`, proof enabled
`[1,0,0,1]`, zero invalid counts and successful recovery. Separately, the query
loop covers a changing surface and refit generation. All decisions remain on
device and can be captured with input producers and output consumers.

A warmed torus profile attributes about 210 ms to retained classification,
188 ms to hull support, 167 ms to group checks, 151 ms to boundary-sample
accumulation, and 130 ms to unproved-sphere checks. These are diagnostic costs,
not the clean timing comparison; substantial remaining costs span reconstruction
and boundary fitting.

The [manifest](../results/validation/vtm/surface-query-cache/manifest.json)
binds 660 artifacts, 230 runtime/test/example/benchmark source hashes, shared
inputs/helpers and both pinned commits. It retains observer failures, both
private cache designs, pilots, original/continued/final assessments, all repeat
meshes, paired-query instrumentation, memory logs and lossless native archives.
Published files are immutable; the verifier checks every hash and recorded raw
archive hash. Reproduction instructions and paths are in the manifest. The
final assessment is
[recorded explicitly](../results/validation/vtm/surface-query-cache/surface-query-cache-final-independent-v3.json),
including the failed strict comparison and accepted baseline variability.
