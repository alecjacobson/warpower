# Bounded orientation in hull-support leaves

Private VTM hull-support leaf queries now use the existing bounded binary32 and
analytic binary64 orientation stages before the original Ball64/exact fallback.
The only runtime edit replaces one predicate call in `record_face_support`.
On NVIDIA L40, the complete 4,096-site torus solve falls from
2.030 s to 1.854 s
(8.7% less time).
This is a GPU implementation comparison; the
[quality-matched CPU outperformance goal](cpu-size-frontier.md) remains unmet.

## Scope and proof

The [existing bounds](bounded-predicate-proof.md) apply to the new consumer without
new constants or arithmetic. Only a strict sign and validity are consumed; a
rounded determinant magnitude is never a geometric output. Binary32 accepts
only finite original coordinates of magnitude at most 16. Unsupported or
uncertain cases continue through analytic binary64 and the original interval
and integer predicate, preserving exact zeros and invalid-input errors.

The hierarchy's conservative halfspace pruning, canonical-owner skips, partitioned
traversal, strided face tails, atomic exterior-witness cancellation, stack errors
and support-mask lifetimes are unchanged. A mask remains usable only after the
current certificate's error checks. The change adds no persistent arrays, kernel
launches or graph decisions. Original coordinates and IDs are preserved.

Standalone power-diagram and Delaunay implementations retain their source and
default execution paths. The source-isolation record checks every runtime file;
only the private leaf predicate call changes. The main README result is unchanged.

## Complete fresh solves

Runtime 6eb1882 is compared with fresh controls from main 9405684, using Warp
1.19.0. All 26 final measurement jobs have unthrottled recorded quota windows.
Complete-mesher records include first and timed replays. Each replay includes
domain refit, fresh sampling, growth, fitting,
quality stages, extraction and diagnostics. Setup, compilation, input upload and
offline validation are separate. The 1k torus uses five measured replays; the
other complete examples use three.

| Complete solve | Before, ms | After, ms |
|---|---:|---:|
| Uniform torus, 1,000 sites | 668.203 | 640.822 |
| Uniform torus, 4,096 sites | 2030.407 | 1854.146 |
| Graded torus, 4,096 sites | 2308.771 | 2115.329 |
| Thin hand, 1,000 sites | 2270.367 | 2266.039 |
| Cavity, 1,000 sites | 877.496 | 867.003 |
| Disconnected solids, 1,000 sites | 610.634 | 602.115 |
| Armadillo uniform, 4,096 sites | 3146.255 | 3110.940 |
| Armadillo graded, 4,096 sites | 6688.157 | 6664.022 |

The independent private pilots measured 667.989 → 640.997 ms at 1k and
2030.618 → 1854.757 ms at 4k; both outputs were bitwise equal to their controls.
All final meshes preserve full/material tetrahedron sets, oriented boundaries,
site roles, work counts, input arrays and parameters. Seven final pairs preserve
positions bitwise; the hand differs in two coordinates by at most
6.505e-19. The validation record lists both values.
Each full hull passes independent CGAL exact global checks. NumPy/AABB checks
evaluate material quality and surface fit, with single-cycle boundary-vertex
links. All six gallery outputs round-trip through VTU exactly and have zero
CGAL boundary self-intersection pairs.

The measured torus allocator-pool footprints remain 273,863,500 bytes at 1k and
1,743,416,484 bytes at 4k. These are workspace-pool measurements, not process-wide
peak usage. Setup and first-replay observations remain in the raw records; no
cold-start speedup is claimed.

## Standalone routines

Nine measured replays per Delaunay operation and thirty per diagram operation
give the following fresh comparisons. All timed quota windows are unthrottled;
the unchanged source and output diagnostics are recorded. Changes stay within
0.9%, and these small variations are not attributed to the VTM-only edit.

| Operation | Before, ms | After, ms |
|---|---:|---:|
| delaunay1000 | 9.197920 | 9.178400 |
| delaunay100000 | 182.145920 | 183.741638 |
| delaunay-grid | 2.210304 | 2.198912 |
| delaunay-fitted-hand | 13.453184 | 13.421856 |
| diagram100000 | 18.157298 | 18.111989 |

## Correctness, memory and capture

The private adapter passes 69 reconstruction tests and 36 support-mask cases.
The integrated focused suite passes **105 tests**, and the full suite passes
**831 tests**. New cases compare masks with independent rational plane signs and
errors with the original full checker; the private probe additionally compares
the old mask traversal. They cover moving exterior witnesses, exact coplanarity,
smallest-subnormal offsets, slanted planes, the range boundary and its next float,
scales 2^-60 and 2^60, translation, signed zero, duplicate-owner promotion,
empty/restored prefixes, early leaves, strided tails and blocks 32/256. Captured
checks forbid host allocation, readback and synchronization.

Six padded Compute Sanitizer jobs finish with zero reported errors: the new
masks, existing support tests and a routed 2,050-site producer/consumer case,
each with pooled padded checking and separate API-memory/lifetime checking.
Lifetime checks disable the diagnostic array pool; all use a 128 KiB diagnostic
stack. The two support suites use full allocation-lifetime tracking; the routed
case uses use-after-free tracking. Production pool/stack settings are unchanged.
The existing [conditional CUB allocation-order instrumentation limitation](cuda-validation.md#conditional-native-radix-sort-scratch)
is not claimed resolved.

The native full-solve trace records 61,070 kernel activities,
33,270 graph kernel nodes and 2,932 device-to-device
copy nodes. The separate eight-phase device producer/constructor/consumer loop
records 1,348 activities, 2,393 kernel nodes and
228 device-to-device copies. Each has one graph launch and one final
synchronization, with no observed host transfers, callbacks or allocation/free
graph nodes. All DOT copy directions are checked as well as recorded transfers;
Nsight can omit individual graph-copy activities. First launches include graph
instantiation/upload APIs. Six valid loop phases match independent SciPy meshes;
invalid count and nonfinite input publish empty results before recovery.

The full native profile's 534 `record_face_support` calls take
279.896 ms, versus 453.998 ms in the preceding unchanged-runtime
profile. The new kernel uses 255 registers per thread and block size
32. This diagnostic isolates the affected kernel; it is separate from the
clean complete-solve timing evidence above. Its traversal cost is not attributed
entirely to point predicates.

The [manifest](../results/validation/vtm/bounded-hull-support/manifest.json)
retains the private source, tests, commands, samples, meshes, independent checks,
sanitizer logs and native traces with verified hashes. Large text/trace artifacts
are losslessly gzip-compressed; decompress them to their original paths before
replaying a historical command that expects raw files. Shared inputs and helpers
are referenced by verified hashes. The first loop-audit harness failed before
capture because a filename-prefix substitution also changed `np.float32`. Its
source and failure log are retained; the corrected harness is syntax-tree
identical to the preceding audited loop and passes the native checks above.
The runtime was unchanged by that harness correction. Follow-up construction work is tracked in
[#186](https://github.com/alecjacobson/warpower/issues/186).
