# Exact coordinate-plane hull support

The private VTM completion checker can establish hull support directly when a
face lies exactly on a coordinate plane. This avoids traversing the original-site
BVH for every coplanar point. The previous interval checker already accepts
non-strict halfspace inclusion; interval uncertainty around exact coplanarity,
especially at nonzero coordinates, is what makes this shortcut useful.

The caller has just certified a positive current tetrahedron. Its oriented face
therefore has the opposite vertex strictly on its interior side. If all three
face vertices have exactly the same finite coordinate `s` on axis `k`, that
coordinate identifies the face plane. If the opposite coordinate is greater
than `s`, a current original-site lower bound `lo[k] >= s` proves support.
Otherwise `hi[k] <= s` proves support. Both bounds must be finite and ordered,
and the opposite coordinate must be finite and different from `s`. These are
exact comparisons of stored binary32 values; signed zeros represent the same
plane. A conservative bound can prevent acceptance but cannot incorrectly prove
support. Failure to prove support retains the existing filtered/exact traversal.

The positive-cell and current-bound preconditions are essential. This is a fresh
proof from current original coordinates, with no cross-frame trust or coordinate
perturbation. The optimization adds no buffers, host decisions or graph nodes.
Only `_proved_completion.record_face_support` changes its execution; independently
callable power-diagram and Delaunay constructors retain their existing paths.
The work is tracked in [issue #214](https://github.com/alecjacobson/warpower/issues/214).

## Complete-solve measurements

The NVIDIA L40 comparison pins runtime `677033b` against `92a8e2b`, using
Warp 1.19.0. Each result is the median of three additional fresh complete resident
solves, including domain refit, sampling, growth, fitting, quality and extraction.
Setup, uploads, exports and independent validation are outside replay. All 36
final jobs have clean observed CPU-quota windows. GPU jobs run serially, without
concurrent heavy CPU validation. Cases use 1,000 sites and 256 initial sites,
except the torus and graded Armadillo use 4,096 sites. Quality has four rounds;
coarse cavity/disconnected use 4/4 quadrature, and the hand uses 20/20.

| Complete solve | Before ms | After ms | Speedup |
|---|---:|---:|---:|
| Hand seed 2 | 2111.655 | 1817.480 | 1.16× |
| Hand seed 7 | 1447.023 | 1300.795 | 1.11× |
| Hand seed 7, four local retries | 1369.554 | 1238.585 | 1.11× |
| Counterbored plate | 1072.741 | 968.477 | 1.11× |
| Coarse cavity | 542.911 | 460.636 | 1.18× |
| Coarse disconnected | 745.758 | 729.865 | 1.02× |
| Torus, 4,096 sites | 1791.111 | 1791.547 | 1.00× |

Other healthy solves—hand seeds 0, 1, 42 and 2026, and graded Armadillo—change
by −0.54% to +1.04% in time. Coarse hand remains explicitly invalid in both
versions, with error 64 and zero public counts. Its timing is excluded from
valid-mesh speed comparisons. The torus has no exactly coordinate-aligned faces
in the observed trajectories and shows no timing benefit.

| Standalone routine | Before ms | After ms | Change |
|---|---:|---:|---:|
| Delaunay, 1,000 sites | 9.1704 | 9.2034 | +0.36% |
| Delaunay, 100,000 sites | 181.4861 | 182.9433 | +0.80% |
| Delaunay, 4³ grid | 2.2148 | 2.2078 | −0.32% |
| Delaunay, fitted hand | 13.4048 | 13.3933 | −0.09% |
| Diagram, 100,000 unweighted sites | 18.2102 | 18.2581 | +0.26% |

Standalone Delaunay uses nine repeats; the diagram uses 30 after five warmups.
Inputs, counts, status and diagnostics agree in all five pairs. The largest
observed slowdown is 0.81%, below the existing 2% investigation threshold. These
measurements show no material regression in this grid; they do not prove a
performance guarantee for every input.

## Correctness and scope

All 24 valid final meshes pass independent full-hull, exact empty-sphere and
orientation checks; material incidence and vertex-link checks; physical quality
and two-sided sampled distance measurements; exact VTU round trips; and boundary
self-intersection checks. Hand and plate also include sampled feature-chain
measurements. These samples are not Hausdorff or exact constrained-edge
conformity certificates.

All 12 valid pairs retain canonical full/material connectivity, oriented boundary
triangles, vertex roles and recorded rounds, steps, moves and rejection counts.
The five non-hand pairs have bitwise-identical positions. Hand pairs retain
small coordinate variations, with a maximum absolute difference of `5.82e-11`
at one seed-42 coordinate. Their exact changed values are retained. This does
not establish bitwise reproducibility or identify a causal kernel; the existing
[variation investigation](https://github.com/alecjacobson/warpower/issues/208)
remains open. The earlier four private pilot pairs have identical positions,
connectivity and recorded work; their results are retained separately.

The complete suite passes 1,102 tests. The new captured primitive test checks
1,039 cases against an arbitrary-precision rational determinant/AABB oracle,
covering all axes and signs, translation, extreme finite scales, signed zero,
on-plane points, one-binary32-step exterior witnesses, nearly aligned faces and
unsupported scalar inputs. It accepts 366 cases with no mismatches. A captured
certificate regression rejects inverted, nonfinite, empty, overcounted and flat
inputs before hull-support processing, and then accepts the restored frame.
Existing moving-bound tests cover refreshed inputs.

Separate full-trajectory audits run the old and new support kernels on identical
current inputs. Across hand seed 7, plate, cavity and torus4096 (including its
smaller prefix workspaces), 3,406 calls compare 3,299,800 face decisions with no
support-mask or error-state differences. The retained original kernel body
matches the pinned baseline AST. Applicability instrumentation finds 331,644
directly supported hand faces, 248,940 plate faces and 250,420 cavity faces;
all use nonzero coordinate planes. No torus face takes the shortcut. These
counts cover the first solve and one fresh replay, not only the final mesh.

Eight isolated Compute Sanitizer jobs pass with zero errors. The two new tests,
36 existing support/motion tests, a complete recovered hand solve with four local
retries, and the four-phase captured failure/recovery loop each pass padded and
pool-disabled allocation-lifetime checking. Instrumentation uses the existing
131,072-byte diagnostic stack allowance; ordinary timings and native traces use
the normal runtime configuration. No tests are excluded and no job reaches its
1,200-second limit.

Native first-launch traces of a complete solve and the four-phase loop show
only device-to-device graph copies, with no host callback or allocation/free
nodes and no allocation/free/memcpy API calls in the execution range. Both
record initial graph instantiation/upload and one external final synchronization.
The loop reports errors `[0,1,0,0]`, validity `[1,0,1,1]` and recovery enable flags
`[1,0,0,1]`; invalid input produces zero counts and the final frame recovers.
Nsight can omit individual graph-copy activities, so all 1,466 DOT copy nodes
are independently checked. The complete-hand profile records 381 support calls
and 153.26 ms of support-kernel activity; diagnostic profiling totals are
separate from the clean complete-solve comparisons above.

The wide comparable-quality native CPU speedup goal remains unmet. These are
GPU-to-GPU improvements on planar examples; the 4k torus remains about 1.79 s.
The README retains its earned Paragram flagship result.

The [evidence manifest](../results/validation/vtm/axis-hull-support/manifest.json)
records pinned sources, inputs, observations, prototypes, timing and independent
validation records, memory logs and native traces. Instrumented observations and
duplicate-decision audits are correctness evidence, not performance timings.
