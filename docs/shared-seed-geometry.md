# Sharing current-input sphere geometry

VTM's internal previous-cell filter now computes a candidate tetrahedron's sphere
cofactors and conservative box once, then shares those values across its 32 query
lanes. Previously every lane recomputed the same geometry. One preparation kernel
writes fixed setup arrays before the existing conservative rejection and
exact/symbolic query loop. All float64 values and error bounds retain their full
precision; no tolerance, predicate or acceptance requirement changes.

Only `ReuseDelaunayWorkspace` enables `CertifiedSeeds(prepare_geometry=True)`.
The default seed filter retains its original kernels, and ordinary independently
callable Delaunay and power-diagram routines receive no new device storage or
launches. The existing seed-kernel ASTs are unchanged. This is current-input
preparation: every call refreshes every live unique cell after validation and
deduplication. It carries no geometric proof across changed inputs.

Storage is 80 bytes for sphere cofactors and 28 bytes for its box per candidate
slot. The 1,000-site torus workspace pool grows from 272,998,732 to 273,862,732
bytes, an 864,000-byte increase. At 4,096 sites, the several preallocated active
prefix constructors add 2,654,208 bytes (1,706,649,917 to 1,709,304,125).
These are allocator-pool measurements, not total process memory peaks. Setup and
first execution are reported separately; no setup-speed gain is claimed.

## Complete solves

Runtime `68e2386` is compared with `a84e274` on NVIDIA L40 / Warp 1.19.0.
Each replay includes domain refit, fresh sampling, progressive growth, fitting,
quality stages and extraction. Setup, compilation, transfer and offline checking
are excluded. Torus1k and plate use five measured replays; the other cases use
three. All admitted timing windows have zero CPU quota throttling.

| Complete fresh solve | Before (ms) | After (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 849.111 | 820.485 | 1.035× |
| Torus, 4,096 sites | 7,073.296 | 7,062.201 | 1.002× |
| Graded torus, 1,000 sites | 1,383.617 | 1,330.226 | 1.040× |
| Thin hand, 1,000 sites, quadrature 20/20 | 2,420.406 | 2,379.723 | 1.017× |
| Cavity, 1,000 sites, quadrature 20/20 | 1,002.496 | 977.371 | 1.026× |
| Disconnected, 1,000 sites, quadrature 20/20 | 708.345 | 687.997 | 1.030× |
| Counterbored plate, 1,000 sites, quadrature 16/16 | 1,528.667 | 1,493.025 | 1.024× |
| Armadillo, 4,096 sites, uniform | 13,731.923 | 13,715.952 | 1.001× |
| Armadillo, 4,096 sites, graded | 15,223.342 | 15,195.914 | 1.002× |

The 4k torus and scanned changes are below 0.2% and treated as timing variation.
The smaller complete examples improve about 1.7–4.0%. These measurements compare
two Warpower versions; the [wide CPU performance goal](cpu-mesher-baseline.md)
remains unmet.

All nine pairs preserve full/material tetrahedron sets, oriented boundaries,
roles and iteration work. Eight preserve coordinates bitwise; the hand's maximum
coordinate change is `5.29e-19`. All nine full hulls pass independent CGAL 6.2.1
exact orientation, global empty-sphere, face-pairing and hull-support checks.
Independent material-quality and single-cycle boundary-link checks pass. All
seven gallery VTU exports round-trip exactly and have zero CGAL boundary
self-intersection pairs. Both scanned meshes match their previously published
arrays exactly. The torus1k also matches the original CPU-comparison GPU mesh
bitwise; this does not make that CPU comparison quality matched.

[Independent checks](../results/validation/vtm/shared-seed-geometry/prepared-independent.json)
and the [original torus comparison](../results/validation/vtm/shared-seed-geometry/prepared-original-torus-comparison.json)
record these results without relaxed tolerances.

Standalone paired medians use nine Delaunay replays and thirty diagram replays:

| Standalone operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Certified Delaunay, 1,000 sites | 9.213792 | 9.195936 |
| Certified Delaunay, 100,000 sites | 181.743652 | 181.890976 |
| Certified Delaunay, grid ties | 2.200800 | 2.195648 |
| Certified Delaunay, fitted-hand cloud | 13.418816 | 13.402240 |
| Diagram, 100,000 sites | 18.129583 | 18.181490 |

All differences are below 0.3% and treated as timing variation. Source isolation
and the unchanged default seed kernels are recorded in the evidence.

## Tests and capture

The focused checks pass 62 tests. The original exhaustive seed tests now cover
both implementations, including duplicates, exact ties, scales from 2^-60 to
2^60, malformed cells/counts and restored prefixes. A new captured producer/filter/
consumer loop alternates a defining tetrahedron between scales 1 and 4 while
holding a query fixed. Reusing stale cofactors or boxes would give the wrong
acceptance on alternating phases. Both 32- and 256-thread launches pass.

The full suite passes **643 tests** in 278.15 seconds. All 25 seed/reuse cases
pass unfiltered padded Compute Sanitizer memcheck (642.97 seconds), and again
with memory pooling disabled, API memory-access checking and full stream-ordered
allocation/lifetime tracking (641.88 seconds). Both report zero errors. The
instrumented runs use 32-byte allocation padding and a 128 KiB diagnostic CUDA
stack; these are validation settings, not production changes. The separate
mixed-process full-mesher instrumentation delay remains tracked in
[#155](https://github.com/alecjacobson/warpower/issues/155).

The native trace records three complete solves, three graph launches and one
final synchronization, with 99,405 kernel activities and 11,769 DOT kernel nodes.
All observed copies and all 994 DOT copy nodes are device-to-device. There are
no host transfers, callbacks or allocation/free nodes. The ordinary five-phase
producer/mesher/consumer loop also passes valid, deformed, invalid-target,
collapsed-domain and recovered-input phases; invalid phases publish empty output.

Across the three profiled solves, the old checker costs 272.547 ms. The new
preparation costs 6.023 ms and its checker 186.340 ms, totaling 192.363 ms.
Each kernel executes 420 times. Preparation uses 92 registers/thread and no
local storage; the exact query kernel still uses 255 registers and 15,936 local
bytes/thread. The gain comes from avoiding repeated geometry computation, not
from reducing the query kernel's register count. These activity durations explain
the complete-solve result; they are not an independent end-to-end speedup.

[Full test log](../results/validation/vtm/shared-seed-geometry/prepared-full-suite.log),
[memory log](../results/validation/vtm/shared-seed-geometry/prepared-memcheck.log),
[lifetime log](../results/validation/vtm/shared-seed-geometry/prepared-lifetime.log),
[native audit](../results/validation/vtm/shared-seed-geometry/native/prepared-full-native-audit.json)
and [kernel attribution](../results/validation/vtm/shared-seed-geometry/prepared-full-native-kernel-summary.json)
retain the evidence.

## Alternatives retained in the record

The first split-checker prototype ran conservative bounds separately and marked
unresolved cells for exact fallback. It regressed complete torus replay from
850.995 to 875.886 ms; removing a pending-mask poll still took 876.076 ms.
Only 27,696 of 738,711 tested cells needed the second pass (3.75%), yet the
bounds-only kernel cost 272.661 ms and exact fallback added 76.452 ms across three
profiled solves. The original checker cost 272.547 ms in the preceding profile.
These are activity durations, not clean complete-solve times. Lower register
count alone did not establish a benefit.

A later variant combined shared preparation with a separate conservative
strict-inside rejection and uncertain-sign fallback. It passed 53 checks,
including 676 independently evaluated rational cofactor intervals, but took
825.167 ms versus 823.034 ms for preparation alone in the private fresh-torus
comparison. Its extra launches and storage provided no further demonstrated
benefit. Neither split variant is adopted.

Separate input-change diagnostics for [#130](https://github.com/alecjacobson/warpower/issues/130)
are also retained. Across torus, graded torus, hand, disconnected and plate runs,
only 6–10 cached reconstruction calls had wholly unchanged coordinates.
Unchanged-corner cells ranged from 16.1% to 47.2%, but that alone does not preserve
an empty-sphere proof: other sites and canonical ownership can change. The
diagnostic adds kernels and copies, so its times are not performance evidence.
No temporal proof cache is adopted here.

## Evidence and reproduction

The [manifest](../results/validation/vtm/shared-seed-geometry/manifest.json)
records runtime/baseline hashes, raw timing samples, commands, source hashes and
artifact hashes. The same directory contains independent checks, test and memory
logs, native traces, private prototypes and input-change diagnostics. Text meshes
and DOT/SQLite traces are losslessly compressed. Prototype scripts target the
recorded baseline revision, where their interception hooks were evaluated.

```bash
python -m benchmarks.mesher --n 1000 --repeats 5 --save-mesh torus.npz --output torus.json
python -m examples.vtm_gallery --case graded-torus --n 1000 --repeats 3 --output graded
python -m pytest -q tests/test_facet_seeds.py tests/test_proved_completion.py tests/test_reuse_delaunay.py tests/test_support_reuse.py
```

Tracked in [#157](https://github.com/alecjacobson/warpower/issues/157) and
[#158](https://github.com/alecjacobson/warpower/pull/158). The README retains the
independently earned Paragram result.
