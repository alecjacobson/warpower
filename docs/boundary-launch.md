# Boundary quadrature launch size

The boundary-assignment kernel now uses 32 threads per block. It retains the
same nearest-site traversal, smallest-index tie rule, float64 masses and relative
moments, scalar diagnostics, error handling and target construction. No arrays,
kernels or graph nodes are added. Only `BoundaryWorkspace._record` changes;
standalone power-diagram and Delaunay source paths remain unchanged.

## Choosing the change

The preceding complete torus trace attributes 362.939 ms across three solves to
boundary quadrature assignment. Active surface samples account for 361.463 ms
in 459 launches; inactive crease slots account for 1.476 ms. These are profiled
activity durations, not clean complete-solve timing. The original kernel uses
48 registers per thread. Contention on shared diagnostic scalars was a hypothesis,
so a private prototype replaced their atomics with block reductions while keeping
assignment and per-site mass/moment accumulation.

An alternating eight-variant experiment measures the complete captured
`BoundaryWorkspace.run()` on saved fitted 1,000-site torus and counterbored-plate
inputs. It includes site-hierarchy rebuilding, corner ownership, assignment and
target generation. Setup, compilation and host checking are excluded; three
warm-ups precede 21 measured samples per variant, with alternating order.
All timing windows report zero CPU quota throttling.

| Variant | Threads per block | Torus (ms) | Plate (ms) |
| --- | ---: | ---: | ---: |
| Original scalar atomics | 256 | 0.900000 | 1.478464 |
| Original scalar atomics | 32 | 0.625120 | 1.235488 |
| Original scalar atomics | 64 | 0.713088 | 1.367008 |
| Original scalar atomics | 128 | 0.768096 | 1.374048 |
| Block reductions | 32 | 0.645952 | 1.248480 |
| Block reductions | 64 | 0.720384 | 1.336768 |
| Block reductions | 128 | 0.779680 | 1.360704 |
| Block reductions | 256 | 0.906496 | 1.442176 |

The 32-thread original kernel is fastest on both inputs. Block reductions give
no consistent additional benefit, so that prototype is not adopted. Independent
chunked NumPy nearest-site, mass, relative-moment, energy and distance calculations
pass for every variant. Owners, roles, corners and float32 targets match exactly.
The private reduction prototype also passes 56 collective checks (including
partial blocks, zero contributions and broad finite float64 magnitudes) and all
11 existing boundary tests; those tests establish behavior, not a reason to
adopt an unhelpful optimization.

## Complete-pipeline gates

Runtime `3e33911` is compared with `1624d62` on an NVIDIA L40 with Warp
1.19.0. Each replay includes domain refit, fresh initialization, progressive
growth, fitting, quality stages and extraction. The torus1k and plate pairs use
five warmed replays; the remaining pairs use three. Setup, compilation, export
and independent host checking are outside replay timing.

| Complete fresh solve | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 892.213 | 850.579 | 1.049× |
| Torus, 4,096 sites | 7,153.856 | 7,073.942 | 1.011× |
| Graded torus, 1,000 sites | 1,433.697 | 1,383.957 | 1.036× |
| Thin hand, 1,000 sites | 2,457.784 | 2,420.655 | 1.015× |
| Cavity, 1,000 sites | 1,023.883 | 1,003.286 | 1.021× |
| Disconnected, 1,000 sites | 726.107 | 709.304 | 1.024× |
| Counterbored plate, 1,000 sites | 1,586.813 | 1,530.008 | 1.037× |
| Armadillo, 4,096 sites, uniform | 13,819.887 | 13,734.627 | 1.006× |
| Armadillo, 4,096 sites, graded | 15,228.608 | 15,216.712 | 1.001× |

The main torus and plate changes are 1.049× and 1.037× respectively. Scanned-case
differences are small and are not claimed as gains. No general CPU outperformance
is established. All admitted timing windows report zero CPU quota throttling.
The original after-torus4k window had two throttled periods (100,041 microseconds)
and the original before-graded-torus window had two (93,809 microseconds).
Both complete pairs were rerun; all four new measurements pass the quota gate.
The original pairs, command records and initial gate failure remain archived.

Changing launch size can change atomic accumulation order. All nine pairs retain
full and material tetrahedron sets, oriented boundaries, site roles and iteration
work. Six preserve coordinates bitwise. Maximum coordinate differences are
`2.40e-18` for the thin hand, `2.33e-10` for the cavity and `4.66e-10` for the
disconnected example. Both scanned meshes match the previously published gallery
arrays exactly. The new torus1k also matches the original CPU-comparison GPU mesh
bitwise; the CPU comparison remains a different-quality baseline, not a speedup.

All nine saved full hulls pass independent CGAL 6.2.1 exact orientation, global
empty-sphere, face-pairing and hull-support checks. Independent NumPy quality and
single-cycle boundary-link checks pass. All seven gallery VTU exports round-trip
exactly and have zero boundary self-intersection pairs under CGAL. No tolerance
was relaxed. The 1,000-site and 4,096-site torus allocator-pool measurements remain
272,998,732 and 1,706,649,917 bytes respectively; these are workspace pool usage,
not total process memory peaks.

The standalone source paths are unchanged. Fresh paired medians are:

| Standalone operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Certified Delaunay, 1,000 sites | 9.211200 | 9.181152 |
| Certified Delaunay, 100,000 sites | 182.114334 | 181.677765 |
| Certified Delaunay, grid ties | 2.195552 | 2.188064 |
| Certified Delaunay, fitted-hand cloud | 13.395424 | 13.385088 |
| Diagram, 100,000 sites | 18.138133 | 18.146941 |

The Delaunay gates use nine replays and the diagram uses thirty. All differences
are below 0.35% and are treated as timing variation.

## Tests and capture

The full suite passes 633 tests in 277.65 seconds. All 11 boundary tests pass
unfiltered padded Compute Sanitizer memcheck (1.00 second), and again with
allocator pooling disabled, API memory-access checking and full stream-ordered
allocation/lifetime tracking (1.56 seconds). Both report zero errors. These use
32-byte allocation padding and a 128 KiB diagnostic CUDA stack allowance outside
production. The combined full-mesher instrumentation limitation remains tracked
separately in [#155](https://github.com/alecjacobson/warpower/issues/155).

The full native trace has three graph launches and one final synchronization,
98,985 kernel activities and 11,757 DOT kernel nodes, matching the preceding
trace. There are no host transfers, host callbacks or allocation/free nodes;
all observed copies and every DOT copy node are device-to-device. All 918
boundary-assignment launches use 32-thread blocks. Active surface-assignment
activity falls from 361.463 to 241.336 ms across three solves; these profiled
activity durations support the measured full-solve change but are not an
independent end-to-end speedup claim. Changing-input producer/mesher/consumer
replays still reject invalid targets and collapsed domains with empty outputs,
then recover on valid inputs.

## Evidence and reproduction

The [manifest](../results/validation/vtm/boundary-launch/manifest.json) records
source revisions and hashes, raw timing samples, commands and artifact hashes.
[Independent checks](../results/validation/vtm/boundary-launch/boundary-independent.json),
[original torus comparison](../results/validation/vtm/boundary-launch/boundary-original-torus-comparison.json),
[full test log](../results/validation/vtm/boundary-launch/boundary-full-suite.log),
[memory log](../results/validation/vtm/boundary-launch/boundary-memcheck.log),
[lifetime log](../results/validation/vtm/boundary-launch/boundary-lifetime.log) and
[native audit](../results/validation/vtm/boundary-launch/native/boundary-full-native-audit.json)
provide the checks. Large text meshes and native DOT/SQLite traces are gzip
compressed with their uncompressed hashes retained. `initial-throttled` preserves
the rejected timing pairs, and `experiments` contains the unadopted reduction
prototype and fixed-input comparisons.

With installation complete:

```sh
pytest -q tests/test_boundary.py
python -m benchmarks.mesher --n 1000 --repeats 5 --output torus1000.json
```

The retained `run-boundary-timings.py` records all example and standalone settings;
`retry-boundary-timings.py` records the two paired reruns. These are environment-
specific drivers: adjust their workspace paths and create a clean baseline
worktree at `1624d62` before reusing them. Run one GPU workload at a time and keep
CPU-heavy checks outside timing windows. No performance claim should use a
throttled timing window. The README retains its independently earned Paragram
adjacency result.
