# Selective predicate recovery experiment

An experimental candidate corrects all seven previously identified neighbor
pairs in both directions on the unchanged full Bonsai, Garden, Bicycle and
Playroom inputs. It also matches rational positive-area facet enumeration for
all 35 cells in seven reduced five-site fixtures; the baseline fails three of
those seven reduced graph fixtures.

**The candidate is not ready for adoption.** A subsequent 20-repetition run
already finds a **3.01% slowdown on 100,000 weighted uniform sites**, with a paired
95% interval of **[+2.30%, +3.74%]**. This exceeds the per-case 2% regression limit.
The full 74-case matrix is continuing. The production geometry code remains at
its baseline implementation in this checkpoint.

## Implementation under evaluation

- A cheap FP32 cancellation screen records the original site IDs supporting an
  uncertain clipping decision. A separate GPU kernel checks the original
  coordinates using the existing filtered/exact predicates. Buffer overflow or
  unresolved uncertainty goes to recovery.
- A bounded recovery pass uses 32 cooperating lanes per difficult cell, with
  parallel vertex classification, compaction and new-vertex calculations. It
  checks boundary connectivity before accepting the cell; unresolved cells use
  the existing retry pipeline.
- Capacity failures can grow the FP32 buffers without automatically promoting
  every cell to FP64. Precision failures have a separate dispatch. Cached FP32
  Cartesian positions occupy three floats instead of four.
- The integrated variant also connects these checks to captured
  `DiagramWorkspace` execution. It passes 66 existing cell, diagram, API,
  filtered-predicate and graph tests.

The cancellation threshold is a heuristic, not a proved forward-error bound.
Checking the recorded predicates and the selected failing pairs does not certify
every edge in a large scene graph. Recovery still includes FP64 clipping.

## Preliminary timing screen

L40; resident identical inputs; three warmups; ten alternating paired,
synchronized wall-clock repetitions including construction allocations. All
recorded timing windows were free of CPU throttling. Inputs match the historical
seed-42 matrix hashes. Values are means; no aggregate improvement overrides a
per-case regression. The higher-repetition failure above supersedes any inference
that the ten-repetition screen established a performance pass.

| Sites | Distribution | Voronoi baseline / candidate (ms) | Power baseline / candidate (ms) |
|---:|---|---:|---:|
| 100,000 | uniform | 12.739 / 12.772 | 14.014 / 14.381 |
| 100,000 | gradient | 12.812 / 13.158 | 13.916 / 14.273 |
| 100,000 | cluster5 | 29.292 / 20.788 | 30.794 / 22.642 |
| 100,000 | cluster10 | 39.414 / 24.147 | 40.600 / 28.771 |
| 1,000,000 | uniform | 102.158 / 101.795 | 115.013 / 107.920 |
| 1,000,000 | gradient | 103.213 / 96.954 | 116.005 / 109.378 |
| 1,000,000 | cluster5 | 119.350 / 117.386 | 128.624 / 131.299 |
| 1,000,000 | cluster10 | 136.265 / 117.798 | 149.518 / 127.318 |

The table uses the `candidate-screen` source patch. The `candidate-integrated`
patch adds captured-workspace integration and is the version used for the 66-test
run. Full per-repetition samples, input hashes, diagnostics and confidence
intervals are in [timing-screen.json](timing-screen.json).

## Reproduction and evidence

Both patches apply to baseline
`655412e84ad7c2c5581f8f26e1e9c41520fff2f8`; applying each to a fresh source copy
reconstructed every recorded Python source hash. The
[evidence archive](https://github.com/alecjacobson/warpower/releases/download/evidence-geometry-updates-4ceb748/robust-neighbor-recovery-screen-20261112.tar.gz)
contains the patches, source manifests, timing samples, selected-pair and reduced
case validations, exact reference program and test log. A fresh download matched
its SHA-256 and all 35 archived members. See [manifest.json](manifest.json) and
[remote-verification.json](remote-verification.json).

The reduced fixtures preserve the original float32 coordinates and source site
IDs. Run `python experiments/robust-neighbor-recovery/exact_reference.py` to
independently verify their expected graphs using rational arithmetic.

Sanitizer checks, broader forced-recovery tests, the full performance gate and
cleanup remain outstanding at this checkpoint. Follow
[#300](https://github.com/alecjacobson/warpower/issues/300) and
[PR #299](https://github.com/alecjacobson/warpower/pull/299). Scene fitting remains
deferred until the geometry work is settled.
