# Surface-classifier register limits: no adoption

Warp's built-in register limits do not provide a useful broad improvement for
the currently expensive ordered full/selection classifiers. Limits of 96 and 128
regress several large cold batches by about 10–27%; 192 is essentially flat there.
Production kernels, public APIs and the README remain unchanged. This closes the
bounded probe in [#250](https://github.com/alecjacobson/warpower/issues/250), not
the wider query-optimization or comparable-quality CPU speedup goals.

The [ordered-query baseline](ordered-surface-queries.md) native uniform 8K trace
attributes 681.7 ms to ordered selection and 400.4 ms to ordered full classification,
both at 255 registers/thread with block size 32. These diagnostic times motivated
the experiment; register count alone does not diagnose an occupancy bottleneck.

## Experiment and measurements

Baseline main `0a2ce5e4ed48`, NVIDIA L40, Warp 1.19.0+cu12. Three private modules copy
the existing classifier arithmetic exactly by AST, changing only the
`cuda_max_registers` decorator option. A private installer selects kernel objects
before capture. Production ordering, cache, key generation and host launch code
are unchanged. The installed Warp API/native-library bytes match their package
RECORD; no Warp source or custom CUDA implementation is added.

The compiled kernels use the requested 96/128/192 registers/thread. Their reported
local memory is respectively 36,704–36,784, 36,496–36,552 and 36,240–36,296 bytes
per thread, versus 35,952–36,008 for the original three kernels. These properties
describe compiled resources; they do not prove a measured occupancy or bandwidth
change.

Each case is an immutable independently validated final mesh. Twenty warmed GPU
samples per row include key generation, sorting, queries and cache rebuilding;
cold rows also clear the cache. Warm rows repeatedly reuse unchanged input.
All 96 rows have clean CPU-quota windows, and every candidate's successful output
is bitwise equal to its original reference. Setup and host verification are
outside timing. These are static batches, not complete solves.

| Static cold batch | Mode | Original, ms | Limit 96, ms | Limit 128, ms | Limit 192, ms |
|---|---|---:|---:|---:|---:|
| torus1000 | full | 3.826 | 3.827 | 3.828 | 3.826 |
| torus1000 | selection | 3.822 | 3.821 | 3.823 | 3.821 |
| torus1000 | retained | 0.838 | 0.839 | 0.838 | 0.838 |
| torus16384 | full | 15.955 | 18.289 | 19.633 | 15.978 |
| torus16384 | selection | 12.179 | 15.423 | 13.655 | 12.163 |
| torus16384 | retained | 3.529 | 3.671 | 3.696 | 3.464 |
| armadillo-uniform8192 | full | 38.694 | 39.504 | 39.544 | 38.733 |
| armadillo-uniform8192 | selection | 18.455 | 20.909 | 20.892 | 18.438 |
| armadillo-uniform8192 | retained | 7.871 | 7.997 | 7.999 | 7.880 |
| armadillo-graded8192 | full | 35.527 | 39.020 | 39.051 | 35.530 |
| armadillo-graded8192 | selection | 18.184 | 19.434 | 19.428 | 18.113 |
| armadillo-graded8192 | retained | 9.141 | 8.535 | 8.543 | 8.775 |

The 1K torus stays below the ordering gate and uses the unchanged scalar path;
it does not exercise capped kernels during its timing rows. Warm-cache scan rows
can improve by roughly 6% with 96/128, but their small absolute costs and the cold
regressions do not establish a useful complete-solve change. Retained classification
has some static improvements, including graded 8K at 96/128. The motivating uniform 8K
native trace contains no executed retained-classifier kernel, so those observations
cannot be substituted for a gain in that full solve. This probe does not rule out
a separately justified targeted policy on other workloads.

## Validation and limits

All 19 existing ordered-query cases pass for each cap: 57 case executions, including
forced ordering at extreme scales, changing live counts/errors, nondefault streams
and inherited cache contracts. The first static harness mistakenly kept the old
two-variant loop labels and stopped with `KeyError` after one baseline row. It had
not timed or compared a candidate. That source, log and partial result remain;
v2 corrects only the phase iteration and reuses the unchanged hash-bound contracts.

No candidate was adopted or measured through a complete solve. There is no new
candidate native capture, sanitizer/lifetime or full-mesh geometry audit claimed.
The current production qualification remains unchanged. The
[manifest](../results/validation/vtm/surface-register-limits/manifest.json)
retains all variants, compiled properties, raw samples, source hashes, the failed
harness and the correction. [#197](https://github.com/alecjacobson/warpower/issues/197)
and [#40](https://github.com/alecjacobson/warpower/issues/40) remain open.
