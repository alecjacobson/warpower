# Fresh GPU neighbor construction against native RadFoam

This comparison fills the large-input gap in [the benchmark claim audit](../benchmark-claims-audit/README.md).
It measures the unchanged `b324c6e` geometry library; all 88 library source hashes match the prior exact-audit candidate.
The new work adds a reproducible benchmark and graph/profile evidence, with no geometry implementation change.

## Measured work and timing

WarPower and Paragram return bounded Voronoi neighbor CSR in original input order, using box padding 1.00001 and 1.0 respectively. Native RadFoam returns unbounded neighbor CSR plus tetrahedra and a permutation, in internally sorted order. Native-to-Warp ratios compare these construction calls, not identical outputs, incremental updates, tetrahedron construction alone, or fitting speed.

One L40, resident identical FP32 inputs, fresh construction in separate sequential processes. Three warmups and five synchronized measured calls per backend/case, except native Bicycle and Playroom: one warmup and three measured calls. That fixed long-case schedule was selected after a 174-second native Bicycle warmup, before collecting its timed samples; the preliminary warmup record is retained. Allocation included; preceding result destroyed before each call; JIT/upload/download/validation excluded. Warp allocator retains 32 GiB. Torch intra/inter-op and OMP/MKL/OpenBLAS thread counts are one. Both visible CPU quotas are four cores. All 86 primary measured calls have zero throttled periods; only the benchmark CUDA process was observed at each GPU snapshot. GPU events are elapsed spans, including idle gaps, not kernel sums.

All table times are medians. The [machine-readable results](results.json) also give means and ranges; the evidence archive contains every repetition.

| Input | Sites | Warpower (ms) | Paragram (ms) | Native RadFoam (ms) | Native / Warpower |
|---|---:|---:|---:|---:|---:|
| live-bonsai | 9,216 | 81.030 | 41.980 | 19.297 | 0.24× |
| uniform-1000000 | 1,000,000 | 100.278 | 94.698 | 4,780.673 | 47.67× |
| bonsai | 1,968,522 | 216.160 | 421.209 | 15,944.314 | 73.76× |
| garden | 4,064,598 | 427.782 | 705.516 | 27,063.019 | 63.26× |
| bicycle | 4,199,873 | 454.172 | 758.075 | 173,955.223 | 383.02× |
| playroom | 3,181,453 | 349.000 | 464.529 | 79,150.645 | 226.79× |

Paragram timings include calls with reported cell failures; use the correctness table below when interpreting them.

| Input | Warpower span (ms) | Paragram span (ms) | Native RadFoam span (ms) |
|---|---:|---:|---:|
| live-bonsai | 81.006 | 41.942 | 19.251 |
| uniform-1000000 | 100.253 | 94.659 | 4,780.555 |
| bonsai | 216.119 | 421.148 | 15,944.087 |
| garden | 427.748 | 705.460 | 27,062.646 |
| bicycle | 454.139 | 758.011 | 173,953.078 |
| playroom | 348.954 | 464.473 | 79,149.766 |

A separate 10-call Bonsai native preflight recorded CPU throttling in one call; all ten calls and its runner are retained. An initial Garden native window overlapped another agent's Warp tests and recorded CPU throttling; it is excluded. Bicycle's following run was interrupted during warmup when competing GPU activity was observed. Both records are retained. Primary results use the fixed windows described above after contention ends. Profiling is separate from primary timing.

## Correctness

The final output of each timing window passes structural CSR checks (offsets,
index bounds, native permutation bijection and unchanged inputs). Status counts
describe those final outputs; successful statuses alone are not a geometric proof.

| Input | Warpower failed cells | Paragram failed cells | Native tetrahedra |
|---|---:|---:|---:|
| live-bonsai | 0 | 143 | 56,674 |
| uniform-1000000 | 0 | 5 | 6,749,146 |
| bonsai | 0 | 613 | 13,165,611 |
| garden | 0 | 1,110 | 26,917,493 |
| bicycle | 0 | 1,477 | 27,659,640 |
| playroom | 0 | 611 | 21,305,667 |

Every directed edge of each large-scene native output was compared to the saved,
validated CGAL graph after mapping indices back to the original site order.
Warpower's current CGAL differences and all previously audited pairs were checked
against the recorded exact rational bounded-facet answers on identical inputs.
All current discrepancies are covered; shared unchanged edges outside those
queries remain uncertified. This reuses existing exact answers, rather than
claiming a new independent certificate of every edge.

| Scene | Native–CGAL directed differences | Warpower exact queries / errors | Paragram known-pair errors |
|---|---:|---:|---:|
| bonsai | 0 | 368 / 0 | 0 / 4 |
| garden | 0 | 1,182 / 0 | 2 / 2 |
| bicycle | 0 | 1,566 / 0 | 2 / 2 |
| playroom | 0 | 474 / 0 | 6 / 6 |

The small live frame retains the prior 172-query exact-audit scope. The uniform
control has structural/status checks only. Complete graph difference lists,
canonical directed-edge hashes and known-pair decisions are retained in the
archive. Differences between bounded graphs and unbounded CGAL are not counted
as errors without the facet test. Known-pair truth was previously rechecked for
both clipping boxes with identical answers. The known-pair test is only fourteen directions
across four scenes, not all of Paragram's incorrect or failed output.

## Where native Bonsai spends its time

A separate warmed Nsight capture measures **15.953 s** wall time and **15.946 s** of non-overlapping GPU kernel execution (**99.96%** of wall time). GPU tetrahedron growth accounts for 10.017 s and initial tetrahedron sampling for 5.670 s. Only 5.811 ms is outside recorded GPU kernel/copy/memset activity. The host spends much of its API time waiting in device-to-host count reads; those waits overlap the GPU work and must not be added to it. The roughly 16-second cost is therefore GPU construction work, not explained by CPU throttling.

The native sources are upstream RadFoam's geometry and Torch allocator, built as the existing headless extension (`-O3`, CUDA 12.6, `sm_89`). Viewer bindings are omitted; this is not a timing of the official training application. Native source revision, binary hash, build manifest and flags are retained.

The separate Warpower profiles are included in `results.json` and the evidence. Large-capacity retries remain serial per cell; this comparison does not implement cooperative 256/512-slot retries. Fitting runs remain deferred.

## The remaining live-frame bottleneck

Separate instrumented Warp profiles resolve the actual CUDA symbols for each construction specialization. GPU kernel sums below exclude host gaps and other kernels; all launches of each specialization are combined. Slot capacities are verified by the untimed pass diagnostics.

| Construction stage | Live frame GPU time (ms) | Large Bonsai GPU time (ms) |
|---|---:|---:|
| Initial FP32, 64 slots | 9.257 | 195.056 |
| FP32 retry, 128 slots | 8.952 | 7.752 |
| FP32 retry, 256 slots | 23.028 | 1.312 |
| FP32 retry, 512 slots | 33.942 | 0.000 |
| Cooperative FP64, 128 slots | 1.354 | 1.089 |

On the live frame, the serial FP32 256/512-slot passes consume **56.969 ms**, about **70.5%** of the 80.766 ms profiled call. The untimed status trace leaves 18 capacity failures after 128 slots, then just site 2021 after 256, and zero after 512. Two numerically difficult cells are recovered by the cooperative FP64 kernel earlier. The large-capacity work is still serial; adding cooperative capacity retries remains the next optimization, with the existing correctness and performance gates.

## Reproduction and evidence

Run [the benchmark](../../benchmarks/radfoam_fresh.py) separately for each backend. For example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --no-sync python -m benchmarks.radfoam_fresh \
  --backend radfoam --input data/radfoam/bonsai.npz \
  --module-dir /path/to/native-extension --warmups 3 --repeats 5 \
  --output /path/to/results/bonsai-radfoam.json
```

Use `--backend warpower` or `--backend paragram` for the other constructors. Omit `--input` and specify `--n 1000000 --seed 42` for the uniform control. `--profile` enables an external Nsight `cudaProfilerApi` capture range around the measured calls; keep profiling separate from primary timing.

See the [evidence manifest](manifest.json) for the [verified archive](https://github.com/alecjacobson/warpower/releases/download/evidence-geometry-updates-4ceb748/radfoam-fresh-20261113.tar.gz), complete command lists, raw samples, profiling captures, graph audits, source/build hashes and runners. Issue [#300](https://github.com/alecjacobson/warpower/issues/300) tracks benchmark qualifications; [#301](https://github.com/alecjacobson/warpower/issues/301) retains the capacity-recovery optimization work.
