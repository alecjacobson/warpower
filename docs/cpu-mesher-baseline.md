# Native CPU comparison: initial torus result

The complete GPU mesher has **not met the CPU speedup goal**. The first native
comparison establishes a reproducible baseline and a substantial remaining
performance gap. Both implementations receive the identical float32 torus
surface, but their point counts, stopping rules, boundary constraints and quality
differ. These are not quality-matched time-to-target results. Subsequent GPU
optimizations reduce the complete 1,000-site solve to **0.737 s** at runtime
`0590626`, preserving the original coordinates and tetrahedron sets. See the
[latest paired GPU measurement](analytic-facet-search.md) and
[original-mesh comparison](../results/validation/vtm/shared-seed-geometry/prepared-original-torus-comparison.json).
The table below retains the original CPU/GPU experiment; its CPU measurements
have not been replaced by a new quality-matched comparison.

The input has 1,152 vertices and 2,304 triangles, exported with exact float32
round-trip coordinates ([OFF](../results/validation/vtm/cpu-mesh3-torus-input.off)).
The GPU is an NVIDIA L40 using Warp 1.19.0. The CPU is an Intel Xeon Platinum
8362 at 2.80 GHz, under a four-core CPU quota. Native release builds use GNU
C++ 11.4.0, CGAL 6.2.1, `-O3 -DNDEBUG`, and either sequential code or TBB limited
to four threads ([source/build hashes](../results/validation/vtm/cpu-mesh3-build.json)).

| Measurement | Warp, 1000 requested sites | CGAL, sequential | CGAL, four threads |
|---|---:|---:|---:|
| Median complete time | 3.635 s | 0.238 s | 0.172 s |
| Output vertices | 1000 | 855 | 891–895 |
| Material tetrahedra | 4225 | 2975 | 3128–3165 |
| Minimum radius ratio | 0.09286 | 0.07194 | 0.04085–0.07348 |
| Median radius ratio | 0.87442 | 0.83567 | 0.83495–0.84017 |
| Relative volume error | −5.089% | −2.655% | −2.632% to −2.562% |
| Two-sided boundary RMS / input bbox diagonal | 0.2562% | 0.1398% | 0.1348–0.1381% |

The GPU has a higher minimum and median radius ratio here; the CPU meshes have
smaller boundary and volume errors. A single speedup ratio would hide these
tradeoffs. All meshes have positive tetrahedra, one material component, a closed
boundary and torus boundary Euler characteristic zero under independent checks.
Boundary RMS uses three area-weighted quadrature samples per triangle in each
direction; it is not a Hausdorff bound.

Each time is the median of three fresh solves. GPU replay starts from fresh
samples and includes domain refit, progressive growth, fitting, four quality
rounds and extraction; allocation, compilation and transfer are outside replay.
Native time includes domain construction, feature protection, initial meshing
and ODT. File I/O, native validity checking and output serialization are reported
separately. Independent NumPy quality checks run after timing.

CGAL uses size 0.32, facet distance 0.025, facet angle 25 degrees, feature angle
45 degrees and cell radius-edge ratio 2. ODT has a 50-iteration cap with the
shipped convergence 0.02 and freeze bound 0.01. The sequential runs stop with
all vertices frozen; the parallel runs stop at convergence. Perturbation and
exudation are disabled. The GPU uses its documented full-pipeline defaults and
seed 2026. This experiment does not force equal iteration counts or assert that
these settings are optimal for either implementation.

The [final sequential samples](../results/validation/vtm/cpu-mesh3-torus-sequential.json),
[parallel samples](../results/validation/vtm/cpu-mesh3-torus-parallel.json), and
[GPU timing samples](../results/validation/vtm/amplitude-after-torus1000.json)
have zero CPU quota throttling. The third sample in the
[initial sequential batch](../results/validation/vtm/cpu-mesh3-torus-sequential-initial.json)
was throttled, so the full batch was repeated; the original record is retained.
TBB construction produces small output differences across runs; each distinct
mesh receives its own independent measurements. The GPU
[quality record](../results/validation/vtm/cpu-mesh3-gpu-quality.json) verifies
that the independently checked gallery mesh and the current timed output have
identical points and canonical connectivity.

Build and reproduction instructions are in
[the native benchmark README](../benchmarks/cpu/README.md). The Python runner
records every sample, input/binary hashes, parameters, quotas and independent
quality. It explicitly reports unthrottled sample counts and medians.

The [progressive/fixed-site schedule study](quality-frontier.md) completes #37's
within-Warp comparison: an optional fixed-site schedule takes 0.628 s on the
original torus with better measured quality and fit than the progressive default,
but does not improve every metric on the other tested inputs. This remains
separate from native CPU quality parity. Remaining work is tracked in #40 (wide
native CPU outperformance), #43 (complete trajectories) and #98 (cooperative
reconstruction). The main README retains the independently
earned Paragram adjacency result; this baseline adds no VTM performance claim.
