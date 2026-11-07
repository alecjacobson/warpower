# Spatial radix surface queries

Surface queries now traverse a spatial radix hierarchy over groups of four
Morton-sorted triangles. The preceding balanced heap grouped triangles by equal
index ranges, which could leave large overlapping boxes. The new topology
reduces traversal work without changing triangle geometry, ray tolerances,
ambiguity retries, excluded-face handling or closest-triangle arithmetic.
Standalone power-diagram and Delaunay construction source paths are unchanged.

## Device construction and captured refit

Setup sorts the same triangle centroids and IDs as before. A Karras binary radix
tree uses the first Morton key of each four-triangle group; equal keys are
separated by sorted group indices. With `L = ceil(F/4)`, the root is 1, internal
nodes occupy `1..L-1`, and leaves occupy `L..2L-1`. One leaf uses root 1 directly.
Every node owns a contiguous range of groups.

Refitting first refreshes the existing balanced segment bounds. One additional
kernel independently gathers each radix node's exact float32 range minimum and
maximum from that segment tree. There is no inter-block publication protocol,
allocation, host geometry decision or transfer. Topology remains fixed across
refits; arbitrary deformations retain coverage but can reduce spatial efficiency.

Precomputed escape links encode the next subtree after a right-first depth-first
visit. Rays, triangle-contact validation and sampler box tests use these links
without local stacks. Closest queries retain near-first traversal with explicit
children and 128 stack entries. The 63 Morton bits and at most 31 varying index
bits bound the depth below 96. At tied nearest faces, traversal can select another
valid face, as permitted by the existing query contract.

The extra persistent arrays occupy `104 * ceil(F/4)` bytes: two bound arrays,
children, escape links, range endpoints, parents and group keys. Original segment
and sorting buffers remain alive. Setup adds three topology kernels; replay adds
only the range-bounds kernel. All triangle arithmetic remains in its caller's
compilation context. The rejected triangle-term caching experiments in
[#151](https://github.com/alecjacobson/warpower/issues/151) are not part of this change.

## Query pipeline measurements

Fresh L40 measurements use Warp 1.19.0, a 96×48 torus with 9,216 triangles,
100,000 fixed queries, three warm-ups and 15 timed replays. Baseline runtime is
`18b18f5`; the new runtime is `2c95331`, with benchmark correction `e08ecb1`.
Both sides execute the same enhanced benchmark script against their respective
source trees. Compilation, setup and host verification are outside replay.

| Captured operation | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Triangle hierarchy refit only | 0.020256 | 0.023424 |
| Signed closest-surface queries | 51.679359 | 27.734625 |
| Deformation, validated domain refit, queries and consumer | 64.834846 | 34.842815 |

The complete query pipeline is 1.861× faster. Allocator pool usage increases
from 14,030,991 to 14,270,607 bytes, exactly the 239,616 additional hierarchy bytes.
Warm-cache setup including capture takes 0.185 → 0.158 seconds in this pair;
these setup observations are not a compilation-speed claim or total process
memory peak. All replay timing windows report zero CPU quota throttling.

## Complete solves and validation

Each entry includes validated domain refit, fresh initialization, progressive
growth, fitting, quality stages and final extraction. The 1,000-site torus and
plate use five warm replays; the remaining pairs use three. All recorded replay
windows have zero CPU quota throttling. Setup, export and independent checking
are outside replay timing.

| Complete fresh solve | Before (ms) | After (ms) | Speedup |
| --- | ---: | ---: | ---: |
| Torus, 1,000 sites | 1,084.527 | 892.210 | 1.216× |
| Torus, 4,096 sites | 7,629.458 | 7,157.218 | 1.066× |
| Graded torus, 1,000 sites | 1,749.526 | 1,433.516 | 1.220× |
| Thin hand, 1,000 sites, quadrature 20/20 | 2,471.258 | 2,456.228 | 1.006× |
| Cavity, 1,000 sites, quadrature 20/20 | 1,020.852 | 1,023.354 | 0.998× |
| Disconnected, 1,000 sites, quadrature 20/20 | 727.457 | 724.533 | 1.004× |
| Counterbored plate, 1,000 sites, quadrature 16/16 | 1,630.359 | 1,586.271 | 1.028× |
| Armadillo, 4,096 sites, uniform sizing | 20,550.186 | 13,809.500 | 1.488× |
| Armadillo, 4,096 sites, graded sizing | 20,603.965 | 15,262.735 | 1.350× |

All nine pairs preserve full and material tetrahedron sets, oriented boundaries,
site roles and iteration work. Eight have bitwise-identical coordinates; the
thin hand's largest difference is `1.81e-18`. Both scanned outputs also match the
previously published gallery arrays exactly. The two coarse recovery controls
(cavity and disconnected components) remain valid, while coarse thin-hand failure
publishes zero counts as before. The small-surface timing shifts are recorded
without claiming a general speedup on those cases.

All eleven valid saved full hulls pass independent CGAL 6.2.1 exact orientation,
global empty-sphere, face-pairing and hull-support checks. NumPy checks verify
material quality, topology and single-cycle boundary vertex links. All nine
gallery VTU round trips reproduce their saved arrays, and CGAL finds zero
boundary self-intersection pairs in those exported surfaces. The current torus
also matches the original CPU-comparison GPU mesh bitwise.

The first new torus setup takes 138.547 seconds because dependent kernels require
fresh compilation; that time is excluded from replay and retained in the raw
record. A separate warm-cache repeat takes 1.468 seconds, versus 1.580 seconds in
the baseline pair. The plate's warm-cache setup takes 1.426 → 1.441 seconds.
Torus workspace pool usage increases by 59,904 bytes, matching its 2,304 triangles.

Standalone routines retain their preceding source paths. Nine-replay certified
Delaunay medians are 9.191 → 9.178 ms at 1,000 random sites, 183.966 → 182.554 ms at
100,000 sites, 2.196 → 2.204 ms for the grid tie case, and 13.413 → 13.417 ms for the
fitted-hand cloud. The 100,000-site diagram's 30-replay median is 18.153 → 18.063 ms.
These small differences are treated as timing variation, not gains attributed
to the surface hierarchy.

The focused suite passes 92 tests, including 49 new structural and query checks.
They cover fused/unfused callers, scales `2^-60`, `1`, `2^60`, six material domains,
triangle-boundary and excluded-face probes, independent winding and distance
oracles, single and partial leaves, repeated Morton keys, arbitrary deformations,
conservative sampler retention, and seven captured valid/invalid/recovery phases.
Ray results and closest distances match the old heap reference in these tests;
nearest-point ties can differ within floating-point roundoff.

An initial integration error used tuple assignment to initialize a mutable
split index from the range start. Warp's generated storage aliased those values,
omitting leaf groups. Separate assignments fix the error. The structural tests
require complete unique triangle coverage and exact descendant bounds, and the
independent closest oracle caught the missing geometry. No tolerances were relaxed.

The full repository suite passes 633 tests in 300.40 seconds. The 89 surface
and sampling tests also pass unfiltered padded Compute Sanitizer memcheck with
zero errors. Six complete-mesher checks pass in separate instrumented processes:
cube, torus, cavity, disconnected components, a changing-input producer/mesher/
consumer, and an outer device-controlled loop. All use 32-byte allocation padding
and a 128 KiB diagnostic stack limit; these settings are outside production.
The changing-input test takes 315.85 seconds under instrumentation. A separate
pool-disabled domain test passes API memory-access and full stream-ordered
allocation/lifetime tracking with zero errors.

The original combined mesher memcheck was deliberately interrupted after
30 minutes 34 seconds with two tests passed and a wait at device synchronization.
It reports no sanitizer errors, but is **incomplete**, not a pass. The separate
successful cases do not resolve this combined-run limitation;
[#155](https://github.com/alecjacobson/warpower/issues/155) tracks it with the
original log and a bounded control on the preceding runtime. That baseline
control also times out at device synchronization during site-hierarchy setup
after one completed case (120-second per-test limit, 178.18 seconds overall,
exit 1, zero reported memory errors). It is another incomplete result, not proof
of the underlying cause.

Native audits of three complete fresh torus solves and three changing-input
surface pipelines observe three graph launches and one final synchronization in
each trace. They find no host transfers, host callbacks or allocations/frees
inside the graphs. The full solve trace contains 98,985 kernel activities and its
DOT graph has 11,757 kernel nodes; the domain trace contains 102 activities and
34 kernel nodes. All observed copies and all DOT copy nodes are device-to-device.
The ordinary changing-input pipeline also rejects an invalid target and a
collapsed domain with empty outputs, then recovers on valid inputs.

CPU outperformance remains unachieved; these results do not establish
quality-matched CPU speedups. The main README retains its independently earned
Paragram adjacency result.

## Reproducing and inspecting the evidence

The [manifest](../results/validation/vtm/surface-radix/manifest.json) records
source revisions, source hashes and every retained artifact hash. Before/after
raw timing samples, setup/storage records and mesh snapshots are in its `timings`
subdirectory. The [independent checks](../results/validation/vtm/surface-radix/radix-independent.json),
[original torus comparison](../results/validation/vtm/surface-radix/radix-original-torus-comparison.json),
[full test log](../results/validation/vtm/surface-radix/radix-full-suite.log),
[full graph audit](../results/validation/vtm/surface-radix/native/full-native-audit.json)
and [domain graph audit](../results/validation/vtm/surface-radix/native/domain-native-audit.json)
provide the validation details. Compressed DOT and SQLite traces permit independent
inspection. Initial integration failures and the incomplete combined memcheck
are retained in `initial-failures`.

With installation complete, run the structural regression tests using:

```sh
pytest -q tests/test_surface_radix.py tests/test_domain.py tests/test_sampling.py
```

The retained `run-radix-pilot.py`, `run-radix-timings.py` and
`run-radix-scan-controls.py` scripts specify the exact benchmark parameters.
They are environment-specific drivers: set their workspace paths and create a
clean baseline worktree at `18b18f5` before reusing them. Both worktrees must run
the enhanced `benchmarks/domain.py` from `e08ecb1` by absolute path, with
`PYTHONPATH=.` selecting each working tree's library. Use one GPU workload at a
time and retain CPU quota measurements. The `commands` directory records the
original commands and exits, including unsuccessful diagnostic attempts.
