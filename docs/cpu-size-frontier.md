# Native CPU comparison near 4,096 vertices

The subsequent [grouped-query GPU optimization](grouped-seed-queries.md) reduces
the same 4,096-site solve to 2.401 seconds with bitwise-identical positions and
matching connectivity. The table below retains the original 2.640-second GPU
record and native measurements; the CPU outperformance target remains unmet.

The complete GPU mesher still does not outperform CGAL Mesh_3/ODT. Near a
4,096-vertex budget, the GPU solve measured here takes **2.640 seconds**. The native
sequential CPU median is **1.280 seconds** and the four-thread median is
**0.736 seconds**. These medians retain all three samples, including samples
with CPU quota throttling. The CPU also has better measured lower-percentile
and median radius ratios and smaller boundary/volume errors; minimum-element
quality varies. This is a size comparison with quality tradeoffs, not a
quality-matched speedup result.

The GPU measurement is the preceding validated runtime `6a82244`; main
`336d25d` has the same runtime source. No new GPU timing was needed for this
native study. Its input is byte-equivalent to the
native reference's float32 round-trip torus: 1,152 vertices and 2,304 triangles.
Hardware, native source/build hashes, timing scope and initial small-case
comparison remain documented in [the CPU baseline](cpu-mesher-baseline.md).
The GPU is an NVIDIA L40 with Warp 1.19.0; the CPU is a Xeon Platinum 8362 at
2.80 GHz under a four-core quota. The native builds use CGAL 6.2.1 and GNU
C++ 11.4.0 with `-O3 -DNDEBUG`, sequentially or with TBB limited to four threads.

## Counts, quality and timing

The two native near-budget configurations use size 0.163, facet distance 0.025,
facet angle 25 degrees, feature angle 45 degrees and cell radius-edge ratio 2.
ODT has a 50-iteration cap, convergence 0.02 and freeze bound 0.01. All twelve
native runs in this study return `CONVERGENCE_REACHED` (code 3); perturbation
and exudation are disabled. CPU vertex counts include material-referenced
vertices; auxiliary exterior triangulation vertices are reported separately
in the raw records. The GPU uses its unchanged complete progressive schedule.

| Measurement | Warp | CGAL sequential | CGAL four threads |
|---|---:|---:|---:|
| Median complete seconds, all three samples | 2.63985 | 1.27978 | 0.73650 |
| Unthrottled samples | 3/3 | 2/3 | 1/3 |
| Median seconds, unthrottled subset only | 2.63985 | 1.27349 | 0.68106 |
| Referenced vertices | 4,096 | 3,932 | 4,063–4,112 |
| Minimum radius ratio | 0.05651 | 0.02295 | 0.04388–0.09121 |
| 1st-percentile radius ratio | 0.40896 | 0.44210 | 0.46665–0.47444 |
| 5th-percentile radius ratio | 0.64908 | 0.65847 | 0.67962–0.68085 |
| Median radius ratio | 0.88455 | 0.89474 | 0.89692–0.90097 |
| Signed relative volume error | −1.73349% | −1.12677% | −1.15967% to −1.13765% |
| Two-sided boundary RMS / input bbox diagonal | 0.08865% | 0.06235% | 0.06243–0.06356% |

The unthrottled subsets are disclosed to show the limitation; they are not
replacement clean three-sample batches. No CPU speedup ratio or universal
quality ordering is inferred. All sequential outputs are identical; parallel
construction yields distinct meshes. Every distinct output has independent
positive-volume, incidence, connected-component, boundary-Euler, radius-ratio,
dihedral and surface-distance measurements. The material topology is one
component with torus boundary Euler characteristic zero. RMS uses three
area-weighted quadrature points per triangle in both directions; it is not a
Hausdorff certificate. These native checks do not claim an additional exhaustive
boundary self-intersection or global empty-sphere audit.

Native time includes domain construction, feature protection, meshing and ODT.
File I/O, native triangulation validation, output serialization and independent
quality checking are separate. Quota windows cover the entire native process,
conservatively including those untimed activities. All native executions in a
batch finish before independent checking; no GPU benchmark ran concurrently.
GPU replay includes domain refit, fresh sampling, growth, fitting, quality
improvement and extraction. GPU allocation, compilation, upload and output
serialization are excluded. Its warm-cache setup is 2.927 seconds and first
graph execution 3.432 seconds, separately from the replay median. These are
not cold-start, transfer-inclusive timings.

## Retained size pilots and failures

Size 0.19 produced fewer sites than intended; size 0.16 overshot the target
slightly. Size 0.163 was selected from these measured counts, without tuning
CPU quality or stopping criteria to make it slower. Every sample is retained.

| Four-thread size criterion | Referenced vertices | Median seconds, all samples | Unthrottled samples | Clean-subset median seconds |
|---|---:|---:|---:|---:|
| 0.19 | 2,661–2,665 | 0.49845 | 2/3 | 0.49347 |
| 0.16 | 4,272–4,321 | 0.69425 | 1/3 | 0.69425 |
| 0.163 | 4,063–4,112 | 0.73650 | 1/3 | 0.68106 |

An initial preflight found an older binary with a different argument interface.
An erroneously ungated invocation nevertheless attempted that binary and exited
with code 2 before meshing. Its failure log is retained; it supplies no timing
sample. The actual study runner verifies the intended source and both binary
hashes against the published build record before each batch. It also verifies
the exact input positions and faces against the saved GPU output.

## Reproduction and remaining work

Build the references with [the native instructions](../benchmarks/cpu/README.md).
For the sequential near-budget result:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.cpu_mesher \
  --binary results/local/cgal-mesh3-sequential \
  --surface results/validation/vtm/cpu-mesh3-torus-input.off \
  --size 0.163 --facet-distance 0.025 --odt-iterations 50 \
  --convergence 0.02 --freeze-bound 0.01 --threads 1 --repeats 3 \
  --output results/local/cpu-torus-4k-sequential
```

Use the parallel binary and `--threads 4` for the parallel reference. The
recorded runs used the [archived adapter](../results/validation/vtm/cpu-size-frontier/run-cpu-frontier.py)
to select the already validated CGAL AABB distance oracle for offline quality
checking; the public command uses the default independent distance calculation.
Neither oracle runs inside native timing. The adapter's workspace-specific
paths and exact binary hashes are intentional provenance, not an installer.
The [manifest](../results/validation/vtm/cpu-size-frontier/manifest.json)
links every result, log, command, source/build reference and losslessly
compressed native mesh, plus the preceding GPU timing, mesh and quality record.
Decompress the `.txt.gz` files to reproduce raw native mesh hashes.

The wide CPU outperformance target remains open in
[#40](https://github.com/alecjacobson/warpower/issues/40). The new counts and
quality expose a substantial remaining gap even on the GPU's favorable
resident replay timing. This evidence changes no runtime code, standalone
power-diagram/Delaunay behavior, defaults or README flagship claim.
