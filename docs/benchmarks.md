# Reproduction

The [complete L40 comparison](../results/summary/report.md) contains all 74 paired
cases, exported figures, failure and empty-cell counts, and memory measurements.
The underlying [CSV](../results/summary/measurements.csv) and
[JSON with provenance](../results/summary/summary.json) are available separately.

Run commands from the checkout using its Python environment. Install the optional
dependencies with `pip install -e '.[dev,datasets]' ninja`. To build the CUDA
reference, install a compatible CUDA toolkit and the pinned upstream source:

```sh
git clone --recursive https://github.com/zenseact/paragram ../paragram
git -C ../paragram checkout 4f36b1f2c8738912c5efc056a06023d4c1d2da07
git -C ../paragram submodule update --init --recursive
pip install -e ../paragram
python -m benchmarks.fetch_scenes
python -m benchmarks.matrix
python -m benchmarks.summarize --require-complete
```

The runner defaults to the local L40's CUDA architecture (`8.9`) and toolkit path
(`/usr/local/cuda-12.6`). Set `CUDA_HOME` and `TORCH_CUDA_ARCH_LIST` when needed.
The [environment record](../results/environment-l40.json) includes exact package
versions (including the CUDA 12 Warp wheel), GPU driver, reference compiler and
submodule revision, and the chunking environment used for the measurements.
Results are written under `results/{backend}-l40/matrix`; detailed logs and a
progress journal are under `.work/matrix`. The runner executes one GPU job at a
time and resumes matching completed cases. Changed results are archived.
The summary command writes CSV, JSON, a Markdown table, and standalone SVG plots
under `results/summary`. It checks input and harness compatibility, rejects mixed
Warpower implementations, and excludes archived measurements. Omit
`--require-complete` to inspect a clearly labeled partial matrix.

The matrix follows the [paper's benchmark definitions](https://arxiv.org/html/2605.06408v1#S5):
uniform, linear-gradient, five-cluster, and ten-cluster distributions at 0.1, 0.5,
1, 2, 5, 10, and 15 million sites. Power weights use Gaussian standard deviation
equal to the squared median nearest-neighbor distance divided by three. Nine
public Radiant Foam scenes are tested in both modes. Each case uses three warm-up
and ten measured runs. The original paper's GPUs differ from the L40 used here.

Synthetic seed 42 and extracted checkpoint hashes make inputs reproducible. The
public v1 checkpoint point counts agree with the rounded scene counts in the
paper; exact identity to the authors' evaluation files is not independently
verified. Geometry is extracted using `torch.load(weights_only=True)`. Large
checkpoints and point arrays are not included in the repository.

## Timing and memory

Timings are synchronized wall time for a fresh library construction call, with
GPU inputs resident. Construction allocation calls, hierarchy building, retries,
and CSR output are included. Input generation/upload, output download, and JIT
compilation are excluded. No precomputed geometry or hierarchy is reused.

The L40 matrix retains up to 32 GiB in Warp's normal CUDA allocator between calls;
PyTorch uses its normal caching allocator. This is recorded in every result.
The library itself does not change the application's global allocator policy.
For the measured repeated-call configuration:

```python
import warp as wp
wp.set_mempool_release_threshold("cuda:0", 32 * 1024**3)
```

Pass `--mempool-retain-mb 0` to measure Warp's default release policy. At 100k
uniform sites, an alternating diagnostic measured approximately 19.5 ms with
release-at-synchronization and 13.5 ms with 1 GiB retention. Every construction
still performs its own allocations.

The initial 1 GiB policy was too small for million-site scratch allocations.
An alternating one-million-site diagnostic measured about 151 ms with 1 GiB
retention versus 101 ms with 8 GiB, while the clipping kernel remained about
88–90 ms. At 15 million uniform sites, six repeated constructions measured
approximately 2.33 s with 1 GiB versus 1.73 s with 32 GiB (after initial warm-up).
These diagnostics preceded consistency recovery and are not the final matrix.
Retention does not preallocate geometry or scratch arrays. Chunking still uses
current OS-reported free memory conservatively; the 32 GiB diagnostic settled
at approximately 3.79 million cells per chunk on repeated calls.

Peak device memory is sampled through NVML in a separate process at approximately
1 ms intervals. It includes inputs, CUDA contexts, and cached allocator reserves,
and is an approximate device-wide peak. The separate process is necessary because
the CUDA reference holds Python's GIL during construction. Run on an otherwise
idle GPU. Device index 0 is used by both the benchmark and sampler.

Run CPU-intensive audits separately from timing. This VM exposes 128 logical
CPUs but its cgroup permits only four CPU cores; simultaneous audits can throttle
benchmark host code. Results now record throttled periods and time at each
visible cgroup CPU limit during measurement. The matrix will not reuse a
throttled result, and the summary rejects one. Historical results collected
before this check are preserved in archives and excluded from the final matrix.

## Correctness

Status counts accompany every timing and describe the final measured call.
Timing alone does not imply correct or
complete output. Small cases use an independent double-precision all-halfspace
oracle; it finds an interior point by linear programming and identifies faces
from Qhull's dual hull vertices. This is a floating-point oracle, not an exact
predicate implementation.

Save a graph with `benchmarks.run --save-diagram PATH`, compare unordered CSR
edges with `python -m benchmarks.compare A.npz B.npz --output comparison.json`,
and audit every disagreeing synthetic row with:

```sh
python -m benchmarks.audit A.npz B.npz --distribution cluster5 --n 100000 \
    --a-label paragram --b-label warpower --output audit.json
```

The audit considers every input site's halfspace for each audited cell. It checks
only rows where the implementations disagree; agreement elsewhere is not proof
that both are correct. Published audits retain per-cell discrepancies.
