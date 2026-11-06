# Optional tetrahedralization validation

The optional `return_tetrahedra=True` output covers the full convex hull, including
tetrahedra whose dual vertices lie outside the ordinary diagram's clipping box.
The [API guide](api.md#optional-full-tetrahedral-mesh) explains connectivity,
weighted hidden sites, and numerical limitations.

These checks ran on 2026-11-06 with an NVIDIA L40 48 GB, Python 3.10.12,
Warp 1.19.0+cu12, NumPy 2.2.6, SciPy 1.15.3, and PyTorch 2.7.1+cu126.

## Default performance

Default calls do not import the optional module or allocate its buffers. The
existing `_build` function has an identical AST, and all pre-existing GPU source
modules are byte-identical to commit `72c679509bc7d3f6d3468ab12aa8dc0425dd7513`.
Instrumented calls confirmed identical GPU launch and allocation-call traces.

The table reports end-to-end mean times for resident CUDA Warp inputs, five
warmups followed by 30 paired repetitions with alternating execution order and
32 GiB memory-pool retention. All cases were free of CPU quota throttling.

| Default call | Before (ms) | With feature available (ms) | Change | GPU launches / allocation calls |
|---|---:|---:|---:|---:|
| 100,000 uniform, Voronoi | 12.718 | 12.747 | +0.234% | 30 / 39 |
| 100,000 five-cluster, Voronoi | 29.660 | 29.619 | -0.137% | 39 / 64 |
| 1,000,000 uniform, Voronoi | 101.792 | 101.792 | +0.000% | 42 / 64 |
| 1,000,000 uniform, power | 114.960 | 114.914 | -0.040% | 42 / 63 |

These differences show no measurable default-performance regression; the paired
95% confidence interval for the mean time difference includes zero in every case.
There is still a small Python flag check; this is not a claim of literally zero additional
CPU instructions. The new meshing work runs only when requested.

[Raw paired samples, source hashes, and quota counters](../results/validation/tetrahedra/default-performance.json)
are retained. Reproduce from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.default_path \
  --output results/local/default-performance.json
```

## Independent full-mesh comparisons

Unweighted meshes match SciPy Delaunay connectivity; weighted meshes match the
lower facets of the independent four-dimensional lifted convex hull. Comparison
ignores row and vertex order but requires every tetrahedron, with no extras or
duplicates. SciPy is used only for validation, never by the library.

| Sites | Mode | Tetrahedra | Missing / extra | Box expansions |
|---:|---|---:|---:|---:|
| 1,000 | Voronoi | 6,318 | 0 / 0 | 6 |
| 1,000 | Power | 6,320 | 0 / 0 | 6 |
| 10,000 | Voronoi | 66,504 | 0 / 0 | 5 |
| 10,000 | Power | 64,316 | 0 / 0 | 5 |
| 100,000 | Voronoi | 671,908 | 0 / 0 | 7 |
| 100,000 | Power | 667,413 | 0 / 0 | 7 |

Inputs use NumPy `default_rng(2026)`, uniform positions in `[-1, 1]^3` converted to
float32, followed by uniform float32 weights in `[-0.002, 0.002]` for the smaller
cases or `[-0.0002, 0.0002]` for 100,000 sites. Each case resets the generator.
The 100,000-site optional calls took 2.64 s and 2.51 s in single observations;
these include adjacency construction and any first-call JIT/module-load costs
and are not repeated steady-state benchmarks.

[Raw reference comparisons](../results/validation/tetrahedra/reference-comparisons.json)
retain all observations. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m benchmarks.tetrahedra \
  --output results/local/tetrahedra-reference.json
```

## Tests and memory checking

The [full suite](../results/validation/tetrahedra/pytest.txt) passed all 70 tests.
The 19 new tetrahedron tests include positive orientation, hull-volume equality,
face incidence, independent weighted and unweighted references, duplicate
ownership, chunking, noncontiguous PyTorch inputs on a nondefault stream, and
verification that default calls never enter the optional builder. Boundary cases
include a circumcenter far outside the initial box and weighted hull sites whose
cells are empty in the bounded diagram.

A built wheel also passed a CUDA smoke check with imports resolved outside the
checkout: default calls left the optional module unloaded, and an opt-in call
returned the expected tetrahedron.

[CUDA memcheck](../results/validation/tetrahedra/memcheck.txt) ran all 19 new tests
with zero errors and no kernel exclusions:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 compute-sanitizer \
  --tool memcheck --error-exitcode 99 python -m pytest tests/test_tetrahedra.py -q
```

The construction and validators use floating-point predicates. They do not
implement exact arithmetic or symbolic perturbation. Ambiguous cospherical or
copower configurations, unresolved near-coplanarity, and exhausted construction
limits raise an error; the cube-corner test verifies rejection of a nonsimplicial
case. Successful random-data comparisons do not establish universal robustness
for arbitrary degenerate inputs.
