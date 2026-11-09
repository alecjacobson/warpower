# FP32 moment products with FP64 accumulation

This unadopted experiment separates product rounding from the rejected FP32
atomic sums in [#274](https://github.com/alecjacobson/warpower/issues/274).
It uses the existing mixed nearest-site query, computes boundary moment
products in FP32, and converts each product back to FP64 before accumulation.
Masses, moments, energy totals and acceptance calculations still use FP64 sums.
Production code does not import the experiment, and no public precision option
or default changes.

The local moment remains `weight * (sample - owner_site)`. Offsets are already
formed relative to the current site in the production method; this experiment
does not introduce centering. The proposal casts the FP64 weight and offset to
FP32 and multiplies them. A nonfinite product, lost nonzero component, or nonzero
component below `1e-37` selects the original FP64 product for that sample.
This range fallback does not undo ordinary FP32 rounding or provide a proof
that the resulting optimization trajectory matches the reference.

## Complete solves

L40 / Warp 1.19.0; median of three additional fresh captured solves. Both sides
use native surface queries, trusted input, mixed boundary queries and four
quality rounds. The scans use 8,192 sites, 256 initial sites, one surface/crease
subdivision, feature angle 120 and seed 2026. Other cases use 1,000 sites and
their gallery defaults. Setup, transfer, export and independent validation are
outside replay timing; their quota windows are checked separately too.

| Case | Reference ms | FP32 products ms | Time change | Minimum radius ratio, reference → products |
|---|---:|---:|---:|---:|
| Uniform Armadillo | 2323.211 | 2320.065 | −0.14% | 0.031019 → 0.031627 |
| Graded Armadillo | 2309.141 | 2376.436 | +2.91% | 0.060836 → 0.061863 |
| Torus | 466.862 | 446.803 | −4.30% | 0.092858 → 0.063767 |
| Cavity | 309.567 | 291.797 | −5.74% | 0.236810 → 0.201787 |
| Counterbored plate | 606.431 | 537.503 | −11.37% | 0.102747 → 0.152158 |

All ten meshes independently pass original-coordinate exact full-hull Delaunay
checks, positive material-cell/incidence checks, boundary vertex links, exact
boundary self-intersection tests and VTU round trips. All complete outputs
change coordinates and connectivity. The cavity also changes vertex roles.
Both scan solves succeed, unlike the earlier all-FP32 accumulation experiment.

These elapsed-time changes are **not quality-matched speedups**. The torus loses
about 31% of its minimum ratio and also worsens its lower tail, mean quality and
sampled surface fit. The cavity loses about 15% of its minimum ratio; two-sided
sampled RMS rises from 0.167783% to 0.174527% of the input box diagonal.
The plate improves minimum quality and sampled RMS (0.087018% → 0.086696%),
but its mean and first-percentile quality decrease slightly. The graded scan
accepts 33 selective steps instead of 15 and takes longer. The uniform scan
retains three selective steps and has little time change. Neither scan
establishes a broad speed improvement.

The variant remains an experiment. The results show that FP64 sums can preserve
successful complete solves while using cheaper products on these inputs, but
they do not qualify a broadly faster precision choice. #274 remains open for
other reductions and trajectory analysis. The main README flagship is unchanged.

## Local error and boundary-pass timing

Four retained 8K site states (frames 0, 2, 90 and 179) are tested with uniform
weights and with the gallery's radial graded weights. The graded-weight audit
reuses those same site states; it is not a recorded graded trajectory.
Each pair alternates timing order for 13 repetitions of a complete boundary
pass. Across all eight comparisons, nearest-site owners are identical, while
70–110 target sites change. The largest target displacement is `6.6744e-8`,
or `3.5653e-5` relative to the local nearest-site spacing. FP64 mass and energy
differences remain at reduction-roundoff scale.

Uniform-weight boundary passes change from 0.964–0.976 ms to 0.797–0.806 ms;
graded-weight passes change from 0.815–0.827 ms to 0.799–0.806 ms. The useful
local reduction does not predict the complete quality/trajectory result above.
All these timing windows have zero observed CPU-quota throttling.

## Scope and reproduction

Three focused tests pass normally, under padded memcheck, and with allocation
pools disabled plus full allocation-order tracking; both instrumented runs
report zero errors. They check product arithmetic and range fallback, owner and
corner behavior, and a captured current-input/failure/recovery boundary pass
with host allocation/readback/synchronization forbidden during replay. These
are focused checks, not full-mesher sanitizer or native-trace qualification.

An AST comparison confirms that the experimental accumulation kernel changes
only the moment-product operand. All production runtime files and the main
README are byte-identical to base `f6d698f`. No standalone performance gate is
claimed for this benchmark-only change. The tested source is `25bc544`.

```bash
python -m pytest -q tests/test_mixed_moment_products_experiment.py
python -m benchmarks.mixed_moment_products --products float32 \
  --report results/local/products-experiment.json \
  --case surface --input armadillo.off --n 8192 --initial 256 \
  --feature-angle 120 --subdivisions 1 --crease-subdivisions 1 \
  --query-backend native --no-validate-geometry \
  --boundary-query-precision mixed --quality-rounds 4 --repeats 3 \
  --output results/local/products-uniform
```

Use `--products baseline` for the paired control. The first plate command used
the unsupported case name `plate` and exited at argument parsing; the corrected
pair uses `counterbored-plate`. Both the failed invocation and completed follow-up
are retained. No failed solve is counted as a performance result.
Compact checks and measurements are in the
[evidence index](../experiments/mixed-moment-products/); raw data stay outside Git.
