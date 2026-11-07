# Stable geometry for extraction and quality improvement

A certified positive tetrahedron can be so flat that an ordinary float64
cross/dot calculation rounds its determinant to zero. That happened in the
1,000-site disconnected-components example: Delaunay certification succeeded,
but material extraction returned error 8 and the full mesher eventually returned
error 64 with zero output counts.

The isolated cell's exact determinant is `9.4565170861282e-26`. The ordinary
calculation produces zero and an infinite radius. Independent exact rational
arithmetic gives a circumcenter near
`[0, 4110418.4799999595, 5.487229979184842e-18]`. It is a thin bridge outside the
two components. Extraction should reject it by the ordinary material-selection
rule while retaining healthy material tetrahedra.

## Device calculation

The extractor and retained-cell quality selector test the determinant against
a conservative bound formed from the six absolute triple products, using the margin from
[stable volume evaluation](stable-volumes.md). Reliable cases retain the
existing cross/dot, offset and radius expressions. Uncertain cases use bounded
integer arithmetic on the original float32 coordinates to compute the physical
determinant and circumcenter numerators and denominator.

The same integer cofactors give stable face normals and total face area for
quality evaluation. Recovering only volume and radius is insufficient: the
finite tetrahedron `[(1,1,1), (0,e,0), (e,0,0), (0,0,e)]`, with `e=2^-100`, has
positive exact volume but ordinary float64 determinant **and face-area sum**
zero. Keeping that zero area would incorrectly clamp its radius ratio to one.
The fallback retains its extremely small physical quality instead.

Integer values are converted to physical float64 estimates before division.
The radius uses the local offset directly, avoiding cancellation from
subtracting an approximate global center. These are stable float64 metric
estimates; the surface query and extraction decision remain numerical geometric
calculations, not exact rational predicates or a correctly rounded float64
division guarantee.

Unlike an inserted site, an extraction center need not fit float32. The fallback
therefore keeps its center and radius in float64 and does not round the center
to a new site. Original positions and tetrahedron indices are unchanged.
Physically flat or inverted cells still fail explicitly. Nonfinite input is
rejected before integer geometry, and a failed extraction publishes zero counts.

Bounded Voronoi volume evaluation applies the same physical-orientation policy
and uses the stable local circumcenter offset for uncertain cells. A cell whose
four sites are on the hull contributes no bounded volume and needs no center
calculation. Signed circumcentric contributions and the existing reduction
cancellation diagnostic are otherwise unchanged. The disconnected trajectory
exposed all three consumers: fixing extraction alone left selection rejecting
the same certified cell and donor-volume validation failing.

The change adds no replay allocation, host decision or CPU geometry fallback.
The independently callable diagram and Delaunay constructors retain their
existing implementations.

## Independent checks

The isolated bridge is checked against Python rational arithmetic under all
24 vertex permutations and exact power-of-two scale changes. This includes an
uncertain case whose center exceeds the float32 range. A separate extraction
fixture checks the extreme needle's face area and quality against rational
geometry. Another fixture combines the bridge with healthy interior material
and verifies that the bridge
is excluded without aborting the healthy output; extraction and quality selection
agree. A second fixture encloses the four bridge sites while preserving its
distant empty sphere. Its four bounded dual volumes agree with independent
halfspace intersections and convex-hull volumes. The oracle tolerance is `1e-7`
for these cells extending millions of input units; ordinary-volume tests retain
their tighter tolerances.

A full captured 1,000-site disconnected trajectory now returns a valid mesh.
Its independent CPU mesh checker also needed a stronger orientation oracle:
ambiguous floating-point determinants are recomputed with rational arithmetic.
The checker still requires strictly positive physical volumes, unique cells,
proper face multiplicities and agreement with convex-hull volume. It never
clamps a determinant or skips a questionable tetrahedron.

The resident audit exercises valid, flat, nonfinite and restored input frames
through extraction, selection and bounded dual volumes with a device producer
and consumer. The dual connectivity is certified at setup. Invalid inputs clear
the public extraction/volume outputs, and the restored frame recovers:

```sh
pytest -q tests/test_wide_center.py tests/test_extraction.py tests/test_voronoi_volume.py tests/test_selective.py
python -m benchmarks.wide_extraction --dot results/local/wide-extraction.dot --output results/local/wide-extraction.json
```

The recovered coarse disconnected mesh completes all four quality rounds, with
15 selective steps and four accepted jitter moves. Its minimum radius ratio is
about 0.0690; the extraction-only intermediate fix stopped at 0.00235 because
the downstream geometry consumers still failed. This remains a bounded-budget
result, not a convergence or minimum-quality guarantee. Quality/time comparisons
remain tracked in [#37](https://github.com/alecjacobson/warpower/issues/37). The coarse
cavity's nonmanifold edge is unchanged and tracked separately in
[#125](https://github.com/alecjacobson/warpower/issues/125). Resolving arithmetic
failure does not establish topology recovery for every domain or a quality
guarantee.

## Measured validation

The [recorded comparison](../results/validation/vtm/wide-pipeline-summary.json)
uses an NVIDIA L40, Warp 1.19 and three warm complete resident solves per case.
Each solve refits the input domain and sizing and starts fresh sampling. Setup,
uploads, downloads, export and independent checking are outside these timings.
The baseline is `354b31e`; the final runtime is `1804b64`. Raw records, saved
meshes and source hashes are listed in the
[artifact manifest](../results/validation/vtm/wide-pipeline-provenance.json).

| Case | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Torus, 1,000 sites | 2,226.517 | 2,229.850 |
| Torus, 4,096 sites | 8,610.172 | 8,619.411 |
| Graded torus, 1,000 sites | 3,717.012 | 3,720.915 |
| Thin hand, 1,000 sites, quadrature 20/20 | 3,364.462 | 3,365.609 |
| Cavity, 1,000 sites, quadrature 20/20 | 1,419.838 | 1,417.110 |
| Disconnected, 1,000 sites, quadrature 20/20 | 1,239.835 | 1,240.364 |

All six median differences are below 0.2%; they do not establish a speedup.
Cell sets, oriented boundary faces and site roles agree before and after.
The hand has small coordinate differences; the other five cases have bitwise
identical positions. Every saved final full-hull mesh passes independent CGAL
6.2.1 exact orientation, global empty-sphere, hull-support and face-pairing
checks. Independent NumPy calculations verify material quality and topology;
gallery VTU exports match the saved connectivity and coordinates.

The recovered coarse disconnected case (quadrature 4/4) takes 11,742.734 ms and
produces 4,324 material tetrahedra, two components and two closed shells, each
with Euler characteristic two. Its minimum radius ratio is 0.06904 and its
two-sided sampled surface RMS error is 0.3367% of the input bounding-box diagonal.
All extraction, selection, jitter, interior and bounded-dual error flags are zero.
The preceding main revision fails this case, so it provides no valid timing
comparison. This complete four-round quality solve also does more work than
the earlier extraction-only fix.

Standalone Delaunay medians change from 9.156 to 9.191 ms at 1,000 sites and
183.757 to 182.798 ms at 100,000 sites. The 100,000-site diagram changes from
18.141 to 18.138 ms. Those implementations and the README are unchanged.
CPU outperformance remains unachieved; these are comparisons with the preceding
Warpower revision.

The final suite passes **517 tests**. Four isolated regressions pass unfiltered
padded Compute Sanitizer checking with zero errors. The four-phase resident
producer/consumer audit also passes unfiltered allocation/lifetime checking.
Its native trace contains 152 kernel activities, 108 memsets, four graph
launches and one final synchronization; the graph has no host copies, host
callbacks, allocation or free nodes.

The full 1,000-site quality trajectory's memory audit is **incomplete**. An
unfiltered run and a run instrumenting only the changed kernels were deliberately
stopped after more than 15 and 10 minutes, respectively. Neither completed or
counts as a pass. Both logs and their scope/stop records are preserved in the
manifest; [#129](https://github.com/alecjacobson/warpower/issues/129) tracks the
remaining audit. The ordinary full trajectory and the isolated memory checks
above did complete.
