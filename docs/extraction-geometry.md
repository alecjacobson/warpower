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
