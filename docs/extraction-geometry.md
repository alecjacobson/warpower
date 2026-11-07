# Stable geometry for material extraction

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

The extractor now tests the determinant against a conservative bound formed
from the six absolute triple products, using the margin from
[stable volume evaluation](stable-volumes.md). Reliable cases retain the
existing cross/dot, offset and radius expressions. Uncertain cases use bounded
integer arithmetic on the original float32 coordinates to compute the physical
determinant and circumcenter numerators and denominator.

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

The change adds no replay allocation, host decision or CPU geometry fallback.
It affects material extraction; the independently callable diagram and Delaunay
constructors retain their existing implementations.

## Independent checks

The isolated bridge is checked against Python rational arithmetic under all
24 vertex permutations and exact power-of-two scale changes. This includes an
uncertain case whose center exceeds the float32 range. An extraction fixture
combines that bridge with healthy interior material and verifies that the bridge
is excluded without aborting the healthy output.

A full captured 1,000-site disconnected trajectory now returns a valid mesh.
Its independent CPU mesh checker also needed a stronger orientation oracle:
ambiguous floating-point determinants are recomputed with rational arithmetic.
The checker still requires strictly positive physical volumes, unique cells,
proper face multiplicities and agreement with convex-hull volume. It never
clamps a determinant or skips a questionable tetrahedron.

The resident audit exercises valid, flat, nonfinite and restored input frames
with a device producer and consumer:

```sh
pytest -q tests/test_wide_center.py tests/test_extraction.py
python -m benchmarks.wide_extraction --dot results/local/wide-extraction.dot --output results/local/wide-extraction.json
```

The recovered coarse disconnected mesh has minimum radius ratio about 0.00235.
Its quality stage finds no accepted progress; worst-element improvement remains
tracked in [#37](https://github.com/alecjacobson/warpower/issues/37). The coarse
cavity's nonmanifold edge is unchanged and tracked separately in
[#125](https://github.com/alecjacobson/warpower/issues/125). Resolving arithmetic
failure does not establish topology recovery for every domain or a quality
guarantee.
