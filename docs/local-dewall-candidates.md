# Certified Local DeWall candidate experiment

The private prototype in [#186](https://github.com/alecjacobson/warpower/issues/186)
improves isolated Delaunay reconstruction on several tested distributions. Its
complete VTM gains are smaller and it uses more memory. **It is not adopted:**
all production modules, public defaults and the main README retain their contents.
The quality-matched wide CPU speedup goal remains unmet.

The reference is Gao and Chen's 2026
[Local DeWall](https://doi.org/10.1016/j.cad.2026.104107), evaluated here through
the publisher abstract and the authors' [MIT implementation at
f6292bb41215](https://github.com/WuhengGao/Local-DeWall/tree/f6292bb41215437bee4ce77a69baa785713d6140).
The full paper was not obtained, and the original CUDA implementation was neither
built nor benchmarked. Its reported performance is not a Warpower measurement.
Source URLs, hashes, licenses and the inspected files are retained in the
[reference bundle](../results/validation/vtm/local-dewall/references/local-dewall-f6292bb41215/provenance.json).

## Construction and acceptance

The wall traversal follows a cross-section of a site's incident tetrahedra using
two advancing endpoints. Coordinate comparisons encode the side of the cut.
For a nondegenerate tetrahedron with geometrically ordered vertices a<b<c<d,
the output rule assigns it to b: one other vertex is smaller and two are larger.
This differs from the current cell builder's smallest-input-ID ownership.
Combining their per-owner outputs without reconciliation would leave gaps or
duplicates.

The prototype uses two kernels instead of the source's lane shuffles, the current
site BVH instead of its grid search, and fixed candidate buffers. Initialization
and walking each have a 256-iteration budget; traversal has a 64-entry stack,
a bounded node-visit budget, and a 64-entry repeated-edge guard. Arrays and
conditional graph structure are allocated during setup. Current canonical owners
are rebuilt before every attempt. Original site IDs and physical positions remain
the output's coordinate system.

Normalized float32 points, approximate sphere queries and finite artificial
vertices are **untrusted candidate arithmetic**. The finite initialization can
omit hull tetrahedra, and float32 searches can produce invalid cells. Closure
flags and successful walks do not establish a complete Delaunay mesh. Individual
failed walks may leave useful candidates; they never establish acceptance.

The final experimental wrapper, `local_dewall_integration_v2.py`, sends raw
candidates through the existing exact original-coordinate symbolic seed filter.
The existing bounded facet completion retains only proofs established against
these current points and owners, repairs missing tetrahedra, and checks positive
orientation, topology, coverage and hull support. Filter errors, candidate
overflow, an empty proposal or exhausted completion select the ordinary device
constructor. Final failure keeps the usual zero-output-count contract.

The wrapper still duplicates some work: the outer Delaunay workspace certifies
an accepted completed mesh again. It also allocates cleanup arrays that the raw
candidate route no longer uses. [#188](https://github.com/alecjacobson/warpower/issues/188)
tracks removing those costs while publishing matching current topology metadata
and preserving fallback, stable addresses and independent APIs.

## Reconstruction pilots

These fresh-process comparisons use production runtime `6eb1882d8af1`, checked
out at `0ea739e1fbb6`, Warp 1.19.0 and an NVIDIA L40. Each entry is the median
of five warmed full reconstruction replays, including hierarchy construction,
exact filtering/completion and any fallback. Setup, compilation, upload and
offline validation are excluded. All timing windows report zero CPU throttling.

| Input | Current default ms | Private candidate ms | Ratio |
|---|---:|---:|---:|
| Uniform, 1,000 | 9.195 | 4.248 | 2.16× |
| Uniform, 4,096 | 12.080 | 8.085 | 1.49× |
| Flattened, 4,096 | 26.883 | 15.323 | 1.75× |
| Five clusters, 4,096 | 16.123 | 10.806 | 1.49× |
| Uniform, 32,768 | 73.786 | 56.786 | 1.30× |
| Cospherical 4³ grid | 2.201 | 2.210 | Approximately unchanged |

The control uses current defaults, including fast cell candidates at 32,768
sites. The private variant disables that alternative to isolate wall construction.
Both use the ordinary cached strategy policy. The grid rejects wall completion
and uses the existing symbolic path; its warmed timing does not measure that
initial rejection. All paired outputs have equal canonical cells. Non-grid
results also match SciPy; the grid matches the common symbolic control because
SciPy chooses a different cospherical triangulation. See the
[commands and raw timing records](../results/validation/vtm/local-dewall/local-dewall-reconstruction-v2/commands.json)
and [independent comparisons](../results/validation/vtm/local-dewall/local-dewall-reconstruction-independent.json).

An earlier direct-completion variant omitted the exact seed filter. It rejected
invalid candidates at 4,096 random sites and fell back, increasing reconstruction
from 12.049 to 19.697 ms. Its uncached 32,768-site comparison disabled the existing
fast path in both arms, so it is not a current-default comparison. Those negative
records remain in `local-dewall-reconstruction-v1`; they are not discarded in
favor of the successful variant. Early float64 and float32 candidate-only probes
also retain their certificate failures, duplicate cells and incomplete hulls.
The archived v4 kernel is float32; its inherited exploratory header still
describes the preceding double-arithmetic prototype.

## Complete VTM and verification

| Full torus solve | Current ms | Private ms | Reduction | Workspace bytes, before → after |
|---|---:|---:|---:|---|
| 1,000 sites | 640.754 | 626.271 | 2.26% | 273,863,500 → 282,658,496 |
| 4,096 sites | 1855.389 | 1840.290 | 0.81% | 1,743,416,484 → 1,806,215,872 |

These include fresh sampling, progressive growth, optimization, quality work and
extraction. The pairs use five and three warmed samples respectively. Points
match bitwise; roles, full/material tetrahedra, oriented boundaries, parameters,
counts and solver work match. Independent CGAL global empty-sphere and hull checks,
NumPy quality calculations, CGAL surface-distance queries and boundary vertex-link
checks pass for both candidate outputs. The
[validation record](../results/validation/vtm/local-dewall/local-dewall-vtm-v2-independent.json)
contains the full evidence. This small pilot gain, especially at 4k, is not a
general VTM performance guarantee or a CPU speedup.

Eight [focused tests](../results/validation/vtm/local-dewall/test_local_dewall_v2.py)
pass. A single captured producer/mesher/consumer graph changes positions, duplicate
owners and scale, injects invalid input and recovers while retaining output
addresses. Capture rejects host allocation/readback/synchronization calls. Other
tests force candidate overflow, an empty proposal, seed-filter failure, completion
failure, insufficient final capacity and extreme finite scales. Both padded
Compute Sanitizer runs report eight passes and zero errors: ordinary pooled
memory and pool-disabled use-after-free tracking, using a 128 KiB diagnostic
stack. Full allocation-order tracking and a complete native VTM trace have not
been run for this prototype.

Production adoption still requires broader feature/nonconvex/graded cases, cold
failure costs, memory reduction, native graph auditing and an evidence-based
dispatch policy. The [artifact manifest](../results/validation/vtm/local-dewall/manifest.json)
identifies the retained snapshots. On a fresh checkout, copy its Python scripts
and `references/` directory into `.work/vtm` and run `test_local_dewall_v2.py`
with pytest and `PYTHONPATH=.:.work/vtm`. The serial drivers create new result
directories, so do not prepopulate those with the archived results.
All reconstruction v2 and complete-VTM meshes are included. Large exploratory
v1 reconstruction meshes remain private; their hashes and comparisons are retained.
