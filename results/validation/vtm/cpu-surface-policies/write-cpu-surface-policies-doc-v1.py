"""Document native benchmark policies and all initial scanned-domain controls."""
import json
from pathlib import Path

root=Path.cwd();w=root/'.work/vtm'
assert json.loads((w/'cpu-surface-policies-production-v1/result.json').read_text())['complete']
review=json.loads((w/'cpu-scan-controls-v1-review.json').read_text());assert review['complete']
table='| CPU control, one thread | Sites | Material valid | Native time, s | Min / mean radius ratio | Boundary RMS / bbox diagonal | Volume error |\n|---|---:|---|---:|---:|---:|---:|\n'
for name,c in review['cases'].items():
    data=json.loads((w/'cpu-scan-controls-v1'/name/'result.json').read_text());r=data['runs'][0]
    q=r.get('independent_quality')
    valid=c['records'][0]['accepted'];assert not r['cpu_throttled']
    quality=f"{q['radius_ratio_percentiles']['0']:.5f} / {q['mean_radius_ratio']:.5f}" if valid else '—'
    fit=f"{q['two_sided_area_weighted_rms_percent_bbox_diagonal']:.4f}%" if valid else '—'
    volume=f"{100*q['relative_volume_error']:+.3f}%" if valid else '—'
    table+=f"| {name} | {r['vertices']:,} | {'yes' if valid else 'no'} | {r['complete_seconds']:.6f} | {quality} | {fit} | {volume} |\n"
text='''# Native surface policies and scanned-domain controls

The CPU benchmark now exposes feature-angle and surface-topology policies while
preserving its historical defaults. It saves native timing/quota records before
independent quality validation, retains failed repeats and refuses to overwrite
an existing output directory. An optional native AABB oracle makes offline
distance measurements practical for dense scans. These changes affect benchmark
tools only: the pure-Warp mesher, standalone power-diagram/Delaunay code, tests,
examples and main README are unchanged.

This study does **not** establish a comparable-quality CPU speedup. It identifies
valid and invalid CPU controls and the remaining quality tradeoffs on the exact
normalized float32 Stanford/libigl Armadillo input used by the GPU examples.

## Controls and compatibility

`--feature-angle` defaults to 45 degrees; `--topology` defaults to `non_manifold`,
the historical CGAL refinement behavior. The default command still works with
old binaries. New binaries accept optional `feature_angle [non_manifold|manifold]`
arguments and report the selected policy. The runner records whether the native
binary explicitly reports topology or uses the historical implied default.
`--distance-binary` selects the offline distance oracle; without it, the original
brute-force metric remains available. [Build and CLI instructions](../benchmarks/cpu/README.md)
describe all timing and validation fields.

CGAL's [manifold refinement option](https://doc.cgal.org/latest/Mesh_3/group__PkgMesh3Parameters.html)
controls the generated surface; native triangulation validity is a different
property from a valid material boundary. Actual output after ODT still needs
independent checking. Equal feature angles do not prove equivalent corner
constraints or boundary updates between implementations.

Runtime `de7b38a60c61` matches the executed private native source byte for byte
and the private runner by AST. Fresh sequential and TBB builds use CGAL 6.2.1,
GNU 11.4.0, `-O3 -DNDEBUG -std=c++17`; original binaries and measurements are
preserved. Seven native functional cases, three private runner cases and the
public CLI checks pass. Historical default output is byte-identical, explicit
feature/topology policies work, invalid criteria reject, native failures retain
records, and attempts to reuse an output directory leave its files unchanged.
The optional AABB metric agrees with the original metric on the functional case.

## Initial scan results

The input has 43,243 vertices and 86,482 triangles. No repair or simplification is
applied. CPU inputs match the GPU's float32 arrays exactly; native output remains
float64. The CPU is an Intel Xeon Platinum 8362 under a four-core quota. Seed 2026,
feature angle 120, facet angle 25 and cell radius-edge ratio 2 are fixed.
Perturbation and exudation are disabled. These are controls for Mesh_3 plus ODT,
not a claim to represent CGAL's best optimizer combination.

The initial `size=0.045`, `facet_distance=0.005`, up-to-32-ODT sequential control
produces 8,240 vertices and 33,269 tetrahedra, but fails the independent material
boundary test. All three meshes are identical. Edge `(748,3975)` has four incident
boundary faces. CGAL's triangulation check passes; this is not an acceptable
material mesh. The original runner lost its in-memory quota records when the
quality assertion failed, so those timings have no clean-quota claim.

With the new runner, one repeated default-policy sample and three manifold-policy
sequential samples still fail material validation. Of three four-thread manifold
samples, two pass material topology, vertex-link and exact self-intersection
checks but are quota-throttled; the third is unthrottled but fails material
validation. There are no accepted unthrottled samples in that batch. Its initially
valid meshes have 8,375 and 8,379 referenced vertices, mean radius ratios about
0.802, minimum ratios 0.00566/0.01243 and boundary RMS about 0.0765% of the input
bbox diagonal. They are retained, not selected as clean speedup evidence.

Seven further sequential controls use the manifold policy and one fresh native
process each. All seven native timing windows are unthrottled. Four outputs pass
the full independent material review, including positive cells, closed oriented
edge incidence, one component, Euler characteristic 2, single-cycle vertex links
and zero exact boundary self-intersections. Rejected outputs remain in the table;
their native time is an observation, not a successful meshing time.

'''+table+'''
`odt0`, `odt1` and `odt8` use size 0.045 and facet distance 0.005 with the indicated
ODT iteration caps. The valid zero-ODT control followed by an invalid one-iteration
control locates the observed failure after optimization for these settings.
`tight-fit` uses distance 0.002; `loose-fit` uses 0.01. Both use size 0.045 and
a 32-iteration cap. `loose-finer` changes size to 0.04. These use convergence 0.02
and freeze bound 0.01. `loose-strict-odt` uses size 0.045, distance 0.01, up to
100 iterations and convergence/freeze bound 0.001. Actual return codes and all
component times remain in the raw reports. One sample per setting is an initial
control, not a repeated performance estimate.

For context, the separately measured [uniform GPU 8K result](ordered-surface-queries.md)
uses 8,192 sites (8,188 material-referenced), takes 3.950 seconds of warmed full
replay, and has minimum/mean radius ratios 0.03102/0.87457, boundary RMS 0.1480%
of the input bbox diagonal and volume error −2.012%. The valid CPU loose-fit
control uses 6,006 vertices with lower minimum/mean quality and slightly better
sampled fit. These are different points on the size/quality/fit tradeoff, not a
matched-quality ratio. Native complete time includes domain construction,
meshing and ODT; GPU warmed replay excludes workspace setup. Native I/O,
triangulation checks, output serialization and independent validation are
reported separately. Boundary RMS is sampled three-point quadrature, not a
Hausdorff guarantee.

## Failure records and evidence

The runner's `complete` flag means it finished recording the study. Its overall
`valid` flag is false if any repeat fails. Accepted medians include only individual
successful repeats; a mixed batch can therefore have an accepted-sample median
and still fail overall. Native-only timing observations are separately named.
The first private review driver incorrectly expected a null median for every
failed group; it stopped on the mixed parallel batch. The unchanged raw records
were then reviewed by a separate script, which preserves both valid/throttled
outputs and the invalid/unthrottled output. The earlier scalar-versus-list Euler
assertion in a functional test is also retained with its corrected test.

The [artifact manifest](../results/validation/vtm/cpu-surface-policies/manifest.json)
pins source/build hashes, all successful and rejected controls, native outputs,
quota records, independent reviews and public CLI checks. Historical input,
reference-source and GPU proof artifacts are linked by hash. CGAL feature
protection may use weighted sites; this study does not assert an independently
certified unweighted full-hull Delaunay mesh for its material-only CPU exports.
The scan's original attribution and redistribution terms remain unchanged.

[#248](https://github.com/alecjacobson/warpower/issues/248) tracks the topology and
failure-record milestone. The broader comparable-quality study in
[#244](https://github.com/alecjacobson/warpower/issues/244) and the wide CPU speedup
goal in [#40](https://github.com/alecjacobson/warpower/issues/40) remain open.
'''
p=root/'docs/cpu-surface-policies.md';assert not p.exists();p.write_text(text)
p=root/'docs/cpu-mesher-baseline.md';s=p.read_text();marker='# Native CPU comparison: initial torus result\n';assert s.startswith(marker);p.write_text(s.replace(marker,marker+'\n[Native surface policies and scanned-domain controls](cpu-surface-policies.md)\nadd explicit feature/topology settings and retained rejection records. The\noriginal torus experiment below remains unchanged.\n',1))
print('Wrote native policy/control report.')
