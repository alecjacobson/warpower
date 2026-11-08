"""Write the complete retained schedule/size study after native controls finish."""
import json
from pathlib import Path

root=Path.cwd()
w=root/'.work/vtm'
reports=[json.loads((w/f'fixed-site-current-{part}-v1-independent.json').read_text()) for part in ['pilot','larger']]
assert all(d['complete'] for d in reports)
rows={k:v for d in reports for k,v in d['cases'].items()}
text='''# Current schedule and mesh-size frontier

An existing fixed-site schedule improves complete-solve time and all nine checked
quality/fit metrics on the 1,000- and 8,192-site torus. It trades away worst-element
quality at 4,096 sites and becomes slower at 16,384 sites. **Defaults remain
unchanged, and the wide CPU speedup goal remains unmet.**

This study uses runtime `b44ab9e7332551a7a8a1c9bf9697f7bf8d7c109c` (numerical
source unchanged since `d8c0cab`), NVIDIA L40 and Warp 1.19.0. All GPU runs use the
same float32 48×24 torus surface, seed 2026 and four surface/crease subdivisions.
There are ten GPU configurations, with every explored outcome retained. This is
a single-input schedule study, not evidence for a universal schedule.

![Schedule time, percentile quality and fit](../results/validation/vtm/fixed-site-current/fixed-site-current-frontier-v1.png)

[PDF](../results/validation/vtm/fixed-site-current/fixed-site-current-frontier-v1.pdf)
and [SVG](../results/validation/vtm/fixed-site-current/fixed-site-current-frontier-v1.svg)
versions are available. Lines connect measured site budgets; they do not describe
continuous scaling or a fitted model.

## Existing public schedules

Progressive uses `(initial, growth steps, polish steps, selective steps, quality
rounds) = (256, 5, 20, 10, 4)`. Fixed20/fixed32 use all requested sites initially,
zero growth steps, 20/32 polish steps and the same selective/quality budgets.
Each timing is the median of three fresh captured solves after two initial
executions. Refit, fresh sampling, all fitting/reconstruction/recovery/quality
work and final extraction are included. Allocation, compilation, upload and
serialization are excluded; setup and first execution are recorded separately.
All initial and measured GPU quota windows are clean. Independent CPU geometry
checking runs after GPU timing, with one GPU process at a time.

Radius ratio is `3r/R` (larger is better). RMS is the two-sided area-weighted
sampled boundary distance as a percentage of the input bounding-box diagonal.
Signed volume error is relative to the input surface. Counts below are requested
sites; the raw records also give material-referenced vertices and tetrahedra.

| Sites / schedule | Complete s | Minimum ratio | 1st percentile | Median ratio | RMS % | Volume error % |
|---|---:|---:|---:|---:|---:|---:|
'''
for name,r in rows.items():
    q=r['independent']['quality']
    p=q['radius_ratio_percentiles']
    text+=f"| {name.removeprefix('torus')} | {r['median_ms']/1000:.5f} | {p['0']:.5f} | {p['1']:.5f} | {p['50']:.5f} | {q['two_sided_area_weighted_rms_percent_bbox_diagonal']:.5f} | {100*q['relative_volume_error']:.4f} |\n"
text+='''
Fixed32 improves the minimum, 1st/5th/50th/95th-percentile ratios, both extreme
dihedral angles, RMS and absolute volume error at 1,000 and 8,192 sites. At 4,096
and 16,384 sites it improves the percentile ratios and fit but worsens the
minimum and extreme dihedrals. Fixed20 at 4,096 sites also lowers the 1st/5th
percentiles and median quality. Full distributions, roles, memory, work and
individual timing samples remain in the records; no quality tradeoff is hidden
by an aggregate score.

At 16,384 sites fixed32 makes 227 full-capacity reconstructions, compared with
161 for progressive plus 41 smaller-prefix reconstructions. Fewer global fitting
steps therefore do not imply less total work. Production at this revision also
uses fragment reuse only through 4,096 reserved sites; larger workspaces use
ordinary full reconstruction. [Issue #226](https://github.com/alecjacobson/warpower/issues/226)
tracks an isolated extension experiment. Its private results are not production
performance in this report.

## Native CPU size controls

The native source and binary hashes match the existing
[CGAL Mesh_3/ODT build](cpu-mesher-baseline.md): CGAL 6.2.1, GNU C++ 11.4.0,
`-O3 -DNDEBUG`, Xeon Platinum 8362 at 2.80 GHz under a four-core quota, sequential
or TBB limited to four threads. The exact float32 input surface matches the GPU.
Native time includes domain construction, feature protection, meshing and ODT;
file I/O, native validity checking, serialization and independent quality checks
are separate. GPU resident and native complete timings have these different
setup scopes and are not transfer-inclusive cold-start comparisons.

All settings use facet distance 0.025, facet angle 25 degrees, feature angle
45 degrees, cell radius-edge ratio 2, ODT iteration cap 50, convergence 0.02 and
freeze bound 0.01. Perturbation/exudation are disabled. Size 0.128 targets the
8k range; the first 0.102 size pilot undershoots 16k, so 0.099 is also measured.
Every three-sample batch and every mesh is retained. Quota-throttled samples
remain in the primary median; clean counts and subset medians are disclosed in
the raw records. These are size controls with quality/stopping-rule differences,
not quality-matched speedups.

| Size / CPU mode | Referenced vertices | Median s, all samples | Clean samples | Minimum ratio | 1st percentile | Median ratio | RMS % | Volume error % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''
def span(values,digits):
    lo,hi=min(values),max(values)
    return f'{lo:.{digits}f}' if lo==hi else f'{lo:.{digits}f}–{hi:.{digits}f}'
codes=set()
for part in ['cpu','cpu-refined']:
    directory=w/f'fixed-site-current-{part}-v1'
    for row in json.loads((directory/'commands.json').read_text()):
        d=json.loads((directory/row['name']/'result.json').read_text())
        assert row['exit_code']==0 and len(d['runs'])==3
        q=[r['independent_quality'] for r in d['runs']]
        codes.update(r['odt_return_code'] for r in d['runs'])
        params=d['parameters']
        text+=f"| {params['size']} / {params['threads']} thread(s) | {span([r['referenced_vertices'] for r in q],0)} | {d['median_complete_seconds']:.5f} | {d['unthrottled_samples']}/3 | {span([r['radius_ratio_percentiles']['0'] for r in q],5)} | {span([r['radius_ratio_percentiles']['1'] for r in q],5)} | {span([r['radius_ratio_percentiles']['50'] for r in q],5)} | {span([r['two_sided_area_weighted_rms_percent_bbox_diagonal'] for r in q],5)} | {span([100*r['relative_volume_error'] for r in q],4)} |\n"
text+=f'\nObserved native ODT return codes: {sorted(codes)}. '
text+='''The source records preserve the
termination result of every sample. No equal-iteration or exact-count equivalence
is assumed, and this study does not establish a native CPU speedup.

## Validation and reproduction

All ten GPU outputs pass independent CGAL original-coordinate full-hull positive
orientation, global empty-sphere, face pairing and hull-support checks. Independent
material checks measure quality, positive volumes, incidence, topology and
surface distances. Every output boundary has single-cycle vertex links and zero
exact self-intersections. The native outputs receive positive-volume, incidence,
topology, quality and AABB surface-distance measurements for every distinct mesh;
this report does not add exhaustive native empty-sphere or self-intersection
audits. All material outputs have one component and torus boundary Euler
characteristic zero. RMS uses three area-weighted samples per triangle in each
direction and is not a Hausdorff certificate.

For the 4,096-site fixed32 schedule, from an installed checkout:

```bash
python -m benchmarks.mesher --case torus --n 4096 --initial 4096 \\
  --growth-iterations 0 --polish-iterations 32 --selective-iterations 10 \\
  --quality-rounds 4 --seed 2026 --subdivisions 4 --repeats 3 \\
  --save-mesh results/local/fixed4096.npz --output results/local/fixed4096.json
```

Use `--initial 256 --growth-iterations 5 --polish-iterations 20` for the
progressive control. Build native references with
[the CPU instructions](../benchmarks/cpu/README.md). All exact commands, raw
samples, mesh snapshots, offline checks, plots, helper references and source/
binary hashes are in the
[manifest](../results/validation/vtm/fixed-site-current/manifest.json).
Native text meshes are losslessly gzip-compressed with both compressed and raw
hashes. Workspace-specific paths in the archived drivers identify provenance;
restore them under a separate checkout or adapt roots before reproduction.

This completes the bounded study in
[#225](https://github.com/alecjacobson/warpower/issues/225).
The implementation and wide comparable-quality CPU goal remain open in
[#29](https://github.com/alecjacobson/warpower/issues/29) and
[#40](https://github.com/alecjacobson/warpower/issues/40). Runtime, independently
callable diagram/Delaunay behavior, defaults and the README flagship are unchanged.
'''
p=root/'docs/fixed-site-current.md'
assert not p.exists()
p.write_text(text)
