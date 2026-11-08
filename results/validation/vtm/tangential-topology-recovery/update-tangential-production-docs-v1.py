from pathlib import Path
p=Path('docs/mesher.md');s=p.read_text();needle='| `sampling_resolution` | 32 |';assert needle in s
s=s.replace(needle,'| `tangential_recovery` | `None` | Inherit `exterior_recovery`; bounded surface-vertex quality recovery after rejected pinched-link transactions, with complete hull recertification |\n'+needle)
needle='Each optional quality stage starts with a valid-mesh checkpoint.';assert needle in s
s=s.replace(needle,'''If those policies still leave a pinched boundary, [tangential topology recovery](tangential-topology-recovery.md)
can move an eligible surface vertex within 10/20% of its shortest incident edge.
At most 32 candidates retain the existing quality-move fit/volume/minimum-quality
bounds, strict deficit improvement, input topology, references and feature roles.
Every candidate certifies the complete hull; exhaustion reconstructs the original
points before reporting failure. `tangential_recovery=False` omits its storage
and work. This recovery remains enabled when `quality_rounds=0`; that option only
omits the ordinary selective/jitter/interior schedule. A small fitting-energy
increase within the quality bound is allowed; the global-step rule is unchanged.

'''+needle)
p.write_text(s)
p=Path('docs/architecture.md');s=p.read_text();needle='The [example gallery](examples.md) includes uniform/graded torus and scanned';assert needle in s
s=s.replace(needle,'''After ordinary extraction recovery fails, the full mesher can use a
[bounded tangential quality transaction](tangential-topology-recovery.md) at a
surface vertex identified from a rejected pinched-link boundary. Each of at most
32 proposals certifies the current complete hull and retains quality, topology,
reference and feature-role guards. Exhaustion restores points and reconstructs
private certificates. This scoped policy resolves the measured uniform/graded
8K Armadillo failures; standalone diagram/Delaunay paths and allocations are
unchanged. The report includes its additional setup/storage cost and bounded
failure semantics.

'''+needle);p.write_text(s)
p=Path('docs/exterior-proof-hierarchy.md');s=p.read_text();old='''The tangential recovery used to expose the large uniform Armadillo workload
remains a separate private experiment under
[#228](https://github.com/alecjacobson/warpower/issues/228).''';assert old in s
s=s.replace(old,'''The tangential recovery used in this historical benchmark was a separate private
experiment under [#228](https://github.com/alecjacobson/warpower/issues/228).
Its later production integration and independent qualification are documented in
[bounded tangential topology recovery](tangential-topology-recovery.md); the timing
pairs below still describe their original pinned revisions and private helper.''');p.write_text(s)
p=Path('docs/vtm.md');s=p.read_text();needle='The wide comparable-quality CPU performance goal remains unmet.';assert needle in s
s=s.replace(needle,'''[Bounded tangential topology recovery](tangential-topology-recovery.md) now resolves
the recorded uniform/graded 8K scanned Armadillo extraction failures while keeping
all meshing decisions in the captured device graph.
'''+needle,1);p.write_text(s)
p=Path('docs/examples.md');s=p.read_text();s+='''
The [8K scanned Armadillo recovery study](tangential-topology-recovery.md) adds
validated uniform and graded examples with the existing gallery commands:

```bash
python -m examples.vtm_gallery --case surface --input path/to/armadillo.off \\
  --n 8192 --initial 256 --seed 2026 --subdivisions 1 --crease-subdivisions 1 \\
  --feature-angle 120 --corner-angle 45 --quality-rounds 4 --output results/local/armadillo8k
python -m examples.vtm_gallery --case graded-surface --input path/to/armadillo.off \\
  --n 8192 --initial 256 --seed 2026 --subdivisions 1 --crease-subdivisions 1 \\
  --feature-angle 120 --corner-angle 45 --quality-rounds 4 --output results/local/graded-armadillo8k
```

Use the [documented normalized input and asset provenance](scanned-armadillo.md).
These scans supplement the paper-inspired analytic and feature examples; they
are not presented as reproductions of the paper's meshes.
''';p.write_text(s)
