# Reproducible full-mesher examples

The gallery runner supplies fixed inputs, runs the complete public device
pipeline, and exports an inspectable tetrahedral mesh. Surface generation or OFF
loading happens before meshing. Each captured replay includes domain and optional
sizing refits, fresh sampling, progressive fitting/refinement, quality improvement
and extraction. Host readback, independent checking and file export happen after
replay. The examples do not add a CPU geometry fallback to the mesher.

From a source checkout:

```bash
pip install -e '.[dev,bench]'
python -m examples.vtm_gallery --case torus --n 1000 --output results/local/torus
python -m examples.vtm_gallery --case graded-torus --n 1000 --output results/local/graded-torus
python -m examples.vtm_gallery --case thin-hand --n 2000 --output results/local/thin-hand
python -m examples.vtm_gallery --case cavity --n 1000 --output results/local/cavity
python -m examples.vtm_gallery --case disconnected --n 1000 --output results/local/disconnected
```

These commands are an initial validation matrix; measured results are pending.
The thin hand is a new extruded palm with five narrow fingers and sharp corners,
not the scanned hand from Alliez et al. The torus tests nonconvex topology;
the cavity and separated components exercise material orientation and extraction.
The graded torus uses a prescribed relative sizing field, ranging from 0.35 near
the leftmost side to at most 1 elsewhere. The field's min-cone extension is
computed on device. `--automatic-sizing` on an ungraded case instead selects
the mesher's pole-based local-feature-size path.

Import an already closed, oriented triangular surface without changing its
geometry:

```bash
python -m examples.vtm_gallery --case surface --input model.off --n 2000 --output results/local/model
python -m examples.vtm_gallery --case graded-surface --input model.off --n 2000 --output results/local/graded-model
```

Input positions are converted to float32, matching the mesher's API. The exported
`input.off` writes those float32 values with 17 significant decimal digits so
native CPU comparisons can receive the same coordinates. Imported assets retain
their own provenance and licensing; this runner bundles no third-party model.

Each successful run writes:

- `result.json`: settings, seed, input hash, hardware, setup/first-replay times,
  additional replay samples, CPU quota observations, topology and quality.
- `input.off` and `boundary.off`: original input and extracted surface.
- `mesh.npz`: points, material/full-hull tetrahedra, boundary faces, roles and
  radius ratios, together with the input surface.
- `mesh.vtu`: a portable VTK tetrahedral mesh with a `radius_ratio` cell field.
  Open it in a mesh viewer and use a clipping plane to inspect interior elements.

Use `--repeats 5` for five additional complete resident-device solves. Initial
setup and the first solve are reported separately. This is not a CPU speedup
measurement. `--independent-quality` adds NumPy tetrahedron solves, incidence
checks, dihedral angles and two-sided area-weighted boundary RMS against all
input triangles. That brute-force validation may be slow on large surfaces; it
is excluded from replay timing. Boundary RMS uses three quadrature samples per
triangle and is not a Hausdorff bound. Invalid solves save diagnostics and do not
export an empty mesh as a successful result.
