# From points to power

A 24-second, 1920 × 1080, 24 fps showcase of a weighted 3D power diagram:
glowing sites become polished cells, separate into an exploded view, and assemble
again under a slowly orbiting camera. The stereo ambient score is synthesized
specifically for this film.

[Watch or download the MP4](https://github.com/alecjacobson/warpower/releases/download/showcase-2026-11-06/warpower-showcase.mp4)
· [Scene and provenance](https://github.com/alecjacobson/warpower/releases/tag/showcase-2026-11-06)

![Exploded power diagram](poster.jpg)

## Geometry and performance claims

`make_geometry.py` calls `warpower.power_diagram` on 416 deterministic sites:
176 interior sites and 240 outer sites that bound the displayed cells. The outer
cells are omitted from the visualization. SciPy reconstructs polygon meshes in
float64 using the neighbor lists returned by Warpower. Each reconstructed vertex
is checked against **all 416 sites' power halfspaces**, rather than just its
reported neighbors. All construction statuses are successful, and the final
adjacency has no unresolved asymmetry. The introductory connections are a subset
of that actual adjacency.

Small gaps, bevels, cell scaling, and radial displacement are presentation effects.
The diagram and its weights stay fixed during the film; this is an offline
Blender render, not a real-time construction or rendering benchmark. Warpower's
library output remains adjacency, offsets, statuses, and diagnostics; mesh
reconstruction is part of this one-off visualization.

The closing card quotes the existing **100,000-site cluster5 Voronoi** benchmark
on NVIDIA L40: 30.4 ms for Warpower versus 66.0 ms for Paragram, or 2.17×. Those
are mean construction times over ten calls on a separate benchmark input. They
do not describe the 176-cell visual scene. The [full report](../results/summary/report.md)
includes slower cases, memory measurements, and benchmark settings.

## Reproduce

Start with the repository's [installation instructions](../README.md), then
install SciPy, Pillow, and imageio-ffmpeg into the same Python environment. The
scene uses Blender 4.5 LTS with Cycles/OptiX on an NVIDIA GPU and the system's
DejaVu Sans fonts at `/usr/share/fonts/truetype/dejavu`.

Run from the repository root, with `blender` pointing to Blender 4.5:

```bash
python -m pip install scipy Pillow imageio-ffmpeg
python -m showcase.make_geometry
blender --background --threads 4 --python showcase/build_scene.py
blender --background results/local/showcase/warpower-showcase.blend \
  --threads 4 --python showcase/render.py
python showcase/finish_video.py --finish
```

Outputs are written to the ignored, persistent `results/local/showcase/`
directory. `build_scene.py` saves the editable `.blend` scene and a preview;
`render.py` explicitly selects the OptiX GPU in each Blender process.
`finish_video.py` adds typography, generates the original score, and exports an
H.264/AAC MP4 with a web-friendly metadata layout. Add `-- --preview` to the
Blender render command to render four representative stills instead of the film.
