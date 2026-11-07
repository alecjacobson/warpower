"""Render an exported mesh as a quality-colored tetrahedral cutaway."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def render(
    mesh,
    output,
    axis=0,
    fraction=0.5,
    title="Variational tetrahedral mesh",
    up_axis="z",
    credit=None,
    elevation=27,
    azimuth=-55,
):
    data = np.load(mesh)
    p, t = data["points"].astype(np.float64), data["tetrahedra"]
    q = data["radius_ratio"]
    coordinate = p[t].mean(axis=1)[:, axis]
    cut = p[:, axis].min() + fraction * np.ptp(p[:, axis])
    selected = np.flatnonzero(coordinate <= cut)
    if not len(selected):
        raise ValueError("This cut removes every tetrahedron; increase --fraction")
    cells = t[selected]
    faces = np.concatenate(
        [cells[:, [1, 2, 3]], cells[:, [0, 3, 2]], cells[:, [0, 1, 3]], cells[:, [0, 2, 1]]]
    )
    owner = np.tile(selected, 4)
    _, first, count = np.unique(
        np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    exposed = first[count == 1]
    triangles = p[faces[exposed]]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    light = np.array([0.3, -0.5, 0.8])
    light /= np.linalg.norm(light)
    cmap, norm = plt.get_cmap("RdYlBu"), colors.Normalize(0, 1)
    rgba = cmap(norm(q[owner[exposed]]))
    rgba[:, :3] *= (0.7 + 0.3 * np.clip(normal @ light, 0, 1))[:, None]
    fig = plt.figure(figsize=(10, 8), facecolor="#f4f7fb")
    ax = fig.add_axes((0.0, 0.08, 0.88, 0.82), projection="3d", facecolor="#f4f7fb")
    ax.add_collection3d(
        Poly3DCollection(
            triangles, facecolors=rgba, edgecolors=(0.12, 0.2, 0.3, 0.35), linewidths=0.25
        )
    )
    visible = triangles.reshape(-1, 3)
    lo, hi = visible.min(axis=0), visible.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.1)
    ax.view_init(elev=elevation, azim=azimuth, vertical_axis=up_axis)
    ax.set_axis_off()
    fig.text(0.055, 0.93, title, color="#142c44", fontsize=22, weight="bold")
    fig.text(
        0.055,
        0.89,
        f"{len(p):,} sites  ·  {len(t):,} tetrahedra",
        color="#52677a",
        fontsize=11,
    )
    cax = fig.add_axes((0.88, 0.25, 0.022, 0.42))
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.set_label("Radius ratio  3r/R", color="#33495e", labelpad=12)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors="#52677a", length=0)
    fig.text(
        0.055,
        0.055,
        (
            "Cutaway removes whole tetrahedra by centroid; exposed faces show the interior mesh."
            if fraction < 1
            else "Boundary faces colored by the quality of their incident tetrahedron."
        ),
        color="#52677a",
        fontsize=9,
    )
    if credit:
        fig.text(0.055, 0.025, credit, color="#52677a", fontsize=8)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--axis", choices=["x", "y", "z"], default="x")
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--title", default="Variational tetrahedral mesh")
    p.add_argument("--up-axis", choices=["x", "y", "z"], default="z")
    p.add_argument("--credit", help="Optional dataset attribution in the figure")
    p.add_argument("--elevation", type=float, default=27)
    p.add_argument("--azimuth", type=float, default=-55)
    args = p.parse_args()
    if not 0 < args.fraction <= 1:
        p.error("fraction must lie in (0, 1]")
    render(
        args.mesh,
        args.output,
        "xyz".index(args.axis),
        args.fraction,
        args.title,
        args.up_axis,
        args.credit,
        args.elevation,
        args.azimuth,
    )


if __name__ == "__main__":
    main()
