"""Shaded pictures of CAD output as PNG, with no OpenGL: ``python -m topocombo.cadview``.

A downstream consumer like :mod:`topocombo.report`: it reads what the pipeline
exported — the design domain's BREP and the optimized ``topology.stl`` — and
never the solver.  Shapes are tessellated into triangles and drawn with
matplotlib's Agg backend using a painter's algorithm, back-face culling and
Lambert shading, so it runs on a headless CI box where PyVista's off-screen
rendering would need OSMesa or EGL.

The view is an axonometric one from the front, right and above, with y up so
the beam reads like the report's side views; sharp edges (a crease of more
than 30 degrees, or an open boundary) are drawn as lines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

#: Camera azimuth about y (from +z towards +x) and elevation, in degrees.
VIEW_AZIMUTH = 30.0
VIEW_ELEVATION = 22.0
#: Crease angle above which an edge is drawn as a line.
FEATURE_ANGLE = 30.0

_FACE_RGB = np.array([0.36, 0.56, 0.78])  # a mid steel blue, legible on light and dark pages
_EDGE_RGBA = (0.10, 0.14, 0.20, 0.9)
_LIGHT = np.array([0.35, 0.8, 0.5])


def _camera(azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(screen right, screen up, towards the viewer) unit vectors, y up."""
    az, el = np.radians(azimuth), np.radians(elevation)
    toward = np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    right = np.cross([0.0, 1.0, 0.0], toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    return right, up, toward


def _weld(points: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge coincident points so neighbouring triangles share their edges."""
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    keys = np.round(points / scale, 9)
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    welded = np.zeros((unique.shape[0], 3))
    welded[inverse.ravel()] = points
    return welded, inverse.ravel()[triangles]


def feature_edges(
    points: np.ndarray, triangles: np.ndarray, angle: float = FEATURE_ANGLE
) -> tuple[np.ndarray, np.ndarray]:
    """(edges as point-index pairs, the triangles on each side, -1 for none).

    An edge is kept when it bounds only one triangle or when the normals of its
    two triangles differ by more than ``angle`` degrees.
    """
    normals = _normals(points, triangles)
    edges = np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    owner = np.tile(np.arange(triangles.shape[0]), 3)
    keys = np.sort(edges, axis=1)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, owner = keys[order], owner[order]
    starts = np.flatnonzero(np.r_[True, np.any(keys[1:] != keys[:-1], axis=1)])
    counts = np.diff(np.r_[starts, keys.shape[0]])

    first = owner[starts]
    second = np.where(counts == 2, owner[np.minimum(starts + 1, owner.size - 1)], -1)
    cos = np.einsum("ij,ij->i", normals[first], normals[np.maximum(second, 0)])
    sharp = (second < 0) | (cos < np.cos(np.radians(angle)))
    return keys[starts][sharp], np.column_stack([first, second])[sharp]


def _normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    tri = points[triangles]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)


def render_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    png_path: Path,
    *,
    title: str | None = None,
    two_sided: bool = False,
    width_px: int = 1400,
    azimuth: float = VIEW_AZIMUTH,
    elevation: float = VIEW_ELEVATION,
) -> Path:
    """Draw a triangle surface to a transparent PNG and return its path.

    ``two_sided`` draws every triangle whichever way it faces (an open surface,
    such as the 2D design domain); otherwise triangles facing away from the
    camera are culled, which assumes a closed, outward-oriented surface.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    points, triangles = _weld(np.asarray(points, float), np.asarray(triangles, int))
    right, up, toward = _camera(azimuth, elevation)
    normals = _normals(points, triangles)
    facing = normals @ toward
    if two_sided:
        normals = np.where(facing[:, None] < 0, -normals, normals)
        facing = np.abs(facing)
    visible = facing > 1e-9

    screen = np.column_stack([points @ right, points @ up])
    depth = points @ toward
    tri_depth = depth[triangles].mean(axis=1)

    light = _LIGHT / np.linalg.norm(_LIGHT)
    shade = 0.45 + 0.55 * np.clip(normals @ light, 0.0, 1.0)
    face_rgba = np.column_stack([np.clip(shade[:, None] * _FACE_RGB, 0, 1), np.ones(len(shade))])

    # triangles and edges share one draw list, sorted far to near; an edge is
    # drawn just after the nearest visible triangle it bounds
    edges, sides = feature_edges(points, triangles)
    side_vis = np.where(sides >= 0, visible[np.maximum(sides, 0)], False)
    side_depth = np.where(side_vis, tri_depth[np.maximum(sides, 0)], -np.inf)
    edge_keep = side_vis.any(axis=1)
    edges, edge_depth = edges[edge_keep], side_depth[edge_keep].max(axis=1)

    tri_idx = np.flatnonzero(visible)
    polys = [screen[triangles[i]][[0, 1, 2, 0]] for i in tri_idx]
    polys += [screen[e] for e in edges]
    keys = np.r_[tri_depth[tri_idx], edge_depth + 1e-9 * max(np.ptp(depth), 1.0)]
    facecolors = np.vstack([face_rgba[tri_idx], np.zeros((len(edges), 4))])
    edgecolors = np.vstack([face_rgba[tri_idx], np.tile(_EDGE_RGBA, (len(edges), 1))])
    linewidths = np.r_[np.full(tri_idx.size, 0.8), np.full(len(edges), 0.9)]
    order = np.argsort(keys, kind="stable")

    lo, hi = screen.min(axis=0), screen.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    margin = 0.08 * span.max()
    aspect = (span[1] + 2 * margin) / (span[0] + 2 * margin)
    dpi = 150
    fig_w = width_px / dpi
    fig_h = min(max(fig_w * aspect, 1.6), fig_w * 1.2)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(lo[0] - margin, hi[0] + margin)
    ax.set_ylim(lo[1] - margin, hi[1] + margin)
    ax.add_collection(
        PolyCollection(
            [polys[i] for i in order],
            closed=False,
            facecolors=facecolors[order],
            edgecolors=edgecolors[order],
            linewidths=linewidths[order],
            antialiased=True,
        )
    )
    _axis_triad(ax, right, up, lo, span, margin)
    if title:
        ax.text(lo[0] - margin * 0.6, hi[1] + margin * 0.4, title, fontsize=9,
                color="#6b7684", va="top")

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, transparent=True)
    plt.close(fig)
    return png_path


def _axis_triad(ax, right, up, lo, span, margin) -> None:
    """x / y / z arrows in the lower-left corner, so the orientation reads."""
    size = 0.06 * span.max()
    origin = np.array([lo[0] - margin, lo[1] - margin]) + 0.9 * margin
    for axis, name in enumerate("xyz"):
        vec = np.eye(3)[axis]
        d = np.array([vec @ right, vec @ up]) * size
        ax.annotate(
            "", xy=origin + d, xytext=origin,
            arrowprops={"arrowstyle": "-|>", "color": "#8a94a0", "lw": 0.9},
        )
        ax.text(*(origin + d * 1.25), name, color="#8a94a0", fontsize=7, ha="center", va="center")


# --------------------------------------------------------------------------
# sources: a CadQuery shape (BREP) and an STL
# --------------------------------------------------------------------------
def shape_triangles(shape, tolerance: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tessellate a CadQuery shape into (points, triangles)."""
    bb = shape.BoundingBox()
    tol = tolerance or max(bb.DiagonalLength, 1e-9) * 1e-3
    vertices, triangles = shape.tessellate(tol, 0.2)
    points = np.array([v.toTuple() for v in vertices], dtype=float)
    return points, np.asarray(triangles, dtype=int).reshape(-1, 3)


def render_brep(brep_path: Path, png_path: Path, title: str | None = None) -> Path:
    """Render an exported CadQuery BREP (a face or a solid) to PNG."""
    import cadquery as cq

    shape = cq.importers.importBrep(str(brep_path)).val()
    points, triangles = shape_triangles(shape)
    has_volume = shape.ShapeType() in ("Solid", "CompSolid", "Compound") and shape.Volume() > 0
    return render_triangles(points, triangles, png_path, title=title, two_sided=not has_volume)


def render_stl(stl_path: Path, png_path: Path, title: str | None = None) -> Path:
    """Render a closed STL surface (e.g. the optimized topology) to PNG."""
    import meshio

    mesh = meshio.read(str(stl_path))
    triangles = np.vstack([b.data for b in mesh.cells if b.type == "triangle"])
    return render_triangles(mesh.points, triangles, png_path, title=title)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m topocombo.cadview",
        description="Render a BREP or STL (from a topocombo run) to a shaded PNG.",
    )
    parser.add_argument("source", type=Path, help=".brep or .stl file")
    parser.add_argument("--out", type=Path, help="PNG to write (default: next to the source)")
    args = parser.parse_args(argv)

    out = args.out or args.source.with_suffix(".png")
    suffix = args.source.suffix.lower()
    if suffix == ".brep":
        render_brep(args.source, out)
    elif suffix == ".stl":
        render_stl(args.source, out)
    else:
        parser.error(f"cannot render {args.source.name}: expected a .brep or .stl file")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
