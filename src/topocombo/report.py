"""Render a run directory into a static HTML page (for GitHub Pages).

This is a downstream consumer of the artifacts written by
:mod:`topocombo.pipeline` — it reads ``run.json`` and ``mesh.npz`` from disk and
never imports the solver or the mesher.  The mesh figure is emitted as inline
SVG so the published page needs no JavaScript, no CDN and no plotting library.
The CAD pictures — the exported design domain and the optimized topology — are
PNGs rendered next to the page by :mod:`topocombo.cadview`.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .elements import ELEMENTS, HEX8, TET4, TET10

_CSS = """
:root {
  color-scheme: light dark;
  --bg: #ffffff; --panel: #f6f7f9; --border: #d9dee5; --text: #1b1f24;
  --muted: #5b6672; --accent: #2f6f4f; --accent-soft: #e5f0ea;
  --fail: #a32020; --code-bg: #f0f2f5;
  /* sequential ramp for magnitude fields, one hue light -> dark */
  --seq-1: #cde2fb; --seq-2: #9ec5f4; --seq-3: #6da7ec; --seq-4: #3987e5;
  --seq-5: #256abf; --seq-6: #184f95; --seq-7: #0d366b;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151a; --panel: #1a1f26; --border: #2c333d; --text: #e6eaef;
    --muted: #9aa5b1; --accent: #6fbf93; --accent-soft: #1d2a24;
    --fail: #e07a7a; --code-bg: #0e1116;
    /* on a dark surface the ramp runs the other way: low recedes, high stands out */
    --seq-1: #104281; --seq-2: #184f95; --seq-3: #256abf; --seq-4: #3987e5;
    --seq-5: #5598e7; --seq-6: #86b6ef; --seq-7: #cde2fb;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
}
.wrap { max-width: 980px; margin: 0 auto; padding: 48px 16px 96px; }
header.top { border-bottom: 1px solid var(--border); padding-bottom: 24px; margin-bottom: 32px; }
h1 { font-size: 1.9rem; margin: 0 0 8px; letter-spacing: -0.02em; }
h2 { font-size: 1.25rem; margin: 40px 0 12px; letter-spacing: -0.01em; }
h3 { font-size: 1rem; margin: 0; }
p.lede { color: var(--muted); margin: 0 0 16px; max-width: 70ch; }
.meta { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.chip {
  font-family: var(--mono); font-size: 0.78rem; padding: 3px 9px;
  border: 1px solid var(--border); border-radius: 999px; color: var(--muted);
  background: var(--panel);
}
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; font-size: 0.8rem; text-transform: uppercase;
     letter-spacing: 0.04em; }
td.num, th.num { text-align: right; font-family: var(--mono); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.card .k { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.card .v { font-family: var(--mono); font-size: 1.35rem; margin-top: 4px; }
.card .s { color: var(--muted); font-size: 0.8rem; }
figure { margin: 0; background: var(--panel); border: 1px solid var(--border);
         border-radius: 10px; padding: 16px; }
figure svg, figure img { width: 100%; height: auto; display: block; }
figure + figure { margin-top: 16px; }
.cad { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr); gap: 16px;
       align-items: start; margin-bottom: 16px; }
.cad table { margin: 0; }
.cad td { overflow-wrap: anywhere; }
pre code { padding: 0; background: none; font-size: inherit; }
.script { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; margin-bottom: 16px; }
.script > .head { display: flex; align-items: center; gap: 10px; padding: 10px 16px;
                  background: var(--panel); border-bottom: 1px solid var(--border);
                  font-size: 0.85rem; color: var(--muted); }
.script > .head a { margin-left: auto; }
@media (max-width: 760px) { .cad { grid-template-columns: minmax(0, 1fr); } }
figcaption { color: var(--muted); font-size: 0.85rem; margin-top: 10px; }
.inputs { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px;
          margin-bottom: 16px; }
.inputs h3 { font-size: 0.9rem; margin: 0 0 4px; }
.inputs td { overflow-wrap: anywhere; }
.step { border: 1px solid var(--border); border-radius: 10px; margin-bottom: 14px; overflow: hidden; }
.step > .head { display: flex; align-items: center; gap: 10px; padding: 12px 16px;
                background: var(--panel); border-bottom: 1px solid var(--border); }
.step > .head .dur { margin-left: auto; font-family: var(--mono); font-size: 0.8rem; color: var(--muted); }
.badge { font-size: 0.72rem; font-family: var(--mono); padding: 2px 8px; border-radius: 999px;
         background: var(--accent-soft); color: var(--accent); border: 1px solid var(--accent); }
.badge.fail { color: var(--fail); border-color: var(--fail); background: transparent; }
pre { margin: 0; padding: 14px 16px; background: var(--code-bg); overflow-x: auto;
      font-family: var(--mono); font-size: 0.82rem; line-height: 1.5; }
details > summary { cursor: pointer; padding: 10px 16px; color: var(--muted); font-size: 0.85rem;
                    border-top: 1px solid var(--border); }
code { font-family: var(--mono); font-size: 0.86em; background: var(--code-bg);
       padding: 1px 5px; border-radius: 4px; }
a { color: var(--accent); }
.pass { color: var(--accent); font-family: var(--mono); }
.legend { display: flex; align-items: center; gap: 10px; margin-top: 12px; flex-wrap: wrap; }
.legend .swatches { display: flex; }
.legend .swatches span { width: 26px; height: 12px; display: block; }
.legend .lbl { color: var(--muted); font-family: var(--mono); font-size: 0.75rem; }
.fail { color: var(--fail); font-family: var(--mono); }
footer { margin-top: 56px; padding-top: 20px; border-top: 1px solid var(--border);
         color: var(--muted); font-size: 0.85rem; }
@media (max-width: 640px) { .wrap { padding: 28px 16px 64px; } h1 { font-size: 1.5rem; } }
"""


def _e(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(v) for v in value)
    return str(value)


#: The six faces of a hexahedron, as corner indices (Gmsh / VTK order).
_HEX_FACES = np.array(HEX8.facets)


def _side_view(
    nodes: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x-y node coordinates, quads, element indices) to draw: the mesh itself in 2D.

    A hex mesh is drawn as its front (z = 0) layer seen from +z: each of those
    hexahedra by its face at the lowest z, corners counter-clockwise in x-y.
    Gmsh's corner numbering says nothing about which face that is, so it is
    found from the coordinates.  With one element through the width the front
    layer is the whole mesh; the returned indices pick each drawn element's
    values out of per-element fields.
    """
    if nodes.shape[1] == 2:
        return nodes, cells, np.arange(cells.shape[0])
    if cells.shape[1] in (4, 10):
        return _tet_front_view(nodes, cells)
    faces = cells[:, _HEX_FACES]  # (m, 6, 4)
    z = nodes[faces, 2]
    # the face normal to z has no z extent; among the two, take the lower one
    score = np.round(np.ptp(z, axis=2), 9) * 1e6 + z.mean(axis=2)
    quads = faces[np.arange(cells.shape[0]), np.argmin(score, axis=1)]  # (m, 4)
    front = np.flatnonzero(np.isclose(nodes[quads[:, 0], 2], nodes[:, 2].min()))
    quads = quads[front]
    xy = nodes[quads, :2]
    angle = np.arctan2(*(xy - xy.mean(axis=1, keepdims=True)).transpose(2, 0, 1)[::-1])
    quads = np.take_along_axis(quads, np.argsort(angle, axis=1), axis=1)
    return nodes[:, :2], quads, front


def _tet_front_view(
    nodes: np.ndarray, cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x-y node coordinates, triangles, owning element of each) of a tet mesh
    seen from the front: the boundary facets facing -z, by their corners,
    ordered back to front so later ones paint over earlier ones.

    A tetrahedral mesh has no layers or grid columns to draw; its front
    surface is what a viewer at z = -inf sees of any part, so every per-element
    field is shown on it, each facet in its element's value.
    """
    el = TET10 if cells.shape[1] == 10 else TET4
    local = np.array([f[:3] for f in el.facets])  # corners, outward order
    facets = cells[:, local].reshape(-1, 3)
    owner = np.repeat(np.arange(cells.shape[0]), local.shape[0])
    _, inverse, counts = np.unique(
        np.sort(facets, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    outer = counts[inverse.ravel()] == 1
    facets, owner = facets[outer], owner[outer]
    p = nodes[facets]
    normal_z = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
    front = normal_z < -1e-9 * max(float(area.max()), 1e-30)
    facets, owner = facets[front], owner[front]
    order = np.argsort(-nodes[facets, 2].mean(axis=1), kind="stable")  # far to near
    tris = facets[order]
    # counter-clockwise in x-y, as the quad views are
    xy = nodes[tris][:, :, :2]
    a, b = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
    signed = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    tris[signed < 0] = tris[signed < 0][:, [0, 2, 1]]
    return nodes[:, :2], tris, owner[order]


#: Projections of a 3D field: view name -> (axis averaged over, horizontal axis,
#: vertical axis, caption).
PROJECTIONS = {
    "side": (2, 0, 1, "side view (x-y), mean through the width"),
    "top": (1, 0, 2, "top view (x-z), mean through the height"),
    "end": (0, 2, 1, "end view (z-y), mean along the length"),
}


def project_field(
    nodes: np.ndarray, cells: np.ndarray, field: np.ndarray, axis: int, u: int, v: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a per-element field along ``axis`` of a structured hex grid.

    Elements sharing the same extent in the (u, v) plane form one column; the
    column's mean is drawn as one rectangle.  Returns (corner points in (u, v),
    rectangles as corner indices, column means).
    """
    xyz = nodes[cells]  # (m, 8, 3)
    lo, hi = xyz.min(axis=1), xyz.max(axis=1)
    keys = np.round(np.column_stack([lo[:, u], lo[:, v], hi[:, u], hi[:, v]]), 9)
    columns, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    means = np.bincount(inverse, weights=field) / np.bincount(inverse)
    u0, v0, u1, v1 = columns.T
    points = np.stack(
        [np.column_stack(c) for c in ((u0, v0), (u1, v0), (u1, v1), (u0, v1))], axis=1
    ).reshape(-1, 2)
    rects = np.arange(points.shape[0]).reshape(-1, 4)
    return points, rects, means


# --------------------------------------------------------------------------
# mesh figure
# --------------------------------------------------------------------------
def mesh_svg(
    npz_path: Path,
    width: int = 900,
    pad: int = 46,
    holes: list[list[float]] | tuple = (),
) -> str:
    """Inline SVG of the quad mesh with the supports and the tip load marked.

    Elements held void (``passive`` in the npz) are drawn empty, with the CAD
    cutouts in ``holes`` — (x, y, diameter) — outlined over them.
    """
    data = np.load(npz_path)
    nodes, quads, drawn = _side_view(data["nodes"], data["cells"])
    n_cells = int(data["cells"].shape[0])
    passive = data["passive"][drawn] if "passive" in data else np.zeros(len(quads), dtype=bool)
    if "fixed_nodes" in data:
        fixed = data["fixed_nodes"]
    else:  # runs from before regions
        fixed = data["set_fixed"] if "set_fixed" in data else np.empty(0, dtype=int)
    load_node = int(data["load_node"][0]) if "load_node" in data else None
    direction = _arrow_direction(data)

    xmin, ymin = nodes.min(axis=0)
    xmax, ymax = nodes.max(axis=0)
    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    scale = (width - 2 * pad) / span_x
    height = int(span_y * scale + 2 * pad)

    def px(p: np.ndarray) -> tuple[float, float]:
        # SVG y grows downwards; flip so the beam reads like the CAD sketch.
        return (pad + (p[0] - xmin) * scale, height - pad - (p[1] - ymin) * scale)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Mesh of the cantilever beam design domain">',
        '<style>'
        '.el{fill:var(--accent-soft);stroke:var(--accent);stroke-width:0.6;stroke-opacity:0.55}'
        '.pv{fill:none;stroke:var(--muted);stroke-width:0.6;stroke-opacity:0.35}'
        '.hole{fill:none;stroke:var(--text);stroke-width:1.4;stroke-dasharray:5 3}'
        '.bd{fill:none;stroke:var(--text);stroke-width:1.6}'
        '.sup{stroke:var(--text);stroke-width:1.6}'
        '.roll{fill:none;stroke:var(--text);stroke-width:1.3}'
        '.ld{stroke:var(--fail);stroke-width:2.4;fill:var(--fail)}'
        '.lbl{fill:var(--muted);font:12px ui-monospace,monospace}'
        '</style>',
    ]

    for quad, void in zip(quads, passive):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (px(nodes[i]) for i in quad))
        cls = "pv" if void else "el"
        parts.append(f'<polygon class="{cls}" points="{pts}"/>')
    for hx, hy, hd in holes:
        cx, cy = px(np.array([hx, hy]))
        parts.append(
            f'<circle class="hole" cx="{cx:.2f}" cy="{cy:.2f}" r="{hd / 2 * scale:.2f}"/>'
        )
        parts.append(
            f'<text class="lbl" x="{cx:.2f}" y="{cy + hd / 2 * scale + 16:.2f}" '
            f'text-anchor="middle">\u2300{hd:g} cutout</text>'
        )

    # outline: the bounding box, only when the mesh fills it (a body-fitted
    # mesh of a notched part shows its own boundary through the elements)
    corners = [
        px(np.array([xmin, ymin])),
        px(np.array([xmax, ymin])),
        px(np.array([xmax, ymax])),
        px(np.array([xmin, ymax])),
    ]
    q = nodes[quads]
    area = 0.5 * np.abs(
        np.sum(q[:, :, 0] * np.roll(q[:, :, 1], -1, axis=1)
               - np.roll(q[:, :, 0], -1, axis=1) * q[:, :, 1], axis=1)
    ).sum()
    if area >= 0.999 * span_x * span_y:
        parts.append(
            '<polygon class="bd" points="'
            + " ".join(f"{x:.2f},{y:.2f}" for x, y in corners)
            + '"/>'
        )

    # supports: a clamped edge is hatched; a node holding only some components
    # gets a roller triangle on the side of each held in-plane component
    clamped = data["clamped_nodes"] if "clamped_nodes" in data else fixed
    if "held" in data and fixed.size:
        centre = (nodes[:, :2].min(axis=0) + nodes[:, :2].max(axis=0)) / 2
        partial = ~np.isin(fixed, clamped)
        for node, held in zip(fixed[partial], data["held"][partial]):
            x, y = px(nodes[node])
            for axis in np.flatnonzero(held[:2]):
                # the triangle sits outside the part, its tip on the node
                side = -1.0 if nodes[node][axis] <= centre[axis] else 1.0
                if axis == 0:
                    bx = x + side * 9
                    tri = f"{x:.2f},{y:.2f} {bx:.2f},{y - 5:.2f} {bx:.2f},{y + 5:.2f}"
                else:
                    by = y - side * 9  # screen y points down
                    tri = f"{x:.2f},{y:.2f} {x - 5:.2f},{by:.2f} {x + 5:.2f},{by:.2f}"
                parts.append(f'<polygon class="roll" points="{tri}"/>')
        held_counts = data["held"][partial][:, :2].sum(axis=0)
        if partial.any():
            words = [f"{int(n)} u{a}" for a, n in zip("xy", held_counts) if n]
            parts.append(
                f'<text class="lbl" x="{pad:.2f}" y="{height - 12:.2f}" text-anchor="start">'
                f'rollers / symmetry: {", ".join(words)}</text>'
            )
    fixed = clamped
    if fixed.size:
        fx, fy0 = px(nodes[fixed][np.argmin(nodes[fixed][:, 1])])
        _, fy1 = px(nodes[fixed][np.argmax(nodes[fixed][:, 1])])
        top, bot = min(fy0, fy1), max(fy0, fy1)
        parts.append(f'<line class="sup" x1="{fx:.2f}" y1="{top:.2f}" x2="{fx:.2f}" y2="{bot:.2f}"/>')
        n_hatch = 14
        for i in range(n_hatch + 1):
            y = top + (bot - top) * i / n_hatch
            parts.append(
                f'<line class="sup" x1="{fx:.2f}" y1="{y:.2f}" '
                f'x2="{fx - 11:.2f}" y2="{y + 11:.2f}"/>'
            )
        parts.append(
            f'<text class="lbl" x="{fx - 12:.2f}" y="{top - 12:.2f}" text-anchor="start">'
            f'clamped ({fixed.size} nodes)</text>'
        )

    # the load: an arrow along the force's in-plane direction, ending at the load node
    if load_node is not None:
        lx, ly = px(nodes[load_node])
        parts.append(_arrow(lx, ly, direction))
        # keep the label inside the viewBox when the load sits on the right edge
        near_right = lx > width / 2
        anchor = "end" if near_right else "start"
        tx = lx - 8 if near_right else lx + 8
        parts.append(
            f'<text class="lbl" x="{tx:.2f}" y="{max(ly - 40 * direction[1] - 6, 14):.2f}" '
            f'text-anchor="{anchor}">F (node {load_node})</text>'
        )

    # dimension labels
    parts.append(
        f'<text class="lbl" x="{width / 2:.2f}" y="{height - 12:.2f}" text-anchor="middle">'
        f'L = {span_x:g} mm, {n_cells} elements'
        + (f' ({int(quads.shape[0])} front facets drawn)' if quads.shape[0] != n_cells else '')
        + '</text>'
    )
    parts.append(
        f'<text class="lbl" x="{pad - 14:.2f}" y="{height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 {pad - 14:.2f} {height / 2:.2f})">H = {span_y:g} mm</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def solution_svg(
    mesh_npz: Path,
    solution_npz: Path,
    width: int = 900,
    pad: int = 46,
    n_steps: int = 7,
) -> tuple[str, list[float]]:
    """Deformed mesh shaded by element compliance; returns (svg, log-scale bin edges).

    The field spans several orders of magnitude (the clamped corners carry almost
    all of the strain energy), so the shading bins are even in log10 — a linear
    ramp would paint everything but the support the lightest step.
    """
    mesh_data = np.load(mesh_npz)
    sol = np.load(solution_npz)
    nodes, quads, drawn = _side_view(mesh_data["nodes"], mesh_data["cells"])
    disp = sol["displacements"][:, :2]
    field = np.asarray(sol["element_compliance"], dtype=float)[drawn]

    xmin, ymin = nodes.min(axis=0)
    xmax, ymax = nodes.max(axis=0)
    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    scale = (width - 2 * pad) / span_x
    height = int(span_y * scale + 2 * pad)

    # exaggerate the deformation to ~12% of the beam height so it is legible
    max_disp = float(np.abs(disp).max())
    exaggeration = (0.12 * span_y / max_disp) if max_disp > 0 else 1.0
    deformed = nodes + exaggeration * disp

    def px(p: np.ndarray) -> tuple[float, float]:
        return (pad + (p[0] - xmin) * scale, height - pad - (p[1] - ymin) * scale)

    positive = field[field > 0]
    lo = float(positive.min()) if positive.size else 1e-12
    hi = float(field.max()) if field.max() > lo else lo * 10.0
    edges = np.linspace(np.log10(lo), np.log10(hi), n_steps + 1)
    bins = np.clip(
        np.digitize(np.log10(np.maximum(field, lo)), edges[1:-1]), 0, n_steps - 1
    )

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Deformed cantilever beam shaded by element compliance">',
        "<style>"
        + "".join(
            f".s{i + 1}{{fill:var(--seq-{i + 1});stroke:var(--seq-{i + 1});stroke-width:0.4}}"
            for i in range(n_steps)
        )
        + ".undef{fill:none;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 4}"
        ".lbl{fill:var(--muted);font:12px ui-monospace,monospace}"
        ".ld{stroke:var(--fail);stroke-width:2.4;fill:var(--fail)}"
        "</style>",
    ]

    undeformed_outline = [
        px(np.array([xmin, ymin])),
        px(np.array([xmax, ymin])),
        px(np.array([xmax, ymax])),
        px(np.array([xmin, ymax])),
    ]
    parts.append(
        '<polygon class="undef" points="'
        + " ".join(f"{x:.2f},{y:.2f}" for x, y in undeformed_outline)
        + '"/>'
    )

    for quad, b in zip(quads, bins):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (px(deformed[i]) for i in quad))
        parts.append(f'<polygon class="s{int(b) + 1}" points="{pts}"/>')

    if "load_node" in mesh_data:
        ln = int(mesh_data["load_node"][0])
        lx, ly = px(deformed[ln])
        parts.append(_arrow(lx, ly, _arrow_direction(mesh_data)))

    parts.append(
        f'<text class="lbl" x="{width / 2:.2f}" y="{height - 12:.2f}" text-anchor="middle">'
        f'displacements exaggerated {exaggeration:.0f}x (dashed: undeformed domain)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts), [float(10.0**e) for e in edges]


def _arrow_direction(data: Any) -> tuple[float, float]:
    """Unit screen direction (SVG y down) of the load's in-plane part; straight
    down when the run predates the recorded force or the force is out of plane."""
    if "load_vector" in data:
        v = np.asarray(data["load_vector"], dtype=float)[:2]
        n = float(np.linalg.norm(v))
        if n > 0:
            return (v[0] / n, -v[1] / n)
    return (0.0, 1.0)


def _arrow(x: float, y: float, d: tuple[float, float], length: float = 34.0) -> str:
    """A load arrow whose head touches (x, y), pointing along screen direction d."""
    dx, dy = d
    x0, y0 = x - dx * length, y - dy * length
    bx, by = x - dx * 11, y - dy * 11  # base of the head
    nx, ny = -dy * 5, dx * 5
    return (
        f'<line class="ld" x1="{x0:.2f}" y1="{y0:.2f}" x2="{x:.2f}" y2="{y:.2f}"/>'
        f'<polygon class="ld" points="{x:.2f},{y:.2f} {bx + nx:.2f},{by + ny:.2f} '
        f'{bx - nx:.2f},{by - ny:.2f}"/>'
    )


def _ramp_legend(edges: list[float], label: str, n_steps: int = 7) -> str:
    swatches = "".join(
        f"<span style='background:var(--seq-{i + 1})'></span>" for i in range(n_steps)
    )
    return (
        "<div class='legend'>"
        f"<span class='lbl'>{_e(f'{edges[0]:.2g}')}</span>"
        f"<span class='swatches'>{swatches}</span>"
        f"<span class='lbl'>{_e(f'{edges[-1]:.2g}')}</span>"
        f"<span class='lbl'>{_e(label)}</span>"
        "</div>"
    )


def _quad_field_svg(
    nodes: np.ndarray,
    quads: np.ndarray,
    bins: np.ndarray,
    width: int,
    pad: int,
    n_steps: int,
    aria: str,
    footer: str,
) -> str:
    """Shared renderer: one polygon per element, shaded by a pre-binned field."""
    xmin, ymin = nodes.min(axis=0)
    xmax, ymax = nodes.max(axis=0)
    span_x = max(xmax - xmin, 1e-12)
    span_y = max(ymax - ymin, 1e-12)
    scale = (width - 2 * pad) / span_x
    height = int(span_y * scale + 2 * pad)

    def px(p: np.ndarray) -> tuple[float, float]:
        return (pad + (p[0] - xmin) * scale, height - pad - (p[1] - ymin) * scale)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        # never wider than drawn, so narrow views (the end view) keep their scale
        f'style="max-width:{width}px;margin:0 auto" '
        f'role="img" aria-label="{_e(aria)}">',
        "<style>"
        + "".join(
            f".s{i + 1}{{fill:var(--seq-{i + 1});stroke:var(--seq-{i + 1});stroke-width:0.4}}"
            for i in range(n_steps)
        )
        + ".lbl{fill:var(--muted);font:12px ui-monospace,monospace}"
        ".ld{stroke:var(--fail);stroke-width:2.4;fill:var(--fail)}"
        ".undef{fill:none;stroke:var(--muted);stroke-width:1.2;stroke-dasharray:4 4}"
        "</style>",
    ]
    for quad, b in zip(quads, bins):
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (px(nodes[i]) for i in quad))
        parts.append(f'<polygon class="s{int(b) + 1}" points="{pts}"/>')

    parts.append(
        f'<text class="lbl" x="{width / 2:.2f}" y="{height - 12:.2f}" text-anchor="middle">'
        f"{_e(footer)}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def density_svg(
    mesh_npz: Path,
    density_npz: Path,
    width: int = 900,
    pad: int = 46,
    n_steps: int = 7,
    max_height: int = 420,
    structured: bool = True,
) -> str:
    """The optimized density field, one shading step per 1/7 of density.

    A 2D run is drawn element by element.  A 3D run is drawn as projections —
    the mean density through the width (side view), and, once the mesh is more
    than one element deep in a direction, the top and end views as well.  With
    one element through the width the side view is exactly the element field.
    A body-fitted 3D mesh (unstructured in x-y, extruded in z) has no grid
    columns across x or y, so it gets the side view only: each front element
    shaded by the mean of the elements stacked behind it.
    """
    mesh_data = np.load(mesh_npz)
    nodes, cells = mesh_data["nodes"], mesh_data["cells"]
    densities = np.asarray(np.load(density_npz)["densities"], dtype=float)
    footer = f"{densities.size} design variables, one density per element"
    if nodes.shape[1] == 3 and cells.shape[1] in (4, 10):  # tetrahedra: the front surface
        xy, tris, owner = _side_view(nodes, cells)
        caption = "front surface (z = min), each facet shaded by its element"
        return _quad_field_svg(
            nodes=xy,
            quads=tris,
            bins=_density_bins(densities[owner], n_steps),
            width=width,
            pad=pad,
            n_steps=n_steps,
            aria=f"Optimized density, {caption}",
            footer=f"{caption} — {footer}",
        )
    if cells.shape[1] != 8:
        return _quad_field_svg(
            nodes=nodes,
            quads=cells,
            bins=_density_bins(densities, n_steps),
            width=width,
            pad=pad,
            n_steps=n_steps,
            aria="Optimized density field of the cantilever beam",
            footer=footer,
        )

    if not structured:
        xy, quads, front = _side_view(nodes, cells)
        caption = PROJECTIONS["side"][3]
        return _quad_field_svg(
            nodes=xy,
            quads=quads,
            bins=_density_bins(_stack_means(nodes, cells, densities)[front], n_steps),
            width=width,
            pad=pad,
            n_steps=n_steps,
            aria=f"Optimized density, {caption}",
            footer=f"{caption} — {footer}",
        )

    parts = []
    for name, (axis, u, v, caption) in PROJECTIONS.items():
        points, rects, means = project_field(nodes, cells, densities, axis, u, v)
        across = [np.unique(np.round(points[:, i], 9)).size - 1 for i in (0, 1)]
        if name != "side" and min(across) < 2:
            continue  # a strip one element thick says nothing the side view does not
        span_u, span_v = np.ptp(points, axis=0)
        # keep tall views (the end view) within max_height
        w = int(min(width, (max_height - 2 * pad) * span_u / max(span_v, 1e-12) + 2 * pad))
        parts.append(
            _quad_field_svg(
                nodes=points,
                quads=rects,
                bins=_density_bins(means, n_steps),
                width=max(w, 160),
                pad=pad,
                n_steps=n_steps,
                aria=f"Optimized density, {caption}",
                footer=caption + (f" — {footer}" if name == "side" else ""),
            )
        )
    return "\n".join(parts)


def _stack_means(nodes: np.ndarray, cells: np.ndarray, field: np.ndarray) -> np.ndarray:
    """Per element, the mean of ``field`` over the elements sharing its x-y
    footprint — the stack through the width of an extruded hex mesh."""
    keys = np.round(nodes[cells][:, :, :2].mean(axis=1), 6)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    return (np.bincount(inverse, weights=field) / np.bincount(inverse))[inverse]


def _density_bins(values: np.ndarray, n_steps: int) -> np.ndarray:
    return np.clip((np.asarray(values) * n_steps).astype(int), 0, n_steps - 1)


# --------------------------------------------------------------------------
# convergence charts
# --------------------------------------------------------------------------
def _nice_ticks(lo: float, hi: float, count: int = 4) -> list[float]:
    """Round tick values (1, 2, 5 x 10^k) spanning [lo, hi]."""
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(
        (m * magnitude for m in (1.0, 2.0, 5.0, 10.0) if m * magnitude >= raw), 10 * magnitude
    )
    start = np.floor(lo / step) * step
    ticks = []
    value = start
    while value <= hi + 0.5 * step:
        if value >= lo - 1e-12:
            ticks.append(float(value))
        value += step
    return ticks


def line_chart_svg(
    xs: list[float],
    ys: list[float],
    title: str,
    value_label: str,
    log_y: bool = False,
    reference: tuple[float, str] | None = None,
    width: int = 430,
    height: int = 210,
) -> str:
    """A single-series line chart: 2px line, hairline grid, one end label.

    One series, so no legend box — the title names what is plotted. Values that
    are not directly labelled live in the iteration table below the charts.
    """
    left, right, top, bottom = 54, 62, 18, 30
    plot_w = width - left - right
    plot_h = height - top - bottom

    def ty(value: float) -> float:
        return np.log10(max(value, 1e-12)) if log_y else value

    y_vals = [ty(v) for v in ys]
    y_lo, y_hi = min(y_vals), max(y_vals)
    if reference is not None:
        y_lo = min(y_lo, ty(reference[0]))
    if y_hi - y_lo < 1e-12:
        y_hi = y_lo + 1.0
    pad_y = 0.08 * (y_hi - y_lo)
    y_lo, y_hi = y_lo - pad_y, y_hi + pad_y

    x_lo, x_hi = min(xs), max(xs)
    x_span = max(x_hi - x_lo, 1e-12)

    def sx(x: float) -> float:
        return left + (x - x_lo) / x_span * plot_w

    def sy(v: float) -> float:
        return top + (y_hi - v) / (y_hi - y_lo) * plot_h

    if log_y:
        decades = range(int(np.floor(y_lo)), int(np.ceil(y_hi)) + 1)
        y_ticks = [(10.0**d, f"1e{d}") for d in decades if y_lo <= d <= y_hi]
    else:
        y_ticks = [(t, f"{t:,.0f}") for t in _nice_ticks(y_lo, y_hi)]

    parts = [
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="{_e(title)}">',
        "<style>"
        ".grid{stroke:var(--border);stroke-width:1;fill:none}"
        ".ln{stroke:var(--seq-4);stroke-width:2;fill:none;stroke-linejoin:round;"
        "stroke-linecap:round}"
        ".dot{fill:var(--seq-4);stroke:var(--panel);stroke-width:2}"
        ".ax{fill:var(--muted);font:11px ui-monospace,monospace}"
        ".ttl{fill:var(--text);font:600 12px -apple-system,BlinkMacSystemFont,sans-serif}"
        ".ref{stroke:var(--muted);stroke-width:1;fill:none}"
        "</style>",
        f'<text class="ttl" x="{left - 40}" y="12">{_e(title)}</text>',
    ]

    for value, label in y_ticks:
        y = sy(ty(value))
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="ax" x="{left - 8}" y="{y + 3.5:.1f}" text-anchor="end">{_e(label)}</text>'
        )

    if reference is not None:
        ref_value, ref_label = reference
        y = sy(ty(ref_value))
        parts.append(f'<line class="ref" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        parts.append(f'<text class="ax" x="{left + 4}" y="{y - 5:.1f}">{_e(ref_label)}</text>')

    points = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, y_vals))
    parts.append(f'<polyline class="ln" points="{points}"/>')
    parts.append(f'<circle class="dot" cx="{sx(xs[-1]):.1f}" cy="{sy(y_vals[-1]):.1f}" r="4.5"/>')
    parts.append(
        f'<text class="ax" x="{sx(xs[-1]) + 9:.1f}" y="{sy(y_vals[-1]) + 4:.1f}">'
        f"{_e(value_label)}</text>"
    )

    baseline = top + plot_h
    parts.append(f'<line class="grid" x1="{left}" y1="{baseline}" x2="{left + plot_w}" y2="{baseline}"/>')
    for x in (x_lo, x_hi):
        parts.append(
            f'<text class="ax" x="{sx(x):.1f}" y="{baseline + 16:.1f}" text-anchor="middle">'
            f"{int(x)}</text>"
        )
    parts.append(
        f'<text class="ax" x="{left + plot_w / 2:.1f}" y="{height - 4:.1f}" text-anchor="middle">'
        "iteration</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def read_history(csv_path: Path) -> list[dict[str, float]]:
    """Parse the loop's ``log.csv`` — the report reads the artifact, not the loop."""
    rows: list[dict[str, float]] = []
    lines = Path(csv_path).read_text().strip().splitlines()
    if not lines:
        return rows
    header = lines[0].split(",")
    for line in lines[1:]:
        values = line.split(",")
        rows.append({k: float(v) for k, v in zip(header, values)})
    return rows


def _history_table(history: list[dict[str, float]], every: int = 5) -> str:
    """Table view of the convergence charts — every value, nothing gated by hover."""
    picked = [
        row
        for i, row in enumerate(history)
        if i == 0 or i == len(history) - 1 or (i + 1) % every == 0
    ]
    rows = "".join(
        f"<tr><td class='num'>{int(r['iteration'])}</td>"
        f"<td class='num'>{r['compliance']:.4g}</td>"
        f"<td class='num'>{r['volume_fraction']:.4f}</td>"
        f"<td class='num'>{r['change']:.4f}</td>"
        f"<td class='num'>{r['measure_of_discreteness']:.1f}</td></tr>"
        for r in picked
    )
    return (
        "<details><summary>Iteration table (every "
        f"{every}th iteration, plus the first and last)</summary>"
        "<table><thead><tr><th class='num'>Iter</th><th class='num'>Compliance</th>"
        "<th class='num'>Volume</th><th class='num'>Max change</th>"
        "<th class='num'>Mnd %</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></details>"
    )


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def _kv_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{_e(k)}</td><td class='num'>{_e(_fmt(v))}</td></tr>" for k, v in rows
    )
    return f"<table><thead><tr><th>Parameter</th><th class='num'>Value</th></tr></thead><tbody>{body}</tbody></table>"


def _cards(cards: list[tuple[str, Any, str]]) -> str:
    body = "".join(
        f"<div class='card'><div class='k'>{_e(key)}</div>"
        f"<div class='v'>{_e(value)}</div>"
        f"<div class='s'>{_e(sub)}</div></div>"
        for key, value, sub in cards
    )
    return f"<div class='grid'>{body}</div>"


def _solve_cards(solve: dict[str, Any]) -> str:
    beam = solve.get("beam_theory") or {}
    return _cards(
        [
            (
                "Compliance",
                f"{solve.get('compliance', 0):.4g}",
                "N\u00b7mm, F\u00b7U at full density",
            ),
            (
                "uy at the load",
                f"{solve.get('tip_uy', 0):.4g} mm",
                f"beam theory {beam.get('total', 0):.4g} mm"
                f" ({(solve.get('beam_theory_rel_diff') or 0) * 100:.2f}% off)"
                if beam else "mean over the load region's nodes",
            ),
            (
                "Peak von Mises",
                f"{solve.get('von_mises_max', 0):.4g} MPa",
                f"min {solve.get('von_mises_min', 0):.3g} MPa at the free end",
            ),
            (
                "Equilibrium",
                f"{solve.get('equilibrium_residual', 0):.1e}",
                f"||KU-F|| on {solve.get('n_free_dofs', 0)} free DOFs",
            ),
        ]
    )


def _optimization_cards(opt: dict[str, Any], solve: dict[str, Any]) -> str:
    ratio = opt.get("compliance_ratio_to_full_density")
    return _cards(
        [
            (
                "Compliance",
                f"{opt.get('compliance', 0):.4g}",
                "N\u00b7mm at the optimized design"
                + (f", {ratio:.2f}x the solid beam" if ratio else ""),
            ),
            (
                "Material used",
                f"{opt.get('volume_fraction', 0) * 100:.1f}%",
                "of the design domain, the volume constraint",
            ),
            (
                "Iterations",
                opt.get("iterations", "-"),
                "converged" if opt.get("converged") else "hit the iteration cap",
            ),
            (
                "Discreteness",
                f"{opt.get('measure_of_discreteness', 0):.1f}%",
                f"Mnd; {opt.get('solid_fraction', 0) * 100:.0f}% solid,"
                f" {opt.get('void_fraction', 0) * 100:.0f}% void",
            ),
        ]
    )


def _stat_cards(summary: dict[str, Any], params: dict[str, Any]) -> str:
    domain = params.get("domain", {})
    mesh = params.get("mesh", {})
    return _cards(
        [
            (
                "Elements",
                summary.get("n_elements", "-"),
                f"{mesh.get('nelx', '?')} x {mesh.get('nely', '?')}"
                + (f" x {mesh['nelz']} hexahedra" if "nelz" in mesh else " quads")
                if mesh.get("mode", "structured") == "structured"
                else "body-fitted "
                + (f"hexahedra, {mesh['nelz']} layer(s) in z" if "nelz" in mesh else "quads"),
            ),
            (
                "Nodes",
                summary.get("n_nodes", "-"),
                f"{summary.get('n_dofs', '-')} displacement DOFs",
            ),
            (
                "Element size",
                f"{summary.get('edge_length_max', 0):.3g} mm",
                f"max aspect ratio {summary.get('aspect_ratio_max', 0):.3f}",
            ),
            (
                "Domain",
                f"{domain.get('length', '?')} x {domain.get('height', '?')}",
                f"{summary['volume_sum']:g} mm\u00b3 meshed"
                if "volume_sum" in summary
                else f"{summary.get('area_sum', 0):g} mm\u00b2 meshed",
            ),
        ]
    )


def _steps_html(steps: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for step in steps:
        lines = step.get("lines", [])
        plain = [l for l in lines if not l.startswith("[gmsh]")]
        gmsh_lines = [l[len("[gmsh] "):] for l in lines if l.startswith("[gmsh]")]
        ok = step.get("status") == "ok"
        dur = step.get("duration_s")
        badge = "ok" if ok else _e(step.get("status", "?"))
        out.append(
            f"<section class='step'><div class='head'>"
            f"<h3>{_e(step.get('title', step.get('name', '')))}</h3>"
            f"<span class='badge{'' if ok else ' fail'}'>{badge}</span>"
            f"<span class='dur'>{'' if dur is None else f'{dur:.2f} s'}</span></div>"
        )
        if plain:
            out.append(f"<pre>{_e(chr(10).join(plain))}</pre>")
        if gmsh_lines:
            out.append(
                f"<details><summary>Gmsh output ({len(gmsh_lines)} lines)</summary>"
                f"<pre>{_e(chr(10).join(gmsh_lines))}</pre></details>"
            )
        out.append("</section>")
    return "".join(out)


def _checks_html(summary: dict[str, Any]) -> str:
    checks = summary.get("checks", {})
    rows = "".join(
        f"<tr><td><code>{_e(name)}</code></td>"
        f"<td class='num'><span class='{'pass' if ok else 'fail'}'>"
        f"{'PASS' if ok else 'FAIL'}</span></td></tr>"
        for name, ok in checks.items()
    )
    return f"<table><thead><tr><th>Check</th><th class='num'>Result</th></tr></thead><tbody>{rows}</tbody></table>"


def _artifacts_html(run: dict[str, Any], copied: dict[str, str]) -> str:
    rows: list[str] = []
    for step in run.get("steps", []):
        for art in step.get("artifacts", []):
            name = Path(art["path"]).name
            link = copied.get(name)
            label = f"<a href='{_e(link)}'>{_e(name)}</a>" if link else _e(name)
            rows.append(
                f"<tr><td>{label}</td><td>{_e(art.get('description', ''))}</td>"
                f"<td class='num'>{art.get('bytes', 0) / 1024:.1f} kB</td></tr>"
            )
    return (
        "<table><thead><tr><th>File</th><th>Contents</th><th class='num'>Size</th></tr>"
        f"</thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _cad_section(
    geometry: dict[str, Any], params: dict[str, Any], copied: dict[str, str],
    cad_image: str | None,
) -> str:
    """The CadQuery input (parameters and the script that ran) and its output."""
    script = geometry.get("cad_script")
    if not script and cad_image is None:
        return ""
    domain = params.get("domain", {})
    three_d = params.get("dim") == 3
    scripted = domain.get("source") == "script"
    rows: list[tuple[str, Any]] = [
        ("dimension", ("3D solid" if scripted else "3D solid box") if three_d else "2D planar face"),
        ("length L, along x (mm)", domain.get("length")),
        ("height H, along y (mm)", domain.get("height")),
    ]
    if three_d:
        rows.append(("width W, along z (mm)", domain.get("width")))
        rows.append(("volume (mm\u00b3)", geometry.get("solid_volume", domain.get("volume"))))
    else:
        rows.append(("thickness, solver only (mm)", domain.get("thickness")))
        rows.append(("area (mm\u00b2)", geometry.get("face_area", domain.get("area"))))
    holes = domain.get("holes") or []
    if scripted:
        rows.append(("x-y profile area (mm\u00b2)", domain.get("material_area")))
        rows.append(("regions", ", ".join(domain.get("regions") or []) or "none"))
        rows.append(("geometry from", domain.get("script_path") or "a CadQuery script"))
    else:
        rows.append((
            "cutouts through z (x, y, \u2300 mm)",
            "; ".join(f"({x:g}, {y:g}, \u2300{d:g})" for x, y, d in holes) if holes else "none",
        ))
        rows.append(("parameters from", "command-line flags / defaults"))
    cut_away = holes or (
        scripted and (domain.get("material_area") or 0) < (domain.get("area") or 0) * (1 - 1e-6)
    )

    image = ""
    if cad_image is not None:
        shape = ("solid" if scripted else "box") if three_d else "planar face"
        image = (
            f"<figure><img src='{_e(cad_image)}' alt='Shaded view of the CadQuery design"
            f" domain: a {_e(domain.get('length'))} by {_e(domain.get('height'))} mm {shape}'>"
            "<figcaption>The CadQuery output, read back from <code>design_domain.brep</code> and"
            " rendered by <code>topocombo.cadview</code>"
            + (
                ". The structured grid covers its envelope and holds the elements outside the"
                " part void."
                if cut_away and params.get("mesh", {}).get("mode") == "structured"
                else " — the exact shape Gmsh meshed."
            )
            + "</figcaption>"
            "</figure>"
        )
    table = f"<div>{_kv_table(rows)}</div>"
    code = ""
    if script:
        link = copied.get("design_domain.py")
        code = (
            "<div class='script'><div class='head'>"
            "<span><code>design_domain.py</code> — the CadQuery input, exactly as it ran</span>"
            + (f"<a href='{_e(link)}' download>download</a>" if link else "")
            + f"</div><pre><code>{_e(script)}</code></pre></div>"
        )
    return (
        "<h2>Geometry (CadQuery)</h2>"
        + (
            "<p class='lede'>The design domain is code: the CadQuery part script below builds"
            " the part and names its regions — the places the study holds and loads it. Every"
            " later stage reads the shape it produced (its bounding box, its x-y profile,"
            " whatever it cut away) rather than any parameters. It opens as-is in"
            " CQ-editor.</p>"
            if scripted else
            "<p class='lede'>The design domain is built by running a generated CadQuery script,"
            " so the input to CAD is explicit and reproducible: the parameters on the left are"
            " written into the script below, which also opens as-is in CQ-editor. For any other"
            " geometry, write a part script and a study and pass it with <code>--study</code>.</p>"
        )
        + f"<div class='cad'>{table}{image}</div>"
        + code
    )


def _fmt_vec(values: Any) -> str:
    return "(" + ", ".join(_fmt(float(v)) for v in values) + ")"


def _fmt_axes(symbol: str, axes: str) -> str:
    return "(" + ", ".join(f"{symbol}{a}" for a in axes) + ")"


def _nodes(count: Any) -> str:
    return "? nodes" if count is None else f"{count} node{'' if count == 1 else 's'}"


def _study_inputs(params: dict[str, Any], summary: dict[str, Any], run: dict[str, Any],
                  geometry: dict[str, Any], copied: dict[str, str]) -> str:
    """What the run was told: mesh size, loads and displacement constraints —
    and, for a study, the study script itself.  Read-only: the study script is
    where these are set."""
    mesh = params.get("mesh", {})
    three_d = params.get("dim") == 3
    axes = "xyz"[: 3 if three_d else 2]
    structured = mesh.get("mode", "structured") == "structured"
    study = params.get("study")
    sets = summary.get("node_sets", {})

    size = mesh.get("element_size")
    if size is None:  # runs from before the size was recorded
        fitted = _mesh_step_data(run).get("element_size")
        size = [fitted] if fitted else None
    mesh_rows: list[tuple[str, Any]] = [("mode", mesh.get("mode", "structured"))]
    if structured:
        mesh_rows += [(f"elements along {a}", mesh.get(f"nel{a}")) for a in axes]
        if size:
            mesh_rows.append((
                "element size " + " \u00d7 ".join(f"d{a}" for a in axes) + " (mm)",
                " \u00d7 ".join(f"{v:.4g}" for v in size),
            ))
    else:
        mesh_rows.append(("target element edge (mm)", size[0] if size else None))
        if three_d and mesh.get("nelz") is not None:  # tetrahedra have no layers
            mesh_rows.append(("layers through the width", mesh.get("nelz")))
            if size and len(size) > 1:
                mesh_rows.append(("layer thickness dz (mm)", size[1]))
    mesh_rows.append(("result", f"{summary.get('n_elements', '?')} elements, "
                                f"{summary.get('n_nodes', '?')} nodes"))

    bcs = params.get("boundary_conditions") or {}
    load_rows: list[tuple[str, Any]] = []
    loads = bcs.get("loads", [])
    several = len({ld.get("case", "load") for ld in loads}) > 1
    for load in loads:
        force = load.get("force", {})
        name = load.get("region") or load.get("node_set", "load")
        case = (
            f"case '{load.get('case')}' (weight {_fmt(load.get('weight', 1.0))}): "
            if several else ""
        )
        load_rows += [
            (f"{case}'{name}': {load.get('kind', '')} region".replace(":  region", ""),
             _nodes(sets.get(name))),
            ("force " + _fmt_axes("F", axes) + " (N, total)",
             _fmt_vec(force.get(f"f{a}", 0.0) for a in axes)),
        ]
    fix_rows: list[tuple[str, Any]] = []
    for c in bcs.get("constraints", []):
        name = c.get("region") or c.get("node_set", "fixed")
        disp = c.get("displacements", {})
        kind = {"clamped": "clamped", "held": "held", "prescribed": "prescribed"}.get(
            c.get("type", "clamped"), c.get("type")
        )
        fix_rows += [
            (f"'{name}': {c.get('kind', '')} region".replace(":  region", ""),
             f"{_nodes(sets.get(name))}, {kind}"),
            ("displacement " + _fmt_axes("u", axes) + " (mm)",
             "(" + ", ".join(
                 _fmt(float(disp[f"u{a}"])) if f"u{a}" in disp else "free" for a in axes
             ) + ")"),
        ]
    passive_rows: list[tuple[str, Any]] = []
    passive_info = {p.get("region"): p for p in summary.get("passive_regions", [])}
    for p in params.get("passive") or []:
        within = f", within {_fmt(float(p.get('within', 0.0)))} mm" if p.get("within") else ""
        count = passive_info.get(p.get("region"), {}).get("elements", "?")
        passive_rows.append(
            (f"'{p.get('region')}': held {p.get('state')}{within}", f"{count} elements")
        )

    def block(title: str, rows: list[tuple[str, Any]]) -> str:
        return f"<div><h3>{_e(title)}</h3>{_kv_table(rows)}</div>"

    source = geometry.get("study_script")
    code = ""
    if source:
        link = copied.get("study.py")
        code = (
            "<div class='script'><div class='head'>"
            "<span><code>study.py</code> — the study, exactly as it ran</span>"
            + (f"<a href='{_e(link)}' download>download</a>" if link else "")
            + f"</div><pre><code>{_e(source)}</code></pre></div>"
        )
    lede = (
        f"Set in <code>{_e(study.get('path'))}</code>, which refers to the regions the part"
        " script names."
        if study else
        "The parametric cantilever: set with command-line flags; the clamp and the tip load"
        " are the beam's own <code>fixed</code> and <code>load</code> regions."
    )
    return (
        f"<p class='lede'>{lede}</p>"
        "<div class='inputs'>"
        + block("Mesh size", mesh_rows)
        + block("Loads", load_rows)
        + block("Displacement constraints", fix_rows)
        + (block("Passive regions", passive_rows) if passive_rows else "")
        + "</div>"
        + code
    )


def _mesh_step_data(run: dict[str, Any]) -> dict[str, Any]:
    for step in run.get("steps", []):
        if step.get("name") == "meshing":
            return step.get("data", {})
    return {}


def render_html(
    run: dict[str, Any],
    svg: str,
    copied: dict[str, str],
    solve_figure: tuple[str, list[float]] | None = None,
    density_figure: str | None = None,
    history: list[dict[str, float]] | None = None,
    cad_image: str | None = None,
    topology_image: str | None = None,
    nav: list[tuple[str, str]] | None = None,
) -> str:
    params = run.get("params", {})
    domain = params.get("domain", {})
    mesh = params.get("mesh", {})
    three_d = params.get("dim") == 3
    # the run's element, from the registry (older runs recorded only the label)
    label = params.get("element") or ("H8 hexahedron" if three_d else "Q4 quadrilateral")
    el = next((e for e in ELEMENTS.values() if e.label == label), None)
    noun = label.split(" ", 1)[-1]
    plural = el.plural if el else noun + "s"
    fitted = mesh.get("mode", "structured") == "body-fitted"
    tets = (el is not None and el.cell_type.startswith("tetra")) or "tetra" in label.lower()
    meshed_as = f"{'body-fitted' if fitted else 'structured'} {plural}"
    solved_as = "a 3D solid problem" if three_d else "a plane-stress problem"
    env = run.get("environment", {})
    tools = env.get("tools", {})

    summary: dict[str, Any] = {}
    solve: dict[str, Any] = {}
    optimization: dict[str, Any] = {}
    geometry: dict[str, Any] = {}
    for step in run.get("steps", []):
        if step.get("name") == "geometry":
            geometry = step.get("data", {})
        elif step.get("name") == "validation":
            summary = step.get("data", {})
        elif step.get("name") == "solve":
            solve = step.get("data", {})
        elif step.get("name") == "optimize":
            optimization = step.get("data", {})

    bcs = params.get("boundary_conditions") or {}
    fixed_names = [c.get("region") or c.get("node_set", "fixed") for c in bcs.get("constraints", [])]
    def _holds(c: dict[str, Any]) -> str:
        name = _e(c.get("region") or c.get("node_set", "fixed"))
        if c.get("type", "clamped") == "clamped":
            return f"{name} is clamped (hatched)"
        comps = ", ".join(
            f"{k} = {_fmt(v)}" for k, v in (c.get("displacements") or {}).items()
        )
        glyph = "" if c.get("type") == "prescribed" else " (triangles)"
        return f"{name} holds {comps}{glyph}"

    load_names = list(dict.fromkeys(
        f.get("region") or f.get("node_set", "load") for f in bcs.get("loads", [])
    ))
    holds = "; ".join(_holds(c) for c in bcs.get("constraints", []))
    held = (
        (holds[:1].upper() + holds[1:]) if holds else "The fixed region is clamped"
    ) + (
        f"; the arrow marks the load on {'/'.join(_e(n) for n in load_names) or 'load'}"
        + (" (first load case)" if len(params.get("load_cases") or []) > 1 else "")
        + "."
    )
    every = f"every {noun}" + (" of the body-fitted mesh" if fitted else "")
    if three_d:
        mesh_caption = f"Side view: {every} is one design variable for the SIMP loop. " + held
    else:
        mesh_caption = every[0].upper() + every[1:] + " is one design variable for the SIMP loop. " + held
    if fitted:
        mesh_caption += (
            " The mesh follows the CAD boundary, so the dashed cutout is a real hole in it."
            if params.get("domain", {}).get("holes") else ""
        )
    if summary.get("passive_elements"):
        mesh_caption += (
            f" The {_e(summary['passive_elements'])} elements whose centres fall "
            + ("inside the dashed CAD cutout" if domain.get("holes") else "outside the CAD part")
            + " are drawn empty: the grid still covers them, but the optimizer holds them void."
        )

    chips = [f"run {run.get('started_at', '')}", f"{run.get('duration_s', 0):.2f} s total"]
    chips += [f"{name} {version}" for name, version in tools.items()]
    chips.append(f"python {env.get('python', '?')}")

    param_rows = [
        ("length L, bounding box (mm)", domain.get("length")),
        ("height H, bounding box (mm)", domain.get("height")),
        ("out-of-plane thickness (mm)", domain.get("thickness")),
        ("aspect ratio L/H", domain.get("aspect_ratio")),
        ("mesh mode", mesh.get("mode", "structured")),
        *(
            [("elements along x (nelx)", mesh.get("nelx")),
             ("elements along y (nely)", mesh.get("nely"))]
            if mesh.get("mode", "structured") == "structured"
            else [("target element size (mm)", _mesh_step_data(run).get("element_size"))]
        ),
        ("load region", ", ".join(load_names)),
        ("load nodes", len(summary.get("load_nodes") or [])),
        ("constrained regions", ", ".join(fixed_names)),
    ]
    material = params.get("material", {})
    simp = params.get("simp", {})
    if material:
        param_rows += [
            ("Young's modulus E (MPa)", material.get("youngs_modulus")),
            ("Poisson's ratio", material.get("poisson_ratio")),
            ("force (N)", _fmt_vec(params.get("load", {}).get(f"f{a}", 0.0)
                                   for a in "xyz"[: 3 if three_d else 2])),
        ]
    if simp:
        param_rows += [
            ("target volume fraction", simp.get("volume_fraction")),
            ("SIMP penalty p", simp.get("penal")),
            ("filter", f"{simp.get('filter_type')}, r = {simp.get('filter_radius')} mm"),
            ("move limit", simp.get("move_limit")),
            ("convergence tolerance", simp.get("tolerance")),
        ]

    solve_section = ""
    if solve:
        figure = ""
        if solve_figure is not None:
            svg_solve, edges = solve_figure
            figure = (
                "<figure>"
                + svg_solve
                + _ramp_legend(edges, "element compliance u_e^T k_e u_e (N\u00b7mm, log scale)")
                + "<figcaption>Strain energy concentrates at the clamped corners and falls to"
                " near zero along the neutral axis. This per-element field is what the SIMP"
                " sensitivities are built from: the high-energy bands near the top and bottom"
                " fibres are where material earns its place, and the low-energy core is what the"
                " optimizer carves away first.</figcaption>"
                "</figure>"
            )
        solve_section = (
            f"<h2>{'3D solid' if three_d else 'Plane-stress'} solve</h2>"
            "<p class='lede'>Full density (every element solid) — the starting point of the"
            " optimization, and the case where an analytical answer exists to check against.</p>"
            + _solve_cards(solve)
            + figure
        )

    optimization_section = ""
    if optimization:
        charts = ""
        if history:
            xs = [row["iteration"] for row in history]
            compliance_chart = line_chart_svg(
                xs,
                [row["compliance"] for row in history],
                title="Compliance per iteration (N\u00b7mm)",
                value_label=f"{history[-1]['compliance']:.0f}",
            )
            change_chart = line_chart_svg(
                xs,
                [max(row["change"], 1e-6) for row in history],
                title="Max density change (log scale)",
                value_label=f"{history[-1]['change']:.3f}",
                log_y=True,
                reference=(simp.get("tolerance", 0.01), f"tol {simp.get('tolerance', 0.01):g}"),
            )
            charts = (
                "<div class='grid'>"
                f"<div class='card'>{compliance_chart}</div>"
                f"<div class='card'>{change_chart}</div>"
                "</div>" + _history_table(history)
            )
        density_fig = ""
        if density_figure is not None:
            density_fig = (
                "<figure>"
                + density_figure
                + _ramp_legend([0.0, 1.0], "element density x (0 = void, 1 = solid)")
                + "<figcaption>The optimizer keeps material where it carries load, in members"
                " running from the load to the constrained region; for the cantilever that is"
                " flanges top and bottom with a triangulated web. Intermediate densities are the"
                " \u201cgrey\u201d that SIMP's penalty pushes towards 0 or 1 — the Mnd figure above"
                " says how much of it is left."
                + (" A tetrahedral mesh is shown by its front surface only; the solid below"
                   " shows the design in 3D." if tets else "")
                + "</figcaption>"
                "</figure>"
            )
        topology_fig = ""
        if topology_image is not None:
            topology_fig = (
                f"<figure><img src='{_e(topology_image)}' alt='Shaded 3D view of the optimized"
                " topology'>"
                "<figcaption>The optimized design as a solid: every element at density"
                " \u2265 0.5, read back from <code>topology.stl</code> — the file to take into"
                " Blender or CAD. It is blocky by construction; each step is one element."
                "</figcaption></figure>"
            )
        optimization_section = (
            "<h2>Topology optimization</h2>"
            "<p class='lede'>SIMP compliance minimisation under a volume constraint:"
            f" p = {_e(simp.get('penal', 3))}, {_e(simp.get('filter_type', ''))} filter of radius"
            f" {_e(simp.get('filter_radius', ''))} mm, Optimality Criteria update with a"
            f" {_e(simp.get('move_limit', ''))} move limit.</p>"
            + _optimization_cards(optimization, solve)
            + density_fig
            + topology_fig
            + charts
        )

    all_ok = summary.get("all_checks_passed", False)
    study_path = (params.get("study") or {}).get("path")
    case_name = Path(study_path).parent.name if study_path else "cantilever beam"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>topocombo — {_e(case_name)}</title>
<meta name="description" content="Procedure log of the topocombo {_e(case_name)} run: CadQuery geometry, Gmsh mesh, FEA solve and SIMP optimization.">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>{_e(case_name[:1].upper() + case_name[1:])} — mesh, solve, optimize</h1>
  <p class="lede">
    The <a href="https://github.com/anroleroux/topocombo">topocombo</a> pipeline so far, run end
    to end: a design domain defined in CadQuery, exported to BREP, meshed into
    {meshed_as} with Gmsh, solved as {solved_as} with the custom solver,
    and driven through a SIMP compliance-minimisation loop until the density field converges.
    Every number and figure below comes from the artifacts this run wrote to disk.
  </p>
  <div class="meta">{''.join(f"<span class='chip'>{_e(c)}</span>" for c in chips)}</div>
  {("<nav class='meta' aria-label='Other runs'>" + "".join(f"<a class='chip' href='{_e(href)}'>{_e(label)}</a>" for label, href in nav) + "</nav>") if nav else ""}
</header>

<h2>Result</h2>
{_stat_cards(summary, params)}

{_cad_section(geometry, params, copied, cad_image)}

<h2>Mesh</h2>
{_study_inputs(params, summary, run, geometry, copied)}
<figure>
{svg}
<figcaption>
  {mesh_caption} Drawn directly from
  <code>mesh.npz</code> — the same file the solver reads.
</figcaption>
</figure>

{solve_section}

{optimization_section}

<h2>Procedure</h2>
<p class="lede">Terminal output of the run, one block per stage, exactly as it was logged.</p>
{_steps_html(run.get("steps", []))}

<h2>Parameters</h2>
{_kv_table(param_rows)}

<h2>Mesh validation</h2>
<p class="lede">
  The mesh is rejected before anything downstream sees it unless all of these hold
  — overall: <span class="{'pass' if all_ok else 'fail'}">{'ALL PASSED' if all_ok else 'FAILED'}</span>.
</p>
{_checks_html(summary)}

<h2>Artifacts</h2>
<p class="lede">Written to the run directory; the visualisation scripts and the future solver read these, not the mesher.</p>
{_artifacts_html(run, copied)}

<footer>
  Generated by <code>python -m topocombo.cli report</code> from <code>run.json</code>,
  <code>mesh.npz</code>, <code>solution.npz</code>, <code>density.npz</code> and the loop's
  <code>log.csv</code> — no plotting happens inside the pipeline itself.
</footer>
</div>
</body>
</html>
"""


def build_site(run_dir: Path, site_dir: Path, nav: list[tuple[str, str]] | None = None) -> Path:
    """Render ``run_dir`` into a self-contained static site at ``site_dir``.

    ``nav`` adds links to other reports, as (label, relative URL) pairs.
    """
    run_dir = Path(run_dir)
    site_dir = Path(site_dir)
    run = json.loads((run_dir / "run.json").read_text())

    site_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = site_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, str] = {}
    for step in run.get("steps", []):
        for art in step.get("artifacts", []):
            src = Path(art["path"])
            if not src.exists():
                continue
            dst = artifacts_dir / src.name
            shutil.copyfile(src, dst)
            copied[src.name] = f"artifacts/{src.name}"
    for extra in ("run.json", "pipeline.log"):
        src = run_dir / extra
        if src.exists():
            shutil.copyfile(src, artifacts_dir / extra)
            copied[extra] = f"artifacts/{extra}"

    mesh_npz = run_dir / "mesh" / "mesh.npz"
    solution_npz = run_dir / "solution" / "solution.npz"
    svg = mesh_svg(mesh_npz, holes=run.get("params", {}).get("domain", {}).get("holes", ()))
    solve_figure = (
        solution_svg(mesh_npz, solution_npz) if solution_npz.exists() else None
    )

    density_npz = run_dir / "optimization" / "density.npz"
    structured = run.get("params", {}).get("mesh", {}).get("mode", "structured") == "structured"
    density_figure = (
        density_svg(mesh_npz, density_npz, structured=structured) if density_npz.exists() else None
    )

    history_csv = run_dir / "optimization" / "log.csv"
    history = read_history(history_csv) if history_csv.exists() else None

    from .cadview import render_brep, render_stl

    cad_image = topology_image = None
    brep = run_dir / "cad" / "design_domain.brep"
    if brep.exists():
        render_brep(brep, site_dir / "figures" / "cad_domain.png")
        cad_image = "figures/cad_domain.png"
    stl = run_dir / "optimization" / "topology.stl"
    if stl.exists():
        render_stl(stl, site_dir / "figures" / "topology.png")
        topology_image = "figures/topology.png"

    index = site_dir / "index.html"
    index.write_text(
        render_html(
            run, svg, copied, solve_figure, density_figure, history, cad_image, topology_image,
            nav,
        )
    )
    return index
