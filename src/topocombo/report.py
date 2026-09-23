"""Render a run directory into a static HTML page (for GitHub Pages).

This is a downstream consumer of the artifacts written by
:mod:`topocombo.pipeline` — it reads ``run.json`` and ``mesh.npz`` from disk and
never imports the solver or the mesher.  The mesh figure is emitted as inline
SVG so the published page needs no JavaScript, no CDN and no plotting library.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

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
figure svg { width: 100%; height: auto; display: block; }
figcaption { color: var(--muted); font-size: 0.85rem; margin-top: 10px; }
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


# --------------------------------------------------------------------------
# mesh figure
# --------------------------------------------------------------------------
def mesh_svg(npz_path: Path, width: int = 900, pad: int = 46) -> str:
    """Inline SVG of the quad mesh with the supports and the tip load marked."""
    data = np.load(npz_path)
    nodes = data["nodes"]
    quads = data["cells"]
    fixed = data["set_fixed"] if "set_fixed" in data else np.empty(0, dtype=int)
    load_node = int(data["load_node"][0]) if "load_node" in data else None

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
        f'role="img" aria-label="Quadrilateral mesh of the cantilever beam design domain">',
        '<style>'
        '.el{fill:var(--accent-soft);stroke:var(--accent);stroke-width:0.6;stroke-opacity:0.55}'
        '.bd{fill:none;stroke:var(--text);stroke-width:1.6}'
        '.sup{stroke:var(--text);stroke-width:1.6}'
        '.ld{stroke:var(--fail);stroke-width:2.4;fill:var(--fail)}'
        '.lbl{fill:var(--muted);font:12px ui-monospace,monospace}'
        '</style>',
    ]

    for quad in quads:
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in (px(nodes[i]) for i in quad))
        parts.append(f'<polygon class="el" points="{pts}"/>')

    # outline
    corners = [
        px(np.array([xmin, ymin])),
        px(np.array([xmax, ymin])),
        px(np.array([xmax, ymax])),
        px(np.array([xmin, ymax])),
    ]
    parts.append(
        '<polygon class="bd" points="'
        + " ".join(f"{x:.2f},{y:.2f}" for x, y in corners)
        + '"/>'
    )

    # clamped edge: hatching along the fixed boundary
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
            f'fixed ({fixed.size} nodes)</text>'
        )

    # tip load: downward arrow at the load node
    if load_node is not None:
        lx, ly = px(nodes[load_node])
        parts.append(f'<line class="ld" x1="{lx:.2f}" y1="{ly - 34:.2f}" x2="{lx:.2f}" y2="{ly:.2f}"/>')
        parts.append(
            f'<polygon class="ld" points="{lx:.2f},{ly:.2f} {lx - 5:.2f},{ly - 11:.2f} '
            f'{lx + 5:.2f},{ly - 11:.2f}"/>'
        )
        # keep the label inside the viewBox when the load sits on the right edge
        near_right = lx > width / 2
        anchor = "end" if near_right else "start"
        tx = lx - 8 if near_right else lx + 8
        parts.append(
            f'<text class="lbl" x="{tx:.2f}" y="{ly - 40:.2f}" text-anchor="{anchor}">'
            f'F (node {load_node})</text>'
        )

    # dimension labels
    parts.append(
        f'<text class="lbl" x="{width / 2:.2f}" y="{height - 12:.2f}" text-anchor="middle">'
        f'L = {span_x:g} mm, {int(quads.shape[0])} quads</text>'
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
    nodes = mesh_data["nodes"]
    quads = mesh_data["cells"]
    disp = sol["displacements"]
    field = np.asarray(sol["element_compliance"], dtype=float)

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
        parts.append(f'<line class="ld" x1="{lx:.2f}" y1="{ly - 34:.2f}" x2="{lx:.2f}" y2="{ly:.2f}"/>')
        parts.append(
            f'<polygon class="ld" points="{lx:.2f},{ly:.2f} {lx - 5:.2f},{ly - 11:.2f} '
            f'{lx + 5:.2f},{ly - 11:.2f}"/>'
        )

    parts.append(
        f'<text class="lbl" x="{width / 2:.2f}" y="{height - 12:.2f}" text-anchor="middle">'
        f'displacements exaggerated {exaggeration:.0f}x (dashed: undeformed domain)</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts), [float(10.0**e) for e in edges]


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
    mesh_npz: Path, density_npz: Path, width: int = 900, pad: int = 46, n_steps: int = 7
) -> str:
    """The optimized density field, one shading step per 1/7 of density."""
    mesh_data = np.load(mesh_npz)
    densities = np.asarray(np.load(density_npz)["densities"], dtype=float)
    bins = np.clip((densities * n_steps).astype(int), 0, n_steps - 1)
    return _quad_field_svg(
        nodes=mesh_data["nodes"],
        quads=mesh_data["cells"],
        bins=bins,
        width=width,
        pad=pad,
        n_steps=n_steps,
        aria="Optimized density field of the cantilever beam",
        footer=f"{densities.size} design variables, one density per element",
    )


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
    beam = solve.get("beam_theory", {})
    return _cards(
        [
            (
                "Compliance",
                f"{solve.get('compliance', 0):.4g}",
                "N\u00b7mm, F\u00b7U at full density",
            ),
            (
                "Tip deflection",
                f"{solve.get('tip_uy', 0):.4g} mm",
                f"beam theory {beam.get('total', 0):.4g} mm"
                f" ({solve.get('beam_theory_rel_diff', 0) * 100:.2f}% off)",
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
                f"{mesh.get('nelx', '?')} x {mesh.get('nely', '?')} quads",
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
                f"{summary.get('area_sum', 0):g} mm\u00b2 meshed",
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


def render_html(
    run: dict[str, Any],
    svg: str,
    copied: dict[str, str],
    solve_figure: tuple[str, list[float]] | None = None,
    density_figure: str | None = None,
    history: list[dict[str, float]] | None = None,
) -> str:
    params = run.get("params", {})
    domain = params.get("domain", {})
    mesh = params.get("mesh", {})
    env = run.get("environment", {})
    tools = env.get("tools", {})

    summary: dict[str, Any] = {}
    solve: dict[str, Any] = {}
    optimization: dict[str, Any] = {}
    for step in run.get("steps", []):
        if step.get("name") == "validation":
            summary = step.get("data", {})
        elif step.get("name") == "solve":
            solve = step.get("data", {})
        elif step.get("name") == "optimize":
            optimization = step.get("data", {})

    chips = [f"run {run.get('started_at', '')}", f"{run.get('duration_s', 0):.2f} s total"]
    chips += [f"{name} {version}" for name, version in tools.items()]
    chips.append(f"python {env.get('python', '?')}")

    param_rows = [
        ("beam length L (mm)", domain.get("length")),
        ("beam height H (mm)", domain.get("height")),
        ("out-of-plane thickness (mm)", domain.get("thickness")),
        ("aspect ratio L/H", domain.get("aspect_ratio")),
        ("elements along x (nelx)", mesh.get("nelx")),
        ("elements along y (nely)", mesh.get("nely")),
        ("load point (mm)", domain.get("load_point")),
        ("tip load node index", summary.get("load_node")),
        ("fixed node set size", summary.get("node_sets", {}).get("fixed")),
        ("load edge node set size", summary.get("node_sets", {}).get("load_edge")),
    ]
    material = params.get("material", {})
    simp = params.get("simp", {})
    if material:
        param_rows += [
            ("Young's modulus E (MPa)", material.get("youngs_modulus")),
            ("Poisson's ratio", material.get("poisson_ratio")),
            ("tip load Fy (N)", params.get("load", {}).get("fy")),
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
            "<h2>Plane-stress solve</h2>"
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
                + "<figcaption>The optimizer keeps material where it carries load: flanges top"
                " and bottom, a triangulated web, and members converging on the clamped edge and"
                " the load point. Intermediate densities are the \u201cgrey\u201d that SIMP's penalty"
                " pushes towards 0 or 1 — the Mnd figure above says how much of it is left."
                "</figcaption>"
                "</figure>"
            )
        optimization_section = (
            "<h2>Topology optimization</h2>"
            "<p class='lede'>SIMP compliance minimisation under a volume constraint:"
            f" p = {_e(simp.get('penal', 3))}, {_e(simp.get('filter_type', ''))} filter of radius"
            f" {_e(simp.get('filter_radius', ''))} mm, Optimality Criteria update with a"
            f" {_e(simp.get('move_limit', ''))} move limit.</p>"
            + _optimization_cards(optimization, solve)
            + density_fig
            + charts
        )

    all_ok = summary.get("all_checks_passed", False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>topocombo — cantilever beam</title>
<meta name="description" content="Procedure log of the topocombo cantilever beam run: CadQuery geometry, Gmsh quad mesh and plane-stress FEA solve.">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>Cantilever beam — mesh, solve, optimize</h1>
  <p class="lede">
    The <a href="https://github.com/anroleroux/topocombo">topocombo</a> pipeline so far, run end
    to end: a parametric design domain defined in CadQuery, exported to BREP, meshed into
    structured quadrilaterals with Gmsh, solved as a plane-stress problem with the custom solver,
    and driven through a SIMP compliance-minimisation loop until the density field converges.
    Every number and figure below comes from the artifacts this run wrote to disk.
  </p>
  <div class="meta">{''.join(f"<span class='chip'>{_e(c)}</span>" for c in chips)}</div>
</header>

<h2>Result</h2>
{_stat_cards(summary, params)}

<h2>Mesh</h2>
<figure>
{svg}
<figcaption>
  Every quad is one design variable for the SIMP loop. The left edge is clamped
  (all {_e(summary.get('node_sets', {}).get('fixed', '?'))} nodes, both DOFs); the tip load acts
  downwards at the mid-height node of the free edge. Drawn directly from
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


def build_site(run_dir: Path, site_dir: Path) -> Path:
    """Render ``run_dir`` into a self-contained static site at ``site_dir``."""
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
    svg = mesh_svg(mesh_npz)
    solve_figure = (
        solution_svg(mesh_npz, solution_npz) if solution_npz.exists() else None
    )

    density_npz = run_dir / "optimization" / "density.npz"
    density_figure = density_svg(mesh_npz, density_npz) if density_npz.exists() else None

    history_csv = run_dir / "optimization" / "log.csv"
    history = read_history(history_csv) if history_csv.exists() else None

    index = site_dir / "index.html"
    index.write_text(
        render_html(run, svg, copied, solve_figure, density_figure, history)
    )
    return index
