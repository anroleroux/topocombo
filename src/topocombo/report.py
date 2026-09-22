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
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151a; --panel: #1a1f26; --border: #2c333d; --text: #e6eaef;
    --muted: #9aa5b1; --accent: #6fbf93; --accent-soft: #1d2a24;
    --fail: #e07a7a; --code-bg: #0e1116;
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
    quads = data["quads"]
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


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def _kv_table(rows: list[tuple[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{_e(k)}</td><td class='num'>{_e(_fmt(v))}</td></tr>" for k, v in rows
    )
    return f"<table><thead><tr><th>Parameter</th><th class='num'>Value</th></tr></thead><tbody>{body}</tbody></table>"


def _stat_cards(summary: dict[str, Any], params: dict[str, Any]) -> str:
    domain = params.get("domain", {})
    mesh = params.get("mesh", {})
    cards = [
        (
            "Elements",
            summary.get("n_elements", "—"),
            f"{mesh.get('nelx', '?')} x {mesh.get('nely', '?')} quads",
        ),
        (
            "Nodes",
            summary.get("n_nodes", "—"),
            f"{summary.get('n_dofs', '—')} displacement DOFs",
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
    body = "".join(
        f"<div class='card'><div class='k'>{_e(key)}</div>"
        f"<div class='v'>{_e(value)}</div>"
        f"<div class='s'>{_e(sub)}</div></div>"
        for key, value, sub in cards
    )
    return f"<div class='grid'>{body}</div>"


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


def render_html(run: dict[str, Any], svg: str, copied: dict[str, str]) -> str:
    params = run.get("params", {})
    domain = params.get("domain", {})
    mesh = params.get("mesh", {})
    env = run.get("environment", {})
    tools = env.get("tools", {})

    summary: dict[str, Any] = {}
    for step in run.get("steps", []):
        if step.get("name") == "validation":
            summary = step.get("data", {})

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

    all_ok = summary.get("all_checks_passed", False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>topocombo — cantilever beam mesh</title>
<meta name="description" content="Procedure log of the CadQuery to Gmsh meshing run for the topocombo cantilever beam example.">
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>Cantilever beam — geometry and mesh</h1>
  <p class="lede">
    The first two stages of the <a href="https://github.com/anroleroux/topocombo">topocombo</a>
    pipeline, run end to end: a parametric design domain defined in CadQuery, exported to BREP,
    and meshed into structured quadrilaterals with Gmsh. The FEA solve and the SIMP loop are not
    part of this run — it stops at a validated mesh plus the boundary node sets they will need.
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
  Generated by <code>python -m topocombo.cli report</code> from <code>run.json</code> and
  <code>mesh.npz</code>. Next stages: plane-stress FEA solve, then the SIMP compliance-minimisation loop.
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

    svg = mesh_svg(run_dir / "mesh" / "mesh.npz")
    index = site_dir / "index.html"
    index.write_text(render_html(run, svg, copied))
    return index
