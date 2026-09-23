"""Standalone PyVista views of a finished run: ``python -m topocombo.viz``.

Reads only the artifacts in a run directory — never the solver — so it works
the same for 2D and 3D runs and for any future FEA backend.  It renders:

* ``topology.png``  the design thresholded at rho >= 0.5 (from ``density.vtu``)
* ``density.png``   the full density field on the mesh
* ``solution.png``  von Mises stress on the full-density solve, warped by the
  displacement

off screen by default, or opens an interactive window with ``--show``.
PyVista is an optional dependency (``pip install -e ".[viz]"``); headless
Linux also needs an off-screen OpenGL (``libosmesa6`` or ``libegl1``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

#: Run-directory files each view reads, relative to the run directory.
VIEW_SOURCES = {
    "topology": "optimization/density.vtu",
    "density": "optimization/density.vtu",
    "solution": "solution/solution.vtu",
}


def available_views(run_dir: Path) -> dict[str, Path]:
    """Views whose source file exists in ``run_dir`` -> that file."""
    run_dir = Path(run_dir)
    found = {name: run_dir / rel for name, rel in VIEW_SOURCES.items()}
    return {name: path for name, path in found.items() if path.exists()}


def _camera(plotter: Any, three_d: bool) -> None:
    """Look at the beam's side (x-y), y up; a 3D run is seen slightly obliquely
    so its width shows."""
    if three_d:
        plotter.view_vector((0.35, 0.3, 1.0), viewup=(0.0, 1.0, 0.0))
    else:
        plotter.view_xy()
    plotter.reset_camera()


def render(
    run_dir: Path,
    out_dir: Path | None = None,
    show: bool = False,
    threshold: float = 0.5,
    window_size: tuple[int, int] = (1200, 600),
) -> dict[str, Path]:
    """Render every available view; return view name -> written PNG (empty with ``show``)."""
    import pyvista as pv

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "figures"
    views = available_views(run_dir)
    if not views:
        raise FileNotFoundError(f"no solution or density .vtu files under {run_dir}")
    if not show:
        out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for name, source in views.items():
        grid = pv.read(source)
        three_d = bool(grid.bounds[5] - grid.bounds[4] > 0)
        plotter = pv.Plotter(off_screen=not show, window_size=window_size)
        if name == "topology":
            solid = grid.threshold(threshold, scalars="density")
            plotter.add_mesh(solid, color="#3a6ea5", show_edges=False, smooth_shading=False)
            plotter.add_mesh(grid.outline(), color="grey")
            plotter.add_text(f"rho >= {threshold:g}: {solid.n_cells} of {grid.n_cells} elements", font_size=10)
        elif name == "density":
            plotter.add_mesh(grid, scalars="density", cmap="Blues", clim=(0.0, 1.0))
        else:  # solution
            size = max(grid.bounds[1] - grid.bounds[0], grid.bounds[3] - grid.bounds[2])
            disp = grid.point_data["displacement"]
            peak = float(abs(disp).max()) or 1.0
            warped = grid.warp_by_vector("displacement", factor=0.12 * size / peak)
            plotter.add_mesh(warped, scalars="von_mises", cmap="viridis", log_scale=True)
            plotter.add_mesh(grid.outline(), color="grey")
        _camera(plotter, three_d)
        if show:
            plotter.show(title=f"topocombo — {name}")
        else:
            path = out_dir / f"{name}.png"
            plotter.screenshot(str(path))
            plotter.close()
            written[name] = path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="topocombo.viz", description="PyVista views of a finished topocombo run."
    )
    parser.add_argument("--run", type=Path, default=Path("results/cantilever"))
    parser.add_argument("--out", type=Path, default=None, help="PNG directory (default: <run>/figures)")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show", action="store_true", help="open interactive windows instead")
    args = parser.parse_args(argv)
    try:
        written = render(args.run, args.out, show=args.show, threshold=args.threshold)
    except ImportError:
        print("PyVista is not installed: pip install -e '.[viz]'", file=sys.stderr)
        return 1
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
