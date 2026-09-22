"""Terminal entry point: ``python -m topocombo.cli ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .geometry import BeamDomain
from .meshing import MeshSpec


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--length", type=float, default=60.0, help="beam length in mm (default: 60)")
    p.add_argument("--height", type=float, default=20.0, help="beam height in mm (default: 20)")
    p.add_argument(
        "--thickness", type=float, default=1.0, help="out-of-plane thickness in mm (default: 1)"
    )
    p.add_argument("--nelx", type=int, default=60, help="elements along the length (default: 60)")
    p.add_argument("--nely", type=int, default=20, help="elements through the height (default: 20)")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/cantilever"),
        help="run directory for artifacts (default: results/cantilever)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="topocombo",
        description="Geometry and meshing stages of the topocombo pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_mesh = sub.add_parser("mesh", help="build the design domain and mesh it")
    _add_model_args(p_mesh)

    p_report = sub.add_parser("report", help="render an HTML report from a run directory")
    p_report.add_argument("--run", type=Path, default=Path("results/cantilever"))
    p_report.add_argument("--site", type=Path, default=Path("site"))

    p_all = sub.add_parser("all", help="mesh, then render the HTML report")
    _add_model_args(p_all)
    p_all.add_argument("--site", type=Path, default=Path("site"))

    args = parser.parse_args(argv)

    if args.command in ("mesh", "all"):
        from .pipeline import run

        domain = BeamDomain(length=args.length, height=args.height, thickness=args.thickness)
        spec = MeshSpec(nelx=args.nelx, nely=args.nely)
        run(domain=domain, spec=spec, out_dir=args.out)

    if args.command in ("report", "all"):
        from .report import build_site

        run_dir = args.run if args.command == "report" else args.out
        index = build_site(run_dir=run_dir, site_dir=args.site)
        print(f"report: {index}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
