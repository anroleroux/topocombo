"""Terminal entry point: ``python -m topocombo.cli ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .geometry import BeamDomain, BeamDomain3D
from .meshing import MeshSpec, MeshSpec3D


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dim",
        type=int,
        choices=(2, 3),
        default=2,
        help="2: plane-stress quads; 3: solid hexahedra (default: 2)",
    )
    p.add_argument("--length", type=float, default=60.0, help="beam length in mm (default: 60)")
    p.add_argument("--height", type=float, default=20.0, help="beam height in mm (default: 20)")
    p.add_argument(
        "--thickness",
        type=float,
        default=1.0,
        help="2D only: out-of-plane thickness in mm (default: 1)",
    )
    p.add_argument(
        "--width", type=float, default=1.0, help="3D only: beam width along z in mm (default: 1)"
    )
    p.add_argument("--nelx", type=int, default=60, help="elements along the length (default: 60)")
    p.add_argument("--nely", type=int, default=20, help="elements through the height (default: 20)")
    p.add_argument(
        "--nelz",
        type=int,
        default=1,
        help="3D only: elements through the width (default: 1, keeps runs light)",
    )
    p.add_argument(
        "--youngs", type=float, default=210_000.0, help="Young's modulus in MPa (default: 210000)"
    )
    p.add_argument("--poisson", type=float, default=0.3, help="Poisson's ratio (default: 0.3)")
    p.add_argument(
        "--load", type=float, default=-1000.0, help="vertical tip load in N (default: -1000)"
    )
    p.add_argument(
        "--volfrac", type=float, default=0.5, help="target volume fraction (default: 0.5)"
    )
    p.add_argument("--penal", type=float, default=3.0, help="SIMP penalty exponent (default: 3)")
    p.add_argument(
        "--rmin", type=float, default=1.5, help="filter radius in mm (default: 1.5)"
    )
    p.add_argument(
        "--filter",
        dest="filter_type",
        choices=("sensitivity", "density"),
        default="sensitivity",
        help="filtering scheme (default: sensitivity)",
    )
    p.add_argument(
        "--max-iter", type=int, default=80, help="iteration cap for the SIMP loop (default: 80)"
    )
    p.add_argument(
        "--tol", type=float, default=0.01, help="convergence tolerance on density change"
    )
    p.add_argument(
        "--no-optimize", action="store_true", help="stop after the full-density solve"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/cantilever"),
        help="run directory for artifacts (default: results/cantilever)",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="topocombo",
        description="The topocombo pipeline: geometry, meshing, FEA solve (2D or 3D), SIMP loop.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="mesh, solve and optimize the design domain")
    _add_model_args(p_run)

    p_report = sub.add_parser("report", help="render an HTML report from a run directory")
    p_report.add_argument("--run", type=Path, default=Path("results/cantilever"))
    p_report.add_argument("--site", type=Path, default=Path("site"))

    p_all = sub.add_parser("all", help="run the pipeline, then render the HTML report")
    _add_model_args(p_all)
    p_all.add_argument("--site", type=Path, default=Path("site"))

    args = parser.parse_args(argv)

    if args.command in ("run", "all"):
        from .fea import Material
        from .optimize import SimpParams
        from .pipeline import run

        if args.dim == 3:
            domain = BeamDomain3D(length=args.length, height=args.height, width=args.width)
            spec = MeshSpec3D(nelx=args.nelx, nely=args.nely, nelz=args.nelz)
        else:
            domain = BeamDomain(length=args.length, height=args.height, thickness=args.thickness)
            spec = MeshSpec(nelx=args.nelx, nely=args.nely)
        material = Material(youngs_modulus=args.youngs, poisson_ratio=args.poisson)
        simp = SimpParams(
            volume_fraction=args.volfrac,
            penal=args.penal,
            filter_radius=args.rmin,
            filter_type=args.filter_type,
            max_iterations=args.max_iter,
            tolerance=args.tol,
        )
        run(
            domain=domain,
            spec=spec,
            out_dir=args.out,
            material=material,
            load_fy=args.load,
            simp=simp,
            optimize_design=not args.no_optimize,
        )

    if args.command in ("report", "all"):
        from .report import build_site

        run_dir = args.run if args.command == "report" else args.out
        index = build_site(run_dir=run_dir, site_dir=args.site)
        print(f"report: {index}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
