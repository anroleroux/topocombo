"""Terminal entry point: ``python -m topocombo.cli ...``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .geometry import BeamDomain, BeamDomain3D, Domain
from .meshing import MESH_MODES, MeshSpec, MeshSpec3D

#: Inputs of the parametric beam and their defaults.  The flags default to None
#: so a flag that --study makes meaningless can be rejected.
CAD_DEFAULTS = {
    "dim": 2, "length": 60.0, "height": 20.0, "thickness": 1.0, "width": 1.0, "holes": (),
}


def _nav(text: str) -> tuple[str, str]:
    """``LABEL=URL`` -> (label, url)."""
    label, sep, url = text.partition("=")
    if not sep or not label or not url:
        raise argparse.ArgumentTypeError(f"expected LABEL=URL, not {text!r}")
    return label, url


def _hole(text: str) -> tuple[float, float, float]:
    """``X,Y,D`` -> (x, y, diameter)."""
    try:
        x, y, d = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y,DIAMETER in mm, not {text!r}") from None
    return x, y, d


#: Flags a study sets itself; given together with --study they are an error.
#: (--out, --site and --no-optimize still apply.)
_STUDY_SETS = (
    "dim", "length", "height", "thickness", "width", "holes", "nelx", "nely", "nelz",
    "mesh_mode", "mesh_size", "youngs", "poisson", "load", "volfrac", "penal", "rmin",
    "filter_type", "max_iter", "tol", "element", "solver", "optimizer",
)


def _add_cad_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--study",
        type=Path,
        default=None,
        help="a study script (study.py): it names the CadQuery part script and sets the"
        " mesh, material, constraints, loads and optimizer. Without it, the flags below"
        " describe a parametric cantilever",
    )
    p.add_argument(
        "--dim",
        type=int,
        choices=(2, 3),
        default=None,
        help="2: plane-stress quads; 3: solid hexahedra (default: 2)",
    )
    p.add_argument("--length", type=float, default=None, help="beam length in mm (default: 60)")
    p.add_argument("--height", type=float, default=None, help="beam height in mm (default: 20)")
    p.add_argument(
        "--thickness",
        type=float,
        default=None,
        help="2D only: out-of-plane thickness in mm (default: 1)",
    )
    p.add_argument(
        "--width", type=float, default=None, help="3D only: beam width along z in mm (default: 1)"
    )
    p.add_argument(
        "--hole",
        dest="holes",
        type=_hole,
        action="append",
        default=None,
        metavar="X,Y,D",
        help="circular cutout through z, diameter D centred at (X, Y) in mm; repeatable"
        " (default: none)",
    )


def _resolve_cad_args(
    parser: argparse.ArgumentParser, sub: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Fill the parametric beam inputs in ``args`` with their defaults; with
    --study the study sets them, so giving them too is rejected."""
    if args.study is not None:
        given = [
            k for k in _STUDY_SETS
            if hasattr(args, k) and getattr(args, k) != sub.get_default(k)
        ]
        if given:
            flags = ", ".join(
                {"holes": "--hole", "mesh_mode": "--mesh", "filter_type": "--filter"}.get(
                    k, "--" + k.replace("_", "-")
                )
                for k in given
            )
            parser.error(f"--study sets these itself; drop {flags}")
        return
    for key, default in CAD_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, default)


def _domain(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Domain:
    try:
        if args.dim == 3:
            return BeamDomain3D(
                length=args.length, height=args.height, width=args.width, holes=args.holes
            )
        return BeamDomain(
            length=args.length, height=args.height, thickness=args.thickness, holes=args.holes
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def _add_model_args(p: argparse.ArgumentParser) -> None:
    _add_cad_args(p)
    p.add_argument("--nelx", type=int, default=60, help="elements along the length (default: 60)")
    p.add_argument("--nely", type=int, default=20, help="elements through the height (default: 20)")
    p.add_argument(
        "--nelz",
        type=int,
        default=1,
        help="3D only: elements through the width (default: 1, keeps runs light)",
    )
    p.add_argument(
        "--optimizer",
        choices=("auto", "oc", "mma", "nlopt"),
        default="auto",
        help="Optimality Criteria, MMA or NLopt's MMA (both with the density filter)"
        " (default: auto, which is OC for the parametric beam's minimum compliance)",
    )
    p.add_argument(
        "--solver",
        choices=("auto", "direct", "cg"),
        default="auto",
        help="linear solver: direct sparse LU, multigrid-preconditioned CG, or auto"
        " (direct up to 30k free DOFs) (default: auto)",
    )
    p.add_argument(
        "--element",
        choices=("hex8", "tet4", "tet10"),
        default="hex8",
        help="3D only: hex8 (structured or extruded hexahedra) or tetrahedra meshed from"
        " the solid, body-fitted only: tet10 (quadratic) or tet4 (linear, stiff in"
        " bending) (default: hex8)",
    )
    p.add_argument(
        "--mesh",
        dest="mesh_mode",
        choices=MESH_MODES,
        default="structured",
        help="structured: transfinite grid over the envelope, cutouts held void;"
        " body-fitted: unstructured quads (extruded hexes in 3D) of the real CAD"
        " profile, cutouts included (default: structured)",
    )
    p.add_argument(
        "--mesh-size",
        type=float,
        default=None,
        help="body-fitted only: target element edge in mm"
        " (default: the structured cell size, min(L/nelx, H/nely))",
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

    p_cad = sub.add_parser(
        "cad",
        help="build only the CadQuery design domain: script, BREP, STEP and a PNG preview",
    )
    _add_cad_args(p_cad)
    p_cad.add_argument(
        "--out",
        type=Path,
        default=Path("results/cad"),
        help="directory for the CAD files (default: results/cad)",
    )

    p_run = sub.add_parser("run", help="mesh, solve and optimize the design domain")
    _add_model_args(p_run)

    p_report = sub.add_parser("report", help="render an HTML report from a run directory")
    p_report.add_argument("--run", type=Path, default=Path("results/cantilever"))
    p_report.add_argument("--site", type=Path, default=Path("site"))
    p_report.add_argument("--nav", type=_nav, action="append", default=None, metavar="LABEL=URL",
                          help="link to another report from this one; repeatable")

    p_all = sub.add_parser("all", help="run the pipeline, then render the HTML report")
    _add_model_args(p_all)
    p_all.add_argument("--site", type=Path, default=Path("site"))
    p_all.add_argument("--nav", type=_nav, action="append", default=None, metavar="LABEL=URL",
                       help="link to another report from this one; repeatable")

    args = parser.parse_args(argv)
    subparsers = {"cad": p_cad, "run": p_run, "all": p_all}
    if args.command in subparsers:
        _resolve_cad_args(parser, subparsers[args.command], args)

    loaded = None
    if getattr(args, "study", None) is not None:
        from .study import load_study

        try:
            loaded = load_study(args.study)
        except (OSError, ValueError) as exc:
            parser.error(f"--study: {exc}")

    if args.command == "cad":
        from .cadview import render_brep
        from .geometry import export_domain

        domain = loaded.domain if loaded else _domain(parser, args)
        print(domain.cadquery_script())
        exported = export_domain(domain, args.out)
        exported["png"] = render_brep(exported["brep"], args.out / "design_domain.png")
        for kind, path in exported.items():
            print(f"{kind}: {path}")

    if args.command in ("run", "all") and loaded is not None:
        from .study import run_loaded

        run_loaded(loaded, args.out, optimize=not args.no_optimize)
    elif args.command in ("run", "all"):
        from .fea import Material
        from .optimize import SimpParams
        from .pipeline import run

        domain = _domain(parser, args)
        mesh_opts = {"mode": args.mesh_mode, "size": args.mesh_size}
        try:
            if args.dim == 3:
                from .elements import BY_NAME

                spec = MeshSpec3D(
                    nelx=args.nelx, nely=args.nely, nelz=args.nelz,
                    element=BY_NAME[args.element], **mesh_opts,
                )
            else:
                if args.element != "hex8":
                    parser.error("--element picks a 3D element; a 2D run uses quad4")
                spec = MeshSpec(nelx=args.nelx, nely=args.nely, **mesh_opts)
        except ValueError as exc:
            parser.error(str(exc))
        material = Material(youngs_modulus=args.youngs, poisson_ratio=args.poisson)
        simp = SimpParams(
            volume_fraction=args.volfrac,
            penal=args.penal,
            filter_radius=args.rmin,
            filter_type=args.filter_type,
            max_iterations=args.max_iter,
            tolerance=args.tol,
            optimizer=args.optimizer,
        )
        run(
            domain=domain,
            spec=spec,
            out_dir=args.out,
            material=material,
            load_fy=args.load,
            simp=simp,
            optimize_design=not args.no_optimize,
            solver=args.solver,
        )

    if args.command in ("report", "all"):
        from .report import build_site

        run_dir = args.run if args.command == "report" else args.out
        index = build_site(run_dir=run_dir, site_dir=args.site, nav=args.nav)
        print(f"report: {index}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
