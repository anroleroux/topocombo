"""The main flow — geometry, mesh, solve, SIMP — as one terminal-driven run.

The domain decides the dimension: a 2D domain (:class:`BeamDomain`, or a part
script that builds a face) runs the plane-stress quad pipeline, a 3D one
(:class:`BeamDomain3D`, or a script that builds a solid) the solid hex
pipeline.  Where the part is held and loaded comes from its named regions
(:mod:`topocombo.regions`), through the constraints and loads of a study
(:mod:`topocombo.study`); the parametric beam brings its own cantilever
regions and defaults.  Everything is written to a run directory; nothing is
plotted or rendered here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .fea import (
    LoadCase,
    Material,
    element_stiffnesses,
    rigid_body_free,
    solve_cases,
    timoshenko_tip_deflection,
)
from .geometry import BEAM_FIXED, BEAM_LOAD, CadDomain, Domain, export_domain
from .mesh_io import check_mesh, load_mesh, save_mesh, save_solution
from .meshing import (
    MeshSpec,
    MeshSpec3D,
    generate_fitted_mesh,
    generate_tet_mesh,
    generate_hex_mesh,
    generate_quad_mesh,
)
from .optimize import (
    SimpParams,
    element_centroids,
    optimize,
    save_density_field,
    save_design,
    save_history,
)
from .regions import embed_points, node_shares, region_dim, region_nodes
from .runlog import RunLog
from .study import Displace, Fix, Force, LoadCase as StudyCase, Passive
from .topology import save_topology_stl

_REGION_KIND = {0: "vertex", 1: "edge", 2: "face", 3: "solid"}


def _element_size(domain: Domain, spec: MeshSpec | MeshSpec3D) -> list[float]:
    """Element edge lengths in mm: the grid cell (dx, dy[, dz]) when structured;
    the target in-plane edge (and the layer thickness in 3D) when body-fitted."""
    if spec.structured:
        size = [domain.length / spec.nelx, domain.height / spec.nely]
    else:
        size = [spec.element_size(domain)]
    if domain.dim == 3 and not getattr(spec, "tetrahedral", False):
        size.append(domain.width / spec.nelz)
    return size


def _region_summary(shape: Any) -> dict[str, Any]:
    bb = shape.BoundingBox()
    dim = region_dim(shape)
    count = len(shape.Faces() if dim == 2 else shape.Edges() if dim == 1 else shape.Vertices())
    return {
        "kind": _REGION_KIND[dim],
        "count": count,
        "bbox": [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax],
    }


def _boundary_conditions(
    domain: Domain, constraints: Sequence[Fix | Displace], cases: Sequence[StudyCase],
    regions: dict[str, Any],
) -> dict[str, Any]:
    """The displacement constraints and forces the solve applies, by region."""
    axes = "xyz"[: domain.dim]
    return {
        "constraints": [
            {
                **c.as_dict(),
                **_region_summary(regions[c.region]),
                "displacements": {f"u{axes[i]}": v for i, v in c.components(domain.dim).items()},
            }
            for c in constraints
        ],
        "loads": [
            {
                **f.as_dict(),
                **_region_summary(regions[f.region]),
                "case": case.name,
                "weight": case.weight,
                "force": {f"f{a}": v for a, v in zip(axes, f.vector)},
            }
            for case in cases
            for f in case.loads
        ],
    }


def _clamped_nodes(mesh: Any, prescribed: dict[int, float]) -> np.ndarray:
    """Nodes with every component held at zero (drawn as a clamp; the rest of
    the constrained nodes as supports that let something move)."""
    d = mesh.dofs_per_node
    held = np.zeros((mesh.n_nodes, d), dtype=bool)
    for dof, value in prescribed.items():
        if value == 0.0:
            held[dof // d, dof % d] = True
    return np.flatnonzero(held.all(axis=1))


def _held(mesh: Any, prescribed: dict[int, float]) -> np.ndarray:
    """(n_nodes, dim): which displacement components each node has held."""
    d = mesh.dofs_per_node
    held = np.zeros((mesh.n_nodes, d), dtype=bool)
    for dof in prescribed:
        held[dof // d, dof % d] = True
    return held


def run(
    domain: Domain,
    spec: MeshSpec | MeshSpec3D,
    out_dir: Path,
    material: Material | None = None,
    load_fy: float = -1000.0,
    simp: SimpParams | None = None,
    optimize_design: bool = True,
    snapshot_every: int = 10,
    echo: bool = True,
    constraints: Sequence[Fix | Displace] | None = None,
    loads: Sequence[Force] | None = None,
    study: Any = None,
    solver: str = "auto",
    load_cases: Sequence[StudyCase] | None = None,
    passive_regions: Sequence[Passive] = (),
) -> tuple[RunLog, dict[str, Any]]:
    """Mesh the domain, solve it at full density, then run the SIMP loop.

    ``constraints`` (:class:`Fix`, :class:`Displace`), the forces — ``loads``
    for one case or ``load_cases`` for several — and ``passive_regions``
    refer to the domain's named regions.  Left out, the parametric beam uses
    its cantilever defaults: ``fixed`` clamped and ``load_fy`` at ``load``.
    ``study`` is the loaded study the run came from (see
    :func:`topocombo.study.run_study`), recorded with the run.
    """
    three_d = domain.dim == 3
    scripted = isinstance(domain, CadDomain)
    if three_d != isinstance(spec, MeshSpec3D):
        raise TypeError("a 3D domain needs a MeshSpec3D, a 2D domain a MeshSpec")
    regions = domain.regions()
    if constraints is None and loads is None and load_cases is None and not scripted:
        constraints = [Fix(BEAM_FIXED)]
        loads = [Force(BEAM_LOAD, (0.0, load_fy, 0.0)[: domain.dim])]
    if loads and load_cases:
        raise ValueError("give loads (one case) or load_cases, not both")
    cases = list(load_cases) if load_cases else ([StudyCase("load", tuple(loads))] if loads else [])
    passive_regions = list(passive_regions)
    if not constraints or not cases:
        raise ValueError("a run needs constraints and loads on the part's named regions")
    forces = [f for case in cases for f in case.loads]
    bc_names = list(dict.fromkeys([c.region for c in constraints] + [f.region for f in forces]))
    unknown = [n for n in bc_names + [p.region for p in passive_regions] if n not in regions]
    if unknown:
        raise ValueError(
            f"unknown region(s) {', '.join(unknown)}; the part defines "
            f"{', '.join(sorted(regions)) or 'none'}"
        )
    solids = [n for n in bc_names if region_dim(regions[n]) == 3]
    if solids:
        raise ValueError(
            f"region(s) {', '.join(solids)} are solids: constraints and forces act on "
            "vertices, edges or faces (solid regions are for Passive)"
        )
    for f in forces:
        if len(f.vector) != domain.dim:
            raise ValueError(f"a {domain.dim}D part takes a {domain.dim}-component force")
    for c in constraints:
        c.components(domain.dim)  # a z component on a 2D part raises here
    force = forces[0]  # the first force of the first case: what one-load summaries show
    bc_regions = {n: regions[n] for n in bc_names}
    tetrahedral = getattr(spec, "tetrahedral", False)
    if three_d and not tetrahedral and not domain.is_prism:
        domain.profile()  # raises, saying why hexahedra cannot mesh this part
    out_dir = Path(out_dir)
    material = material or Material()
    simp = simp or SimpParams()
    # the out-of-plane size: plane-stress thickness in 2D, the modelled width in 3D
    depth = domain.width if three_d else domain.thickness
    analysis = "3D solid" if three_d else "plane-stress"
    log = RunLog(
        name=f"topology optimization — meshing, {analysis} solve and SIMP",
        out_dir=out_dir,
        echo=echo,
    )
    log.params = {
        "dim": 3 if three_d else 2,
        "element": spec.element.label,
        "element_name": spec.element.name,
        "domain": domain.as_dict(),
        "mesh": {**spec.as_dict(), "element_size": _element_size(domain, spec)},
        "material": material.as_dict(),
        "load": {f"f{a}": v for a, v in zip("xyz", force.vector)},
        "load_cases": [
            {"name": c.name, "weight": c.weight, "forces": [f.as_dict() for f in c.loads]}
            for c in cases
        ],
        "boundary_conditions": _boundary_conditions(domain, constraints, cases, regions),
        "passive": [p.as_dict() for p in passive_regions],
        "regions": {name: _region_summary(shape) for name, shape in regions.items()},
        "simp": simp.as_dict(),
        "solver": solver,
    }
    if study is not None:
        log.params["study"] = {"path": str(study.path), "part": study.study.part}

    import cadquery
    import meshio
    import scipy

    log.tool("cadquery", cadquery.__version__)
    log.tool("meshio", meshio.__version__)
    log.tool("numpy", np.__version__)
    log.tool("scipy", scipy.__version__)

    with log.step("geometry", "1. Define parametric geometry (CadQuery)"):
        if scripted:
            log.log(f"CadQuery input: {domain.path or 'a script'} (any geometry, as code)")
            log.log(
                f"bounding box {domain.length:g} x {domain.height:g}"
                + (f" x {domain.width:g}" if three_d else "")
                + f" mm (aspect ratio {domain.aspect_ratio:.2f})"
                + ("" if three_d else f", out-of-plane thickness {domain.thickness:g} mm")
            )
        if three_d:
            if not scripted:
                log.log(
                    f"design domain {domain.length} x {domain.height} x {domain.width} mm box "
                    f"(aspect ratio {domain.aspect_ratio:.2f})"
                )
            solid = domain.solid()
            log.log(
                f"solid volume: {solid.Volume():.3f} mm^3 (expected {domain.material_volume:.3f})"
            )
            cad_record = {"solid_volume": solid.Volume()}
        else:
            if not scripted:
                log.log(
                    f"design domain {domain.length} x {domain.height} mm "
                    f"(aspect ratio {domain.aspect_ratio:.2f}), "
                    f"out-of-plane thickness {domain.thickness} mm"
                )
            face = domain.face()
            log.log(
                f"planar face area: {face.Area():.3f} mm^2 (expected {domain.material_area:.3f})"
            )
            cad_record = {"face_area": face.Area()}
        for x, y, d in domain.holes:
            log.log(f"cutout through z: diameter {d:g} mm at x = {x:g}, y = {y:g}")
        if scripted and domain.is_prism:
            cut = domain.area - domain.material_area
            log.log(
                f"x-y profile: {domain.material_area:.3f} of {domain.area:.3f} mm^2 envelope"
                + (f" ({cut:.3f} mm^2 cut away)" if domain.has_cutouts else " (fills its bounding box)")
            )
        elif scripted:
            log.log(
                f"not a prism along z: {domain.material_volume:.3f} of {domain.volume:.3f} mm^3 "
                "bounding box — meshed with tetrahedra"
            )
        for name, info in log.params["regions"].items():
            lo, hi = info["bbox"][:3], info["bbox"][3:]
            log.log(
                f"region '{name}': {info['count']} {info['kind']}(s), "
                f"({', '.join(f'{v:g}' for v in lo)}) .. ({', '.join(f'{v:g}' for v in hi)})"
            )
        exported = export_domain(domain, out_dir / "cad")
        log.log(
            "built by running the CadQuery script (shown in the report)" if scripted
            else "built by running the generated CadQuery script (shown in the report)"
        )
        descriptions = {
            "script": "design domain, the CadQuery script that built it (runs in CQ-editor)",
            "brep": "design domain, BREP format",
            "step": "design domain, STEP format",
            "envelope": "design envelope without cutouts, BREP format"
            + (" — what Gmsh meshes" if spec.structured else " (for the structured mesh mode)"),
        }
        for kind, path in exported.items():
            log.artifact(path, descriptions[kind])
        if study is not None:
            study_copy = out_dir / "study.py"
            study_copy.write_text(study.source)
            log.log(f"study: {study.path} (part: {study.study.part})")
            log.artifact(study_copy, "the study script: mesh, material, constraints, loads, optimizer")
        log.record(
            study_script=None if study is None else study.source,
            study_path=None if study is None else str(study.path),
            **cad_record,
            **domain.as_dict(),
            cad_script=domain.cadquery_script(),
        )

    cells_word = spec.element.plural
    kind = "structured" if spec.structured else "body-fitted"
    with log.step("meshing", f"2. Mesh with Gmsh ({kind} {cells_word})"):
        if spec.structured:
            grid = f"{spec.nelx} x {spec.nely}" + (f" x {spec.nelz}" if three_d else "")
            log.log(
                f"transfinite grid: {grid} = {spec.n_elements} {cells_word}, "
                f"{spec.n_nodes} nodes expected"
            )
            if domain.has_cutouts:
                log.log(
                    "the structured grid covers the whole envelope; elements outside the "
                    "part (inside its cutouts) are held void by the optimizer"
                )
            mesher = generate_hex_mesh if three_d else generate_quad_mesh
            brep = exported.get("envelope", exported["brep"])
        elif tetrahedral:
            mesher = generate_tet_mesh
            brep = exported["brep"]
            log.log("body-fitted tetrahedra of the solid itself: any 3D shape")
        else:
            mesher = generate_fitted_mesh
            if three_d:  # the hexes are extruded from a mesh of the x-y profile
                brep = out_dir / "cad" / "design_profile.brep"
                domain.profile().exportBrep(str(brep))
                log.artifact(brep, "x-y profile of the design domain, BREP — what Gmsh meshes")
            else:
                brep = exported["brep"]
            log.log(
                "body-fitted: the mesh follows the CAD boundary, cutouts included"
                if domain.has_cutouts else "body-fitted: the mesh follows the CAD boundary"
            )
        tol = 1e-6 * max(domain.length, domain.height, getattr(domain, "width", 0.0))
        if spec.structured:
            extra = {}
        elif tetrahedral:
            extra = {"regions": bc_regions}
        else:
            extra = {"points": embed_points(bc_regions, domain.profile(), tol)}
        msh_path, _ = mesher(
            domain=domain,
            spec=spec,
            brep_path=brep,
            out_dir=out_dir / "mesh",
            log=log,
            **extra,
        )
        log.artifact(msh_path, f"{spec.element.label} mesh with physical groups (Gmsh 2.2 ASCII)")
        size = None if spec.structured else spec.element_size(domain)
        log.record(**spec.as_dict(), element_size=size)

    with log.step("validation", "3. Load the mesh and check it"):
        mesh = load_mesh(msh_path)
        for name, shape in bc_regions.items():
            mesh.node_sets[name] = region_nodes(mesh.nodes, shape, tol)
        empty = [n for n in bc_names if mesh.node_sets[n].size == 0]
        if empty:
            raise RuntimeError(f"region(s) {', '.join(empty)} have no mesh nodes")
        # every force as nodal loads (fea.LoadCase), grouped by load case
        nodal: list[list[LoadCase]] = []
        for case in cases:
            nodal.append([])
            for f in case.loads:
                nodes_f = mesh.node_sets[f.region]
                shares_f = node_shares(
                    mesh.nodes, mesh.cells, mesh.cell_type, nodes_f,
                    region_dim(regions[f.region]), f.region,
                )
                comps = dict(zip(("fx", "fy", "fz"), f.vector))
                nodal[-1].append(
                    LoadCase(node=tuple(int(n) for n in nodes_f),
                             shares=tuple(float(v) for v in shares_f), **comps)
                    if nodes_f.size > 1 else LoadCase(node=int(nodes_f[0]), **comps)
                )
        load = nodal[0][0]
        load_nodes = load.nodes
        load_node = int(load_nodes[0])
        # constraints: DOF -> held displacement, shared by every case
        prescribed: dict[int, float] = {}
        held_by: dict[int, str] = {}
        for c in constraints:
            for comp, value in c.components(domain.dim).items():
                for n in mesh.node_sets[c.region]:
                    dof = mesh.dofs_per_node * int(n) + comp
                    if dof in prescribed and prescribed[dof] != value:
                        raise ValueError(
                            f"regions '{held_by[dof]}' and '{c.region}' hold node {int(n)}'s "
                            f"u{'xyz'[comp]} at {prescribed[dof]:g} and {value:g}"
                        )
                    prescribed[dof], held_by[dof] = value, c.region
        constrained = np.array(sorted(prescribed), dtype=int)
        loose = rigid_body_free(mesh, constrained)
        if loose:
            raise ValueError(
                f"the constraints leave the part free to move: {loose} rigid-body motion(s) "
                "(translations or rotations) are not held; fix more components"
            )
        fixed_nodes = np.unique(constrained // mesh.dofs_per_node)
        summary = check_mesh(mesh, domain, expected_elements=spec.n_elements)
        summary["mesh_mode"] = spec.mode
        summary["load_node"] = load_node
        summary["load_nodes"] = [int(n) for n in load_nodes]
        summary["load_node_coords"] = [float(c) for c in mesh.nodes[load_node]]
        if spec.structured:
            cutouts = domain.void_mask(element_centroids(mesh))
        else:  # the cutouts are not meshed at all
            cutouts = np.zeros(mesh.n_elements, dtype=bool)
        if cutouts.any():
            measures = mesh.cell_measures()
            if three_d:
                cut = domain.volume - domain.material_volume
            else:
                cut = domain.area - domain.material_area
            summary["passive_elements"] = int(cutouts.sum())
            summary["passive_measure"] = float(measures[cutouts].sum())
            summary["cutout_measure"] = float(cut)
        # passive regions: held void (with the cutouts) or held solid
        passive, solid = cutouts.copy(), np.zeros(mesh.n_elements, dtype=bool)
        centroids = element_centroids(mesh)
        for p_region in passive_regions:
            picked = region_nodes(centroids, regions[p_region.region], max(p_region.within, tol))
            if picked.size == 0:
                raise RuntimeError(f"Passive('{p_region.region}') holds no elements: widen `within`")
            (solid if p_region.state == "solid" else passive)[picked] = True
            summary.setdefault("passive_regions", []).append(
                {**p_region.as_dict(), "elements": int(picked.size)}
            )
        if np.any(solid & passive):
            raise ValueError("some elements are in both a solid and a void passive region")
        summary["passive_void_elements"] = int(passive.sum())
        summary["passive_solid_elements"] = int(solid.sum())
        summary["constrained_dofs"] = int(constrained.size)
        summary["load_cases"] = [
            {"name": case.name, "weight": case.weight,
             "forces": [{"region": f.region, "nodes": int(mesh.node_sets[f.region].size),
                         "vector": list(f.vector)} for f in case.loads]}
            for case in cases
        ]

        log.log(f"{summary['n_nodes']} nodes, {summary['n_elements']} {cells_word}, {summary['n_dofs']} DOFs")
        log.log(
            "element edge lengths: "
            f"{summary['edge_length_min']:.4f} – {summary['edge_length_max']:.4f} mm, "
            f"max aspect ratio {summary['aspect_ratio_max']:.4f}"
        )
        m, unit = mesh.measure_name, "mm^3" if three_d else "mm^2"
        log.log(
            f"meshed {m} {summary[f'{m}_sum']:.6f} {unit} vs "
            f"{'domain' if spec.structured else 'CAD (cutouts removed)'} "
            f"{summary[f'domain_{m}']:.6f} {unit}"
        )
        for name, count in summary["node_sets"].items():
            log.log(f"node set '{name}': {count} nodes")
        if cutouts.any():
            log.log(
                f"cutouts: {summary['passive_elements']} elements held void, "
                f"{summary['passive_measure']:.4g} {unit} of grid vs "
                f"{summary['cutout_measure']:.4g} {unit} in the CAD model "
                "(the grid resolves a cutout to whole elements)"
            )
        for p_info in summary.get("passive_regions", []):
            log.log(
                f"passive '{p_info['region']}' held {p_info['state']}: {p_info['elements']} "
                f"elements within {p_info['within']:g} mm"
            )
        for case in summary["load_cases"]:
            for f_info in case["forces"]:
                log.log(
                    f"load case '{case['name']}': force on '{f_info['region']}' "
                    f"spread over {f_info['nodes']} node(s)"
                )
        for name, passed in summary["checks"].items():
            log.log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not summary["all_checks_passed"]:
            failed = [n for n, ok in summary["checks"].items() if not ok]
            hint = ""
            if any(n.startswith("tet_quality") for n in failed):
                hint = (
                    f" (worst tetrahedron quality {summary['tet_quality_min']:.3g}: elements"
                    " larger than the part's thinnest feature make slivers — try a smaller"
                    " Mesh.size)"
                )
            raise RuntimeError(f"mesh validation failed: {', '.join(failed)}{hint}")
        log.record(**summary)

    with log.step("export", "4. Write mesh artifacts"):
        paths = save_mesh(
            mesh, out_dir / "mesh", load_node=load_node, load_nodes=load_nodes,
            passive=passive if passive.any() else None,
            fixed_nodes=fixed_nodes,
            load_vector=force.vector,
            solid=solid if solid.any() else None,
            clamped_nodes=_clamped_nodes(mesh, prescribed),
            held=_held(mesh, prescribed)[fixed_nodes],
        )
        log.artifact(paths["npz"], f"nodes, {mesh.cell_type} connectivity and boundary node sets (numpy)")
        log.artifact(paths["vtu"], "mesh for PyVista / ParaView")

    with log.step("solve", f"5. Assemble and solve ({analysis}, full density)"):
        log.log(
            f"material: E = {material.youngs_modulus:g} MPa, nu = {material.poisson_ratio:g}, "
            f"G = {material.shear_modulus:g} MPa"
        )
        for c in constraints:
            comps = c.components(domain.dim)
            what = ", ".join(f"u{'xyz'[i]} = {v:g}" for i, v in comps.items())
            log.log(
                f"constraint on '{c.region}' ({mesh.node_sets[c.region].size} nodes): {what}"
            )
        log.log(f"{constrained.size} constrained DOFs; every rigid-body motion held")
        for case, loads_c in zip(cases, nodal):
            for f, lc in zip(case.loads, loads_c):
                shares_text = (
                    f" (shares {', '.join(f'{v:.3g}' for v in lc.node_shares())})"
                    if 1 < lc.nodes.size <= 6 else ""
                )
                log.log(
                    f"case '{case.name}' (weight {case.weight:g}): F = "
                    f"({', '.join(f'{v:g}' for v in f.vector)}) N on '{f.region}', spread over "
                    f"{lc.nodes.size} node(s) by {_REGION_KIND[region_dim(regions[f.region])]}"
                    f" tributary share{shares_text}"
                )
        ke_all = element_stiffnesses(mesh, material, depth)
        n_edof = ke_all.shape[1]
        log.log(
            f"assembled {ke_all.shape[0]} element stiffness matrices "
            f"({n_edof}x{n_edof}, {len(mesh.element.quadrature[1])}-point quadrature)"
        )
        density0 = np.where(passive, 0.0, 1.0) if passive.any() else None
        results = solve_cases(
            mesh=mesh,
            material=material,
            thickness=depth,
            loads=nodal,
            prescribed=prescribed,
            densities=density0,
            ke_all=ke_all,
            solver=solver,
        )
        result = results[0]
        weights = np.array([case.weight for case in cases])
        total = float(weights @ [r.compliance for r in results])
        tip_uy = float(result.component(1)[load.nodes].mean())
        reactions = [float(result.component(i, "reactions").sum()) for i in range(mesh.dim)]
        reaction_y = reactions[1]
        # a reference only the parametric cantilever has: tip-loaded beam theory
        beam = None if scripted or len(forces) != 1 else timoshenko_tip_deflection(
            domain.length, domain.height, depth, material, load.fy
        )
        rel = abs(abs(tip_uy) - beam["total"]) / beam["total"] if beam else None

        how = (
            f"multigrid-preconditioned CG, {result.solver_iterations} iterations"
            if result.solver == "amg-cg" else f"sparse {result.solver}"
        )
        log.log(f"solved {result.n_free_dofs} free DOFs ({how})"
                + (f", {len(cases)} load cases" if len(cases) > 1 else ""))
        case_summaries = []
        for case, r, loads_c in zip(cases, results, nodal):
            applied = sum(np.array(lc.components(mesh.dim)) for lc in loads_c)
            react = [float(r.component(i, "reactions").sum()) for i in range(mesh.dim)]
            log.log(
                f"case '{case.name}': compliance = {r.compliance:.6g} N*mm; reactions "
                f"({', '.join(f'{v:.6g}' for v in react)}) N vs applied "
                f"({', '.join(f'{0.0 - v:g}' for v in applied)}) N; "
                f"||KU - F|| on free DOFs = {r.equilibrium_residual:.3e}"
            )
            case_summaries.append({
                "name": case.name, "weight": case.weight, "compliance": r.compliance,
                "reactions": react, "applied": [float(v) for v in applied],
                "equilibrium_residual": r.equilibrium_residual,
                "max_displacement": float(r.displacement_magnitude.max()),
            })
        if len(cases) > 1:
            log.log(f"weighted compliance = {total:.6g} N*mm")
        log.log(f"compliance F.U = {total:.6g} N*mm"
                + (" (f.u - u_p.r_p: prescribed displacements do work)" if any(prescribed.values()) else ""))
        log.log(
            f"uy at the load = {tip_uy:.6g} mm"
            + (f" (mean over {load.nodes.size} nodes)" if load.nodes.size > 1 else "")
        )
        if beam:
            log.log(
                f"Timoshenko beam theory: {beam['total']:.6g} mm "
                f"(bending {beam['bending']:.4g} + shear {beam['shear']:.4g}) "
                f"-> {rel * 100:.2f}% difference"
                + (" (beam theory ignores the cutouts)" if domain.has_cutouts else "")
            )
        log.log(
            f"equilibrium: ||KU - F|| on free DOFs = {result.equilibrium_residual:.3e}, "
            f"sum of reactions = ({', '.join(f'{v:.6g}' for v in reactions)}) N"
        )
        log.log(
            f"von Mises at centroids: {result.von_mises.min():.4g} – "
            f"{result.von_mises.max():.4g} MPa"
            + (f" (case '{cases[0].name}')" if len(cases) > 1 else "")
        )
        log.log(
            "element compliances u_e^T k_e u_e sum to "
            f"{result.element_compliance.sum():.6g} N*mm — the SIMP sensitivity basis"
        )
        solve_summary = {
            "compliance": total,
            "cases": case_summaries,
            "tip_uy": tip_uy,
            "max_deflection": result.max_deflection(),
            "beam_theory": beam,
            "beam_theory_rel_diff": rel,
            "equilibrium_residual": result.equilibrium_residual,
            "reaction_y": reaction_y,
            "reactions": reactions,
            "n_free_dofs": result.n_free_dofs,
            "solver": result.solver,
            "solver_iterations": result.solver_iterations,
            "von_mises_min": float(result.von_mises.min()),
            "von_mises_max": float(result.von_mises.max()),
            "element_compliance_sum": float(result.element_compliance.sum()),
            "load": load.as_dict(),
            "material": material.as_dict(),
        }
        log.record(**solve_summary)
        summary["solve"] = solve_summary

    with log.step("solution_export", "6. Write solution artifacts"):
        sol_paths = save_solution(mesh, result, out_dir / "solution")
        log.artifact(sol_paths["npz"], "displacements, element compliances, von Mises (numpy)")
        log.artifact(sol_paths["vtu"], "displacement and stress fields for PyVista / ParaView")
        log.log("the SIMP loop reuses this solve per iteration; nothing here plots.")

    if optimize_design:
        with log.step("optimize", "7. SIMP compliance minimisation"):
            log.log(
                f"volume fraction {simp.volume_fraction:g}, penalty p = {simp.penal:g}, "
                f"{simp.filter_type} filter with radius {simp.filter_radius:g} mm, "
                f"move limit {simp.move_limit:g}"
            )
            log.log(
                f"stopping at max density change < {simp.tolerance:g} "
                f"or {simp.max_iterations} iterations"
            )
            snapshots_dir = out_dir / "optimization" / "snapshots"

            def on_iteration(record: dict[str, float], densities) -> None:
                it = int(record["iteration"])
                if it <= 3 or it % 5 == 0:
                    log.log(
                        f"  it {it:3d}  c = {record['compliance']:10.4f}  "
                        f"vol = {record['volume_fraction']:.4f}  "
                        f"change = {record['change']:.4f}  "
                        f"Mnd = {record['measure_of_discreteness']:5.1f}%"
                    )
                if snapshot_every and it % snapshot_every == 0:
                    save_density_field(
                        mesh, densities, snapshots_dir / f"density_iter_{it:04d}.vtu"
                    )

            if solid.any():
                log.log(f"{int(solid.sum())} elements held solid, {int(passive.sum())} held void")
            if len(cases) > 1:
                log.log(
                    "objective: the weighted sum of the load cases' compliances ("
                    + " + ".join(f"{c.weight:g} x {c.name}" for c in cases) + ")"
                )
            design = optimize(
                mesh=mesh,
                material=material,
                thickness=depth,
                load=None,
                fixed_node_set=None,
                params=simp,
                on_iteration=on_iteration,
                passive=passive,
                solver=solver,
                cases=[(case.weight, loads_c) for case, loads_c in zip(cases, nodal)],
                prescribed=prescribed,
                solid=solid,
            )
            reduction = (design.compliance / total) if total else float("nan")
            if len(cases) > 1:
                for case, c_value in zip(cases, design.case_compliances):
                    log.log(f"case '{case.name}': compliance {c_value:.6g} N*mm")
            log.log(
                f"{'converged' if design.converged else 'stopped'} after "
                f"{design.iterations} iterations"
            )
            log.log(
                f"compliance {design.compliance:.4f} N*mm at {design.volume_fraction:.3f} "
                f"volume fraction ({reduction:.2f}x the full-density compliance, "
                f"with {simp.volume_fraction:g} of the material)"
            )
            log.log(
                f"measure of discreteness Mnd = {design.measure_of_discreteness():.1f}% "
                f"({np.mean(design.densities > 0.9) * 100:.0f}% of elements solid, "
                f"{np.mean(design.densities < 0.1) * 100:.0f}% void)"
            )
            opt_summary = design.as_dict()
            opt_summary["compliance_ratio_to_full_density"] = reduction
            log.record(**opt_summary)
            summary["optimization"] = opt_summary

        with log.step("design_export", "8. Write the optimized design"):
            paths = save_design(mesh, design, out_dir / "optimization")
            log.artifact(paths["npz"], "optimized density field (numpy)")
            log.artifact(paths["vtu"], "density field for PyVista / ParaView")
            csv_path = save_history(design.history, out_dir / "optimization" / "log.csv")
            log.artifact(csv_path, "per-iteration scalar log (compliance, volume, change)")

            stl = save_topology_stl(
                mesh, design.densities, out_dir / "optimization" / "topology.stl",
                thickness=depth,
            )
            log.log(
                f"topology: {stl['solid_elements']} of {mesh.n_elements} elements at "
                f"rho >= {stl['threshold']:g} -> {stl['triangles']} triangles enclosing "
                f"{stl['volume']:.4g} mm^3"
                + ("" if three_d else f" (extruded by the {depth:g} mm thickness)")
            )
            log.artifact(stl["path"], "thresholded topology as a closed surface, for Blender (STL)")
            log.record(topology={k: v for k, v in stl.items() if k != "path"})

    json_path, text_path = log.write()
    print(f"\nrun log: {json_path}")
    print(f"terminal log: {text_path}")
    return log, summary
