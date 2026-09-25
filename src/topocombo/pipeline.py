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
    solve as fea_solve,
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
from .study import Fix, Force
from .topology import save_topology_stl

_REGION_KIND = {0: "vertex", 1: "edge", 2: "face"}


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
    domain: Domain, constraints: Sequence[Fix], loads: Sequence[Force], regions: dict[str, Any]
) -> dict[str, Any]:
    """The displacement constraints and forces the solve applies, by region."""
    axes = "xyz"[: domain.dim]
    return {
        "constraints": [
            {
                **c.as_dict(),
                **_region_summary(regions[c.region]),
                "displacements": {f"u{a}": 0.0 for a in axes},
            }
            for c in constraints
        ],
        "loads": [
            {
                **f.as_dict(),
                **_region_summary(regions[f.region]),
                "force": {f"f{a}": v for a, v in zip(axes, f.vector)},
            }
            for f in loads
        ],
    }


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
    constraints: Sequence[Fix] | None = None,
    loads: Sequence[Force] | None = None,
    study: Any = None,
    solver: str = "auto",
) -> tuple[RunLog, dict[str, Any]]:
    """Mesh the domain, solve it at full density, then run the SIMP loop.

    ``constraints`` and ``loads`` refer to the domain's named regions.  Left
    out, the parametric beam uses its cantilever defaults: ``fixed`` clamped
    and ``load_fy`` at ``load``.  ``study`` is the loaded study the run came
    from (see :func:`topocombo.study.run_study`), recorded with the run.
    """
    three_d = domain.dim == 3
    scripted = isinstance(domain, CadDomain)
    if three_d != isinstance(spec, MeshSpec3D):
        raise TypeError("a 3D domain needs a MeshSpec3D, a 2D domain a MeshSpec")
    regions = domain.regions()
    if constraints is None and loads is None and not scripted:
        constraints = [Fix(BEAM_FIXED)]
        loads = [Force(BEAM_LOAD, (0.0, load_fy, 0.0)[: domain.dim])]
    if not constraints or not loads:
        raise ValueError("a run needs constraints and loads on the part's named regions")
    unknown = [n for n in [c.region for c in constraints] + [f.region for f in loads]
               if n not in regions]
    if unknown:
        raise ValueError(
            f"unknown region(s) {', '.join(unknown)}; the part defines "
            f"{', '.join(sorted(regions)) or 'none'}"
        )
    if len(loads) != 1:
        raise ValueError("exactly one Force for now")
    force = loads[0]
    if len(force.vector) != domain.dim:
        raise ValueError(f"a {domain.dim}D part takes a {domain.dim}-component force")
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
        "boundary_conditions": _boundary_conditions(domain, constraints, loads, regions),
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
            extra = {"regions": regions}
        else:
            extra = {"points": embed_points(regions, domain.profile(), tol)}
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
        for name in dict.fromkeys([c.region for c in constraints] + [force.region]):
            mesh.node_sets[name] = region_nodes(mesh.nodes, regions[name], tol)
        load_nodes = mesh.node_sets[force.region]
        if load_nodes.size == 0:
            raise RuntimeError(f"region '{force.region}' has no mesh nodes to carry the load")
        shares = node_shares(
            mesh.nodes, mesh.cells, mesh.cell_type, load_nodes,
            region_dim(regions[force.region]), force.region,
        )
        load_node = int(load_nodes[0])
        summary = check_mesh(mesh, domain, expected_elements=spec.n_elements)
        summary["mesh_mode"] = spec.mode
        summary["load_node"] = load_node
        summary["load_nodes"] = [int(n) for n in load_nodes]
        summary["load_node_coords"] = [float(c) for c in mesh.nodes[load_node]]
        if spec.structured:
            passive = domain.void_mask(element_centroids(mesh))
        else:  # the cutouts are not meshed at all
            passive = np.zeros(mesh.n_elements, dtype=bool)
        if passive.any():
            measures = mesh.cell_measures()
            if three_d:
                cut = domain.volume - domain.material_volume
            else:
                cut = domain.area - domain.material_area
            summary["passive_elements"] = int(passive.sum())
            summary["passive_measure"] = float(measures[passive].sum())
            summary["cutout_measure"] = float(cut)

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
        if passive.any():
            log.log(
                f"cutouts: {summary['passive_elements']} elements held void, "
                f"{summary['passive_measure']:.4g} {unit} of grid vs "
                f"{summary['cutout_measure']:.4g} {unit} in the CAD model "
                "(the grid resolves a cutout to whole elements)"
            )
        coords = ", ".join(f"{c:.3f}" for c in summary["load_node_coords"])
        log.log(
            f"load region '{force.region}': {load_nodes.size} node(s), "
            f"first #{load_node} at ({coords})"
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
            fixed_nodes=np.unique(np.concatenate(
                [mesh.node_sets[c.region] for c in constraints]
            )),
            load_vector=force.vector,
        )
        log.artifact(paths["npz"], f"nodes, {mesh.cell_type} connectivity and boundary node sets (numpy)")
        log.artifact(paths["vtu"], "mesh for PyVista / ParaView")

    with log.step("solve", f"5. Assemble and solve ({analysis}, full density)"):
        components = dict(zip(("fx", "fy", "fz"), force.vector))
        if load_nodes.size > 1:
            load = LoadCase(
                node=tuple(int(n) for n in load_nodes), shares=tuple(float(v) for v in shares),
                **components,
            )
        else:
            load = LoadCase(node=load_node, **components)
        fixed_sets = [c.region for c in constraints]
        n_fixed = int(np.unique(np.concatenate([mesh.node_sets[n] for n in fixed_sets])).size)
        log.log(
            f"material: E = {material.youngs_modulus:g} MPa, nu = {material.poisson_ratio:g}, "
            f"G = {material.shear_modulus:g} MPa"
        )
        vector = ", ".join(f"{v:g}" for v in force.vector)
        log.log(
            f"constraints: {', '.join(repr(n) for n in fixed_sets)} clamped "
            f"({n_fixed} nodes, {mesh.dofs_per_node * n_fixed} DOFs)"
        )
        shares_text = (
            f" (shares {', '.join(f'{v:.3g}' for v in load.node_shares())})"
            if 1 < load.nodes.size <= 6 else ""
        )
        log.log(
            f"load: F = ({vector}) N on '{force.region}', spread over "
            f"{load.nodes.size} node(s) by {_REGION_KIND[region_dim(regions[force.region])]}"
            f" tributary share{shares_text}"
        )
        ke_all = element_stiffnesses(mesh, material, depth)
        n_edof = ke_all.shape[1]
        gauss = "x".join(["2"] * mesh.dim)
        log.log(
            f"assembled {ke_all.shape[0]} element stiffness matrices "
            f"({n_edof}x{n_edof}, {gauss} Gauss)"
        )
        result = fea_solve(
            mesh=mesh,
            material=material,
            thickness=depth,
            load=load,
            fixed_node_set=fixed_sets,
            densities=np.where(passive, 0.0, 1.0) if passive.any() else None,
            ke_all=ke_all,
            solver=solver,
        )
        tip_uy = float(result.component(1)[load.nodes].mean())
        reactions = [float(result.component(i, "reactions").sum()) for i in range(mesh.dim)]
        reaction_y = reactions[1]
        # a reference only the parametric cantilever has: tip-loaded beam theory
        beam = None if scripted else timoshenko_tip_deflection(
            domain.length, domain.height, depth, material, load.fy
        )
        rel = abs(abs(tip_uy) - beam["total"]) / beam["total"] if beam else None

        how = (
            f"multigrid-preconditioned CG, {result.solver_iterations} iterations"
            if result.solver == "amg-cg" else f"sparse {result.solver}"
        )
        log.log(f"solved {result.n_free_dofs} free DOFs ({how})")
        log.log(f"compliance F.U = {result.compliance:.6g} N*mm")
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
            f"sum of reactions = ({', '.join(f'{v:.6g}' for v in reactions)}) N vs applied "
            f"({', '.join(f'{0.0 - v:g}' for v in force.vector)}) N"
        )
        log.log(
            f"von Mises at centroids: {result.von_mises.min():.4g} – "
            f"{result.von_mises.max():.4g} MPa"
        )
        log.log(
            "element compliances u_e^T k_e u_e sum to "
            f"{result.element_compliance.sum():.6g} N*mm — the SIMP sensitivity basis"
        )
        solve_summary = {
            "compliance": result.compliance,
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

            design = optimize(
                mesh=mesh,
                material=material,
                thickness=depth,
                load=load,
                fixed_node_set=fixed_sets,
                params=simp,
                on_iteration=on_iteration,
                passive=passive,
                solver=solver,
            )
            reduction = (design.compliance / result.compliance) if result.compliance else float("nan")
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
