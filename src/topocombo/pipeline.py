"""The main flow — geometry, mesh, solve, SIMP — as one terminal-driven run.

The domain decides the dimension: a :class:`BeamDomain` runs the 2D
plane-stress quad pipeline, a :class:`BeamDomain3D` the solid hex pipeline
with the tip load spread along a line across the width.  Everything is
written to a run directory; nothing is plotted or rendered here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .fea import (
    LoadCase,
    Material,
    element_stiffnesses,
    solve as fea_solve,
    timoshenko_tip_deflection,
)
from .geometry import BeamDomain, BeamDomain3D, export_domain
from .mesh_io import (
    check_mesh,
    find_node,
    load_mesh,
    nodes_on_segment,
    save_mesh,
    save_solution,
)
from .meshing import (
    PHYS_FIXED,
    PHYS_LOAD,
    MeshSpec,
    MeshSpec3D,
    generate_fitted_mesh,
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
from .runlog import RunLog
from .topology import save_topology_stl


def run(
    domain: BeamDomain | BeamDomain3D,
    spec: MeshSpec | MeshSpec3D,
    out_dir: Path,
    material: Material | None = None,
    load_fy: float = -1000.0,
    simp: SimpParams | None = None,
    optimize_design: bool = True,
    snapshot_every: int = 10,
    echo: bool = True,
    cad_config: Path | None = None,
) -> tuple[RunLog, dict[str, Any]]:
    """Mesh the domain, solve it at full density, then run the SIMP loop.

    ``cad_config`` is only recorded: the file the domain's parameters were read
    from, if any (see :func:`topocombo.geometry.load_cad_config`).
    """
    three_d = isinstance(domain, BeamDomain3D)
    if three_d != isinstance(spec, MeshSpec3D):
        raise TypeError("a 3D domain needs a MeshSpec3D, a 2D domain a MeshSpec")
    out_dir = Path(out_dir)
    material = material or Material()
    simp = simp or SimpParams()
    # the out-of-plane size: plane-stress thickness in 2D, the modelled width in 3D
    depth = domain.width if three_d else domain.thickness
    analysis = "3D solid" if three_d else "plane-stress"
    log = RunLog(
        name=f"cantilever beam — meshing, {analysis} solve and SIMP optimization",
        out_dir=out_dir,
        echo=echo,
    )
    log.params = {
        "dim": 3 if three_d else 2,
        "element": "H8 hexahedron" if three_d else "Q4 quadrilateral",
        "domain": domain.as_dict(),
        "mesh": spec.as_dict(),
        "material": material.as_dict(),
        "load": {"fy": load_fy},
        "simp": simp.as_dict(),
    }

    import cadquery
    import meshio
    import scipy

    log.tool("cadquery", cadquery.__version__)
    log.tool("meshio", meshio.__version__)
    log.tool("numpy", np.__version__)
    log.tool("scipy", scipy.__version__)

    with log.step("geometry", "1. Define parametric geometry (CadQuery)"):
        if three_d:
            log.log(
                f"design domain {domain.length} x {domain.height} x {domain.width} mm box "
                f"(aspect ratio {domain.aspect_ratio:.2f})"
            )
            (x0, y0, z0), (_, _, z1) = domain.load_line
            log.log(
                f"clamped face: x = 0; tip load along x = {x0:g}, y = {y0:g}, "
                f"z = {z0:g} .. {z1:g}"
            )
            solid = domain.solid()
            log.log(
                f"solid volume: {solid.Volume():.3f} mm^3 (expected {domain.material_volume:.3f})"
            )
            cad_record = {"solid_volume": solid.Volume()}
        else:
            log.log(
                f"design domain {domain.length} x {domain.height} mm "
                f"(aspect ratio {domain.aspect_ratio:.2f}), "
                f"out-of-plane thickness {domain.thickness} mm"
            )
            log.log(f"clamped edge: x = 0; tip load applied at {domain.load_point}")
            face = domain.face()
            log.log(
                f"planar face area: {face.Area():.3f} mm^2 (expected {domain.material_area:.3f})"
            )
            cad_record = {"face_area": face.Area()}
        for x, y, d in domain.holes:
            log.log(f"cutout through z: diameter {d:g} mm at x = {x:g}, y = {y:g}")
        if cad_config is not None:
            log.log(f"CAD parameters read from {cad_config}")
        exported = export_domain(domain, out_dir / "cad")
        log.log("built by running the generated CadQuery script (shown in the report)")
        descriptions = {
            "script": "design domain, the CadQuery script that built it (runs in CQ-editor)",
            "brep": "design domain, BREP format",
            "step": "design domain, STEP format",
            "envelope": "design envelope without cutouts, BREP format"
            + (" — what Gmsh meshes" if spec.structured else " (for the structured mesh mode)"),
        }
        for kind, path in exported.items():
            log.artifact(path, descriptions[kind])
        log.record(
            **cad_record,
            **domain.as_dict(),
            cad_script=domain.cadquery_script(),
            cad_config=None if cad_config is None else str(cad_config),
        )

    cells_word = "hexahedra" if three_d else "quadrilaterals"
    kind = "structured" if spec.structured else "body-fitted"
    with log.step("meshing", f"2. Mesh with Gmsh ({kind} {cells_word})"):
        if spec.structured:
            grid = f"{spec.nelx} x {spec.nely}" + (f" x {spec.nelz}" if three_d else "")
            log.log(
                f"transfinite grid: {grid} = {spec.n_elements} {cells_word}, "
                f"{spec.n_nodes} nodes expected"
            )
            if domain.holes:
                log.log(
                    "the structured grid covers the whole envelope; elements inside the "
                    "cutouts are held void by the optimizer"
                )
            mesher = generate_hex_mesh if three_d else generate_quad_mesh
            brep = exported.get("envelope", exported["brep"])
        else:
            mesher = generate_fitted_mesh
            if three_d:  # the hexes are extruded from a mesh of the x-y profile
                brep = out_dir / "cad" / "design_profile.brep"
                BeamDomain(
                    length=domain.length, height=domain.height, holes=domain.holes
                ).face().exportBrep(str(brep))
                log.artifact(brep, "x-y profile of the design domain, BREP — what Gmsh meshes")
            else:
                brep = exported["brep"]
            log.log(
                "body-fitted: the mesh follows the CAD boundary, cutouts included"
                if domain.holes else "body-fitted: the mesh follows the CAD boundary"
            )
        msh_path, _ = mesher(
            domain=domain,
            spec=spec,
            brep_path=brep,
            out_dir=out_dir / "mesh",
            log=log,
        )
        mesh_word = "hexahedral" if three_d else "quadrilateral"
        log.artifact(msh_path, f"{mesh_word} mesh with physical groups (Gmsh 2.2 ASCII)")
        size = None if spec.structured else spec.element_size(domain)
        log.record(**spec.as_dict(), element_size=size)

    with log.step("validation", "3. Load the mesh and check it"):
        mesh = load_mesh(msh_path)
        if three_d:
            load_nodes = nodes_on_segment(
                mesh, *domain.load_line, candidates=mesh.node_sets[PHYS_LOAD]
            )
            if load_nodes.size == 0:  # pragma: no cover - nely odd: no node row at H/2
                load_nodes = np.array([find_node(mesh, domain.load_point)])
        else:
            load_nodes = np.array([find_node(mesh, domain.load_point)])
        load_node = int(load_nodes[0])
        miss = float(np.linalg.norm(mesh.nodes[load_node][:2] - np.asarray(domain.load_point[:2])))
        if miss > 1e-6 * domain.length:
            raise RuntimeError(f"no mesh node at the load point ({miss:.3g} mm away)")
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
        if load_nodes.size > 1:
            log.log(f"tip load line: {load_nodes.size} nodes, starting at #{load_node} ({coords})")
        else:
            log.log(f"tip load node: #{load_node} at ({coords})")
        for name, passed in summary["checks"].items():
            log.log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not summary["all_checks_passed"]:
            failed = [n for n, ok in summary["checks"].items() if not ok]
            raise RuntimeError(f"mesh validation failed: {', '.join(failed)}")
        log.record(**summary)

    with log.step("export", "4. Write mesh artifacts"):
        paths = save_mesh(
            mesh, out_dir / "mesh", load_node=load_node, load_nodes=load_nodes,
            passive=passive if passive.any() else None,
        )
        log.artifact(paths["npz"], f"nodes, {mesh.cell_type} connectivity and boundary node sets (numpy)")
        log.artifact(paths["vtu"], "mesh for PyVista / ParaView")

    with log.step("solve", f"5. Assemble and solve ({analysis}, full density)"):
        if load_nodes.size > 1:
            load = LoadCase.along_line(mesh, load_nodes, fy=load_fy)
        else:
            load = LoadCase(node=load_node, fy=load_fy)
        log.log(
            f"material: E = {material.youngs_modulus:g} MPa, nu = {material.poisson_ratio:g}, "
            f"G = {material.shear_modulus:g} MPa"
        )
        log.log(
            f"boundary conditions: node set '{PHYS_FIXED}' clamped "
            f"({mesh.dofs_per_node * summary['node_sets'][PHYS_FIXED]} DOFs), "
            + (
                f"Fy = {load.fy:g} N spread over {load.nodes.size} nodes "
                f"(shares {', '.join(f'{v:.3g}' for v in load.node_shares())})"
                if load.nodes.size > 1
                else f"Fy = {load.fy:g} N at node {load.node}"
            )
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
            fixed_node_set=PHYS_FIXED,
            densities=np.where(passive, 0.0, 1.0) if passive.any() else None,
            ke_all=ke_all,
        )
        beam = timoshenko_tip_deflection(
            domain.length, domain.height, depth, material, load.fy
        )
        tip_uy = float(result.component(1)[load.nodes].mean())
        rel = abs(abs(tip_uy) - beam["total"]) / beam["total"]
        reaction_y = float(result.component(1, "reactions").sum())

        log.log(f"solved {result.n_free_dofs} free DOFs (sparse direct)")
        log.log(f"compliance F.U = {result.compliance:.6g} N*mm")
        log.log(
            f"tip deflection uy = {tip_uy:.6g} mm"
            + (" (mean over the load line)" if load.nodes.size > 1 else "")
        )
        log.log(
            f"Timoshenko beam theory: {beam['total']:.6g} mm "
            f"(bending {beam['bending']:.4g} + shear {beam['shear']:.4g}) "
            f"-> {rel * 100:.2f}% difference"
            + (" (beam theory ignores the cutouts)" if domain.holes else "")
        )
        log.log(
            f"equilibrium: ||KU - F|| on free DOFs = {result.equilibrium_residual:.3e}, "
            f"sum of vertical reactions = {reaction_y:.6g} N vs applied {-load.fy:g} N"
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
            "n_free_dofs": result.n_free_dofs,
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
                fixed_node_set=PHYS_FIXED,
                params=simp,
                on_iteration=on_iteration,
                passive=passive,
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
