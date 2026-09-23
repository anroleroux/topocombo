"""The geometry → mesh stage of the main flow, as one terminal-driven run.

Everything is written to a run directory; nothing is plotted or rendered here.
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
from .geometry import BeamDomain, export_domain
from .mesh_io import check_mesh, find_node, load_mesh, save_mesh, save_solution
from .meshing import PHYS_FIXED, MeshSpec, generate_quad_mesh
from .optimize import (
    SimpParams,
    optimize,
    save_density_field,
    save_design,
    save_history,
)
from .runlog import RunLog


def run(
    domain: BeamDomain,
    spec: MeshSpec,
    out_dir: Path,
    material: Material | None = None,
    load_fy: float = -1000.0,
    simp: SimpParams | None = None,
    optimize_design: bool = True,
    snapshot_every: int = 10,
    echo: bool = True,
) -> tuple[RunLog, dict[str, Any]]:
    """Mesh the domain, solve it at full density, then run the SIMP loop."""
    out_dir = Path(out_dir)
    material = material or Material()
    simp = simp or SimpParams()
    log = RunLog(
        name="cantilever beam — meshing, plane-stress solve and SIMP optimization",
        out_dir=out_dir,
        echo=echo,
    )
    log.params = {
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
        log.log(
            f"design domain {domain.length} x {domain.height} mm "
            f"(aspect ratio {domain.aspect_ratio:.2f}), "
            f"out-of-plane thickness {domain.thickness} mm"
        )
        log.log(f"clamped edge: x = 0; tip load applied at {domain.load_point}")
        face = domain.face()
        log.log(f"planar face area: {face.Area():.3f} mm^2 (expected {domain.area:.3f})")
        exported = export_domain(domain, out_dir / "cad")
        for kind, path in exported.items():
            log.artifact(path, f"design domain, {kind.upper()} format")
        log.record(face_area=face.Area(), **domain.as_dict())

    with log.step("meshing", "2. Mesh with Gmsh (structured quadrilaterals)"):
        log.log(
            f"transfinite grid: {spec.nelx} x {spec.nely} = {spec.n_elements} quads, "
            f"{spec.n_nodes} nodes expected"
        )
        msh_path, _ = generate_quad_mesh(
            domain=domain,
            spec=spec,
            brep_path=exported["brep"],
            out_dir=out_dir / "mesh",
            log=log,
        )
        log.artifact(msh_path, "quadrilateral mesh with physical groups (Gmsh 2.2 ASCII)")
        log.record(**spec.as_dict())

    with log.step("validation", "3. Load the mesh and check it"):
        mesh = load_mesh(msh_path)
        load_node = find_node(mesh, domain.load_point)
        summary = check_mesh(mesh, domain, expected_elements=spec.n_elements)
        summary["load_node"] = load_node
        summary["load_node_coords"] = [float(c) for c in mesh.nodes[load_node]]

        log.log(f"{summary['n_nodes']} nodes, {summary['n_elements']} {mesh.cell_type}s, {summary['n_dofs']} DOFs")
        log.log(
            "element edge lengths: "
            f"{summary['edge_length_min']:.4f} – {summary['edge_length_max']:.4f} mm, "
            f"max aspect ratio {summary['aspect_ratio_max']:.4f}"
        )
        log.log(
            f"meshed area {summary['area_sum']:.6f} mm^2 vs domain {summary['domain_area']:.6f} mm^2"
        )
        for name, count in summary["node_sets"].items():
            log.log(f"node set '{name}': {count} nodes")
        log.log(
            f"tip load node: #{load_node} at "
            f"({summary['load_node_coords'][0]:.3f}, {summary['load_node_coords'][1]:.3f})"
        )
        for name, passed in summary["checks"].items():
            log.log(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not summary["all_checks_passed"]:
            failed = [n for n, ok in summary["checks"].items() if not ok]
            raise RuntimeError(f"mesh validation failed: {', '.join(failed)}")
        log.record(**summary)

    with log.step("export", "4. Write mesh artifacts"):
        paths = save_mesh(mesh, out_dir / "mesh", load_node=load_node)
        log.artifact(paths["npz"], f"nodes, {mesh.cell_type} connectivity and boundary node sets (numpy)")
        log.artifact(paths["vtu"], "mesh for PyVista / ParaView")

    with log.step("solve", "5. Assemble and solve (plane stress, full density)"):
        load = LoadCase(node=load_node, fy=load_fy)
        log.log(
            f"material: E = {material.youngs_modulus:g} MPa, nu = {material.poisson_ratio:g}, "
            f"G = {material.shear_modulus:g} MPa"
        )
        log.log(
            f"boundary conditions: node set '{PHYS_FIXED}' clamped "
            f"({mesh.dofs_per_node * summary['node_sets'][PHYS_FIXED]} DOFs), "
            f"Fy = {load.fy:g} N at node {load.node}"
        )
        ke_all = element_stiffnesses(mesh, material, domain.thickness)
        n_edof = ke_all.shape[1]
        log.log(
            f"assembled {ke_all.shape[0]} element stiffness matrices "
            f"({n_edof}x{n_edof}, 2x2 Gauss)"
        )
        result = fea_solve(
            mesh=mesh,
            material=material,
            thickness=domain.thickness,
            load=load,
            fixed_node_set=PHYS_FIXED,
            ke_all=ke_all,
        )
        beam = timoshenko_tip_deflection(
            domain.length, domain.height, domain.thickness, material, load.fy
        )
        tip_uy = float(result.component(1)[load_node])
        rel = abs(abs(tip_uy) - beam["total"]) / beam["total"]
        reaction_y = float(result.component(1, "reactions").sum())

        log.log(f"solved {result.n_free_dofs} free DOFs (sparse direct)")
        log.log(f"compliance F.U = {result.compliance:.6g} N*mm")
        log.log(f"tip deflection uy = {tip_uy:.6g} mm")
        log.log(
            f"Timoshenko beam theory: {beam['total']:.6g} mm "
            f"(bending {beam['bending']:.4g} + shear {beam['shear']:.4g}) "
            f"-> {rel * 100:.2f}% difference"
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
                thickness=domain.thickness,
                load=load,
                fixed_node_set=PHYS_FIXED,
                params=simp,
                on_iteration=on_iteration,
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

    json_path, text_path = log.write()
    print(f"\nrun log: {json_path}")
    print(f"terminal log: {text_path}")
    return log, summary
