"""The geometry → mesh stage of the main flow, as one terminal-driven run.

Everything is written to a run directory; nothing is plotted or rendered here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import BeamDomain, export_domain
from .mesh_io import check_mesh, find_node, load_mesh, save_mesh
from .meshing import MeshSpec, generate_quad_mesh
from .runlog import RunLog


def run(
    domain: BeamDomain,
    spec: MeshSpec,
    out_dir: Path,
    echo: bool = True,
) -> tuple[RunLog, dict[str, Any]]:
    """Build the design domain, mesh it, validate the mesh and export artifacts."""
    out_dir = Path(out_dir)
    log = RunLog(name="cantilever beam — geometry and meshing", out_dir=out_dir, echo=echo)
    log.params = {"domain": domain.as_dict(), "mesh": spec.as_dict()}

    import cadquery
    import gmsh
    import meshio
    import numpy

    log.tool("cadquery", cadquery.__version__)
    log.tool("meshio", meshio.__version__)
    log.tool("numpy", numpy.__version__)

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

        log.log(f"{summary['n_nodes']} nodes, {summary['n_elements']} quads, {summary['n_dofs']} DOFs")
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

    with log.step("export", "4. Write result artifacts"):
        paths = save_mesh(mesh, out_dir / "mesh", load_node=load_node)
        log.artifact(paths["npz"], "nodes, quad connectivity and boundary node sets (numpy)")
        log.artifact(paths["vtu"], "mesh for PyVista / ParaView")
        log.log("next stages (FEA solve, SIMP loop) consume mesh.npz; nothing here plots.")

    json_path, text_path = log.write()
    print(f"\nrun log: {json_path}")
    print(f"terminal log: {text_path}")
    return log, summary
