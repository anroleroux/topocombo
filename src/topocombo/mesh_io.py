"""Read the Gmsh mesh into plain arrays and check it.

The SIMP loop and the visualisation scripts both consume on-disk artifacts
rather than talking to Gmsh, so this module is the single place where the .msh
file is interpreted.  It produces:

* ``mesh.npz``  — nodes, cell connectivity and boundary node sets (numpy)
* ``mesh.vtu``  — the same mesh for PyVista/ParaView

and a quality summary used to validate the mesh before any solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import meshio
import numpy as np

from .geometry import BeamDomain
from .meshing import PHYS_FIXED, PHYS_LOAD


#: Nodes per cell for every supported cell type (meshio naming).
CELL_NODES = {"quad": 4}


@dataclass
class Mesh:
    """A finite-element mesh of one cell type, with named boundary node sets.

    ``nodes`` has one column per spatial dimension, so the same container serves
    the 2D quad mesh and, later, a 3D hex mesh; element routines dispatch on
    ``cell_type``.
    """

    nodes: np.ndarray  # (n_nodes, dim) float
    cells: np.ndarray  # (n_elements, nodes_per_cell) int; quads counter-clockwise
    node_sets: dict[str, np.ndarray]  # name -> node indices
    cell_type: str = "quad"

    def __post_init__(self) -> None:
        if self.cell_type not in CELL_NODES:
            raise ValueError(f"unsupported cell type {self.cell_type!r}")
        if self.cells.ndim != 2 or self.cells.shape[1] != CELL_NODES[self.cell_type]:
            raise ValueError(
                f"{self.cell_type} cells need {CELL_NODES[self.cell_type]} nodes each, "
                f"got shape {self.cells.shape}"
            )

    @property
    def dim(self) -> int:
        return int(self.nodes.shape[1])

    @property
    def dofs_per_node(self) -> int:
        """Displacement components per node: one per spatial dimension."""
        return self.dim

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.cells.shape[0])

    @property
    def n_dofs(self) -> int:
        return self.dofs_per_node * self.n_nodes

    def cell_measures(self) -> np.ndarray:
        """Size of every cell: area in 2D, volume in 3D."""
        if self.cell_type == "quad":
            return self._quad_areas()
        raise NotImplementedError(f"cell_measures for {self.cell_type!r}")  # pragma: no cover

    def _quad_areas(self) -> np.ndarray:
        """Signed shoelace area of every quad (positive = counter-clockwise)."""
        xy = self.nodes[self.cells]  # (m, 4, 2)
        x, y = xy[:, :, 0], xy[:, :, 1]
        return 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)

    def edge_lengths(self) -> np.ndarray:
        """Lengths of the four edges of every quad, shape (n_elements, 4)."""
        if self.cell_type != "quad":  # pragma: no cover - hex edges come in step 3
            raise NotImplementedError(f"edge_lengths for {self.cell_type!r}")
        xy = self.nodes[self.cells]
        return np.linalg.norm(np.roll(xy, -1, axis=1) - xy, axis=2)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower corner, upper corner), one entry per dimension."""
        return self.nodes.min(axis=0), self.nodes.max(axis=0)


def _physical_tags(mesh: meshio.Mesh) -> dict[str, int]:
    """Map physical-group name -> gmsh physical tag."""
    return {name: int(value[0]) for name, value in mesh.field_data.items()}


def _nodes_in_group(mesh: meshio.Mesh, tag: int) -> np.ndarray:
    """Node indices of every line element carrying physical tag ``tag``."""
    picked: list[np.ndarray] = []
    for block, data in zip(mesh.cells, mesh.cell_data.get("gmsh:physical", [])):
        if block.type != "line":
            continue
        rows = block.data[np.asarray(data) == tag]
        if rows.size:
            picked.append(rows.reshape(-1))
    if not picked:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(picked)).astype(int)


def load_mesh(msh_path: Path) -> Mesh:
    """Load a 2D quad mesh and its boundary node sets from a Gmsh file."""
    mesh = meshio.read(str(msh_path))
    nodes = np.asarray(mesh.points, dtype=float)[:, :2]

    quad_blocks = [np.asarray(b.data, dtype=int) for b in mesh.cells if b.type == "quad"]
    if not quad_blocks:
        raise ValueError(f"{msh_path} contains no quadrilateral elements")
    quads = np.vstack(quad_blocks)

    # Enforce counter-clockwise orientation so element stiffness integration
    # later sees a positive Jacobian everywhere.
    xy = nodes[quads]
    x, y = xy[:, :, 0], xy[:, :, 1]
    signed = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    quads[signed < 0] = quads[signed < 0][:, ::-1]

    tags = _physical_tags(mesh)
    node_sets: dict[str, np.ndarray] = {}
    for name in (PHYS_FIXED, PHYS_LOAD):
        if name in tags:
            node_sets[name] = _nodes_in_group(mesh, tags[name])

    return Mesh(nodes=nodes, cells=quads, node_sets=node_sets, cell_type="quad")


def find_node(mesh: Mesh, point: tuple[float, ...]) -> int:
    """Index of the mesh node closest to ``point``."""
    d = np.linalg.norm(mesh.nodes - np.asarray(point, dtype=float), axis=1)
    return int(np.argmin(d))


def check_mesh(mesh: Mesh, domain: BeamDomain, expected_elements: int) -> dict[str, Any]:
    """Validate the mesh against the design domain; return a quality summary."""
    areas = mesh.cell_measures()
    edges = mesh.edge_lengths()
    aspect = edges.max(axis=1) / edges.min(axis=1)
    lower, upper = mesh.bounding_box()
    xmin, ymin = (float(v) for v in lower)
    xmax, ymax = (float(v) for v in upper)

    checks = {
        "element_count_matches_spec": mesh.n_elements == expected_elements,
        "all_elements_positive_area": bool(np.all(areas > 0)),
        "area_sum_matches_domain": bool(abs(areas.sum() - domain.area) < 1e-6 * domain.area),
        "bbox_matches_domain": bool(
            abs(xmin) < 1e-9
            and abs(ymin) < 1e-9
            and abs(xmax - domain.length) < 1e-9 * domain.length
            and abs(ymax - domain.height) < 1e-9 * domain.height
        ),
        "no_orphan_nodes": bool(np.unique(mesh.cells).size == mesh.n_nodes),
        "fixed_set_non_empty": bool(mesh.node_sets.get(PHYS_FIXED, np.empty(0)).size > 0),
        "load_set_non_empty": bool(mesh.node_sets.get(PHYS_LOAD, np.empty(0)).size > 0),
    }

    return {
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "n_dofs": mesh.n_dofs,
        "area_sum": float(areas.sum()),
        "domain_area": domain.area,
        "element_area_min": float(areas.min()),
        "element_area_max": float(areas.max()),
        "edge_length_min": float(edges.min()),
        "edge_length_max": float(edges.max()),
        "aspect_ratio_max": float(aspect.max()),
        "bounding_box": [xmin, ymin, xmax, ymax],
        "node_sets": {name: int(idx.size) for name, idx in mesh.node_sets.items()},
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def vtk_points(mesh: Mesh) -> np.ndarray:
    """Node coordinates padded to three columns, as VTK expects."""
    return np.column_stack([mesh.nodes, np.zeros((mesh.n_nodes, 3 - mesh.dim))])


def vtk_vectors(values: np.ndarray) -> np.ndarray:
    """Per-node vectors padded to three components, as VTK expects."""
    return np.column_stack([values, np.zeros((values.shape[0], 3 - values.shape[1]))])


def save_mesh(
    mesh: Mesh,
    out_dir: Path,
    load_node: int | None = None,
) -> dict[str, Path]:
    """Write the solver-facing ``mesh.npz`` and the visualisation-facing ``mesh.vtu``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, Any] = {
        "nodes": mesh.nodes,
        "cells": mesh.cells,
        "cell_type": np.asarray(mesh.cell_type),
    }
    for name, idx in mesh.node_sets.items():
        arrays[f"set_{name}"] = idx
    if load_node is not None:
        arrays["load_node"] = np.asarray([load_node], dtype=int)

    npz = out_dir / "mesh.npz"
    np.savez_compressed(npz, **arrays)

    vtu = out_dir / "mesh.vtu"
    meshio.write_points_cells(str(vtu), vtk_points(mesh), [(mesh.cell_type, mesh.cells)])

    return {"npz": npz, "vtu": vtu}


def save_solution(mesh: Mesh, result: Any, out_dir: Path) -> dict[str, Path]:
    """Write the FEA result as ``solution.npz`` (solver-facing) and ``solution.vtu``.

    ``result`` is a :class:`topocombo.fea.FEResult`; it is taken structurally to
    keep this module free of a dependency on the solver.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    displacements = result.u.reshape(-1, mesh.dofs_per_node)
    npz = out_dir / "solution.npz"
    np.savez_compressed(
        npz,
        displacements=displacements,
        compliance=np.asarray([result.compliance]),
        element_compliance=result.element_compliance,
        von_mises=result.von_mises,
        reactions=result.reactions.reshape(-1, mesh.dofs_per_node),
    )

    vtu = out_dir / "solution.vtu"
    meshio.write_points_cells(
        str(vtu),
        vtk_points(mesh),
        [(mesh.cell_type, mesh.cells)],
        point_data={"displacement": vtk_vectors(displacements)},
        cell_data={
            "element_compliance": [result.element_compliance],
            "von_mises": [result.von_mises],
        },
    )

    return {"npz": npz, "vtu": vtu}
