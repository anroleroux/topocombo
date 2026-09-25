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

from .geometry import Domain
from .meshing import PHYS_DOMAIN


#: Nodes per cell for every supported cell type (meshio naming).
CELL_NODES = {"quad": 4, "hexahedron": 8}

#: Spatial dimension of each cell type, and the boundary cell type that carries
#: its node sets (edges bound a quad mesh, faces bound a hex mesh).
CELL_DIM = {"quad": 2, "hexahedron": 3}
BOUNDARY_CELL = {"quad": "line", "hexahedron": "quad"}

#: Reference coordinates of the cell corners, in Gmsh / VTK node order: quads
#: counter-clockwise; hexes bottom face (zeta = -1) then top face, each
#: counter-clockwise seen from +z.
REFERENCE_NODES = {
    "quad": np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]),
    "hexahedron": np.array(
        [
            [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
        ]
    ),
}

#: Corner pairs joined by an edge.
CELL_EDGES = {
    "quad": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "hexahedron": [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ],
}

#: Gauss-Legendre points (weights 1) of the 2-point rule, per coordinate.
_G = 1.0 / np.sqrt(3.0)


def shape_derivatives(cell_type: str, point: np.ndarray) -> np.ndarray:
    """dN/dxi of the (bi/tri)linear shape functions at ``point``, shape (n_nodes, dim).

    ``N_a = prod_i (1 + xi_a,i * xi_i) / 2^dim`` for both quads and hexes.
    """
    ref = REFERENCE_NODES[cell_type]
    dim = ref.shape[1]
    factors = 1.0 + ref * np.asarray(point, dtype=float)  # (n, dim)
    out = np.empty_like(ref)
    for i in range(dim):
        others = np.prod(np.delete(factors, i, axis=1), axis=1)
        out[:, i] = ref[:, i] * others
    return out / 2.0**dim


def gauss_points(dim: int) -> np.ndarray:
    """The 2^dim points of the 2-point Gauss rule on the reference cell (weights 1)."""
    grids = np.meshgrid(*([[-_G, _G]] * dim), indexing="ij")
    return np.column_stack([g.ravel() for g in grids])


def jacobian_determinants(nodes: np.ndarray, cells: np.ndarray, cell_type: str, point) -> np.ndarray:
    """det(J) of the isoparametric map of every cell at reference ``point``."""
    dn = shape_derivatives(cell_type, np.asarray(point, dtype=float))  # (n, dim)
    jac = np.einsum("ni,enj->eij", dn, nodes[cells])  # (m, dim, dim)
    return np.linalg.det(jac)


@dataclass
class Mesh:
    """A finite-element mesh of one cell type, with named boundary node sets.

    ``nodes`` has one column per spatial dimension, so the same container serves
    the 2D quad mesh and the 3D hex mesh; element routines dispatch on
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
        if self.nodes.ndim != 2 or self.nodes.shape[1] != CELL_DIM[self.cell_type]:
            raise ValueError(
                f"{self.cell_type} meshes need {CELL_DIM[self.cell_type]}D nodes, "
                f"got shape {self.nodes.shape}"
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

    @property
    def measure_name(self) -> str:
        return "area" if self.dim == 2 else "volume"

    def cell_measures(self) -> np.ndarray:
        """Signed size of every cell: area in 2D, volume in 3D.

        Integrates det(J) with the 2-point Gauss rule, which is exact for
        bilinear quads and trilinear hexes; a negative value means the cell is
        inverted (clockwise quad, left-handed hex).
        """
        return sum(
            jacobian_determinants(self.nodes, self.cells, self.cell_type, point)
            for point in gauss_points(self.dim)
        )

    def edge_lengths(self) -> np.ndarray:
        """Lengths of every edge of every cell, shape (n_elements, n_edges)."""
        a, b = np.array(CELL_EDGES[self.cell_type]).T
        xyz = self.nodes[self.cells]
        return np.linalg.norm(xyz[:, b] - xyz[:, a], axis=2)

    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """(lower corner, upper corner), one entry per dimension."""
        return self.nodes.min(axis=0), self.nodes.max(axis=0)


def _physical_tags(mesh: meshio.Mesh) -> dict[str, int]:
    """Map physical-group name -> gmsh physical tag."""
    return {name: int(value[0]) for name, value in mesh.field_data.items()}


def _nodes_in_group(mesh: meshio.Mesh, tag: int, block_type: str) -> np.ndarray:
    """Node indices of every ``block_type`` element carrying physical tag ``tag``."""
    picked: list[np.ndarray] = []
    for block, data in zip(mesh.cells, mesh.cell_data.get("gmsh:physical", [])):
        if block.type != block_type:
            continue
        rows = block.data[np.asarray(data) == tag]
        if rows.size:
            picked.append(rows.reshape(-1))
    if not picked:
        return np.empty(0, dtype=int)
    return np.unique(np.concatenate(picked)).astype(int)


#: Node permutation that mirrors a cell, turning an inverted one right way round:
#: a quad is walked backwards, a hex has both faces walked backwards.
_FLIP = {"quad": [3, 2, 1, 0], "hexahedron": [0, 3, 2, 1, 4, 7, 6, 5]}


def orient_cells(nodes: np.ndarray, cells: np.ndarray, cell_type: str) -> np.ndarray:
    """Reorder inverted cells so det(J) is positive at every cell centre.

    Element stiffness integration needs a positive Jacobian; Gmsh's ordering is
    normally right already, so this is a guard rather than a transformation.
    """
    cells = np.array(cells, copy=True)
    centre = np.zeros(CELL_DIM[cell_type])
    inverted = jacobian_determinants(nodes, cells, cell_type, centre) < 0
    cells[inverted] = cells[inverted][:, _FLIP[cell_type]]
    return cells


def load_mesh(msh_path: Path) -> Mesh:
    """Load a quad (2D) or hex (3D) mesh and its boundary node sets from a Gmsh file.

    The cell type decides the dimension: hexahedra make a 3D mesh whose node
    sets come from boundary faces; otherwise quads make a 2D mesh whose node
    sets come from boundary edges.
    """
    mesh = meshio.read(str(msh_path))
    present = {b.type for b in mesh.cells}
    cell_type = "hexahedron" if "hexahedron" in present else "quad"
    if cell_type not in present:
        raise ValueError(f"{msh_path} contains no quadrilateral or hexahedral elements")

    nodes = np.asarray(mesh.points, dtype=float)[:, : CELL_DIM[cell_type]]
    cells = np.vstack([np.asarray(b.data, dtype=int) for b in mesh.cells if b.type == cell_type])
    cells = orient_cells(nodes, cells, cell_type)

    tags = _physical_tags(mesh)
    node_sets: dict[str, np.ndarray] = {}
    for name, tag in tags.items():  # boundary groups, if the .msh carries any
        if name != PHYS_DOMAIN:
            node_sets[name] = _nodes_in_group(mesh, tag, BOUNDARY_CELL[cell_type])

    return Mesh(nodes=nodes, cells=cells, node_sets=node_sets, cell_type=cell_type)


def find_node(mesh: Mesh, point: tuple[float, ...]) -> int:
    """Index of the mesh node closest to ``point``."""
    d = np.linalg.norm(mesh.nodes - np.asarray(point, dtype=float), axis=1)
    return int(np.argmin(d))


def nodes_on_segment(
    mesh: Mesh,
    start: tuple[float, ...],
    end: tuple[float, ...],
    candidates: np.ndarray | None = None,
    tol: float = 1e-6,
) -> np.ndarray:
    """Nodes lying on the segment ``start``-``end``, ordered from ``start``.

    Used to pick the tip-load line out of the loaded face; ``candidates`` limits
    the search (e.g. to that face's node set).
    """
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    idx = np.arange(mesh.n_nodes) if candidates is None else np.asarray(candidates)
    rel = mesh.nodes[idx] - a
    axis = b - a
    length = float(np.linalg.norm(axis))
    t = rel @ axis / length**2
    off = np.linalg.norm(rel - np.outer(t, axis), axis=1)
    scale = tol * max(length, 1.0)
    on = (off < scale) & (t > -scale / length) & (t < 1.0 + scale / length)
    return idx[on][np.argsort(t[on])]


def _in_plane_edge_ratio(mesh: Mesh) -> np.ndarray:
    """Per element, longest over shortest edge, counting only edges in x-y."""
    a, b = np.array(CELL_EDGES[mesh.cell_type]).T
    xyz = mesh.nodes[mesh.cells]
    vec = xyz[:, b] - xyz[:, a]
    lengths = np.linalg.norm(vec, axis=2)
    if mesh.dim == 3:
        in_plane = np.abs(vec[:, :, 2]) < 1e-9 * lengths.max()
        lengths = np.where(in_plane, lengths, np.nan)
    return np.nanmax(lengths, axis=1) / np.nanmin(lengths, axis=1)


#: Body-fitted checks: the meshed measure may differ from the CAD one by the
#: chordal error of straight element edges on curved boundaries, and no element
#: may be stretched past this edge-length ratio in the x-y plane (extruded
#: hexes are as long through the width as the layers make them, by design).
FITTED_MEASURE_RTOL = 5e-3
FITTED_EDGE_RATIO_MAX = 4.0


def check_mesh(
    mesh: Mesh, domain: Domain, expected_elements: int | None
) -> dict[str, Any]:
    """Validate the mesh against the design domain; return a quality summary.

    Summary keys name the cell measure — ``area_*`` for a 2D mesh,
    ``volume_*`` for a 3D one.  ``expected_elements`` is the structured grid's
    count; None marks a body-fitted mesh, which is checked against the CAD
    shape (cutouts removed) and for element quality instead.
    """
    fitted = expected_elements is None
    if domain.dim == 3:
        extent = np.array([domain.length, domain.height, domain.width])
        domain_measure = domain.material_volume if fitted else domain.volume
    else:
        extent = np.array([domain.length, domain.height])
        domain_measure = domain.material_area if fitted else domain.area
    if extent.size != mesh.dim:
        raise ValueError(f"a {mesh.dim}D mesh cannot be checked against a {extent.size}D domain")

    name = mesh.measure_name
    measures = mesh.cell_measures()
    edges = mesh.edge_lengths()
    aspect = edges.max(axis=1) / edges.min(axis=1)
    lower, upper = mesh.bounding_box()

    rtol = FITTED_MEASURE_RTOL if fitted else 1e-6
    checks = {}
    if fitted:
        checks[f"edge_ratio_below_{FITTED_EDGE_RATIO_MAX:g}"] = bool(
            _in_plane_edge_ratio(mesh).max() < FITTED_EDGE_RATIO_MAX
        )
    else:
        checks["element_count_matches_spec"] = mesh.n_elements == expected_elements
    checks |= {
        f"all_elements_positive_{name}": bool(np.all(measures > 0)),
        f"{name}_sum_matches_{'cad' if fitted else 'domain'}": bool(
            abs(measures.sum() - domain_measure) < rtol * domain_measure
        ),
        "bbox_matches_domain": bool(
            np.all(np.abs(lower) < 1e-9) and np.all(np.abs(upper - extent) < 1e-9 * extent)
        ),
        "no_orphan_nodes": bool(np.unique(mesh.cells).size == mesh.n_nodes),
    }
    for set_name, idx in mesh.node_sets.items():
        checks[f"region_{set_name}_has_nodes"] = bool(np.asarray(idx).size > 0)

    return {
        "n_nodes": mesh.n_nodes,
        "n_elements": mesh.n_elements,
        "n_dofs": mesh.n_dofs,
        "dim": mesh.dim,
        "cell_type": mesh.cell_type,
        f"{name}_sum": float(measures.sum()),
        f"domain_{name}": domain_measure,
        f"element_{name}_min": float(measures.min()),
        f"element_{name}_max": float(measures.max()),
        "edge_length_min": float(edges.min()),
        "edge_length_max": float(edges.max()),
        "aspect_ratio_max": float(aspect.max()),
        "bounding_box": [float(v) for v in (*lower, *upper)],
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
    load_nodes: np.ndarray | None = None,
    passive: np.ndarray | None = None,
    fixed_nodes: np.ndarray | None = None,
    load_vector: Any = None,
) -> dict[str, Path]:
    """Write the solver-facing ``mesh.npz`` and the visualisation-facing ``mesh.vtu``.

    ``passive`` marks the elements held void (inside a CAD cutout).
    """
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
    if load_nodes is not None:  # a load spread over several nodes (3D load line)
        arrays["load_nodes"] = np.asarray(load_nodes, dtype=int)
    if passive is not None:
        arrays["passive"] = np.asarray(passive, dtype=bool)
    if fixed_nodes is not None:  # every constrained node, whatever its region
        arrays["fixed_nodes"] = np.asarray(fixed_nodes, dtype=int)
    if load_vector is not None:  # the total force, for drawing its direction
        arrays["load_vector"] = np.asarray(load_vector, dtype=float)

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
