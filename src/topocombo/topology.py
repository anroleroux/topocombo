"""The optimized topology as a closed triangle surface (STL), for Blender & co.

The density field is thresholded per element (solid where ``rho >= threshold``)
and the surface is the set of solid-element faces that no other solid element
shares.  That gives a blocky but always watertight, outward-oriented surface
with no extra dependencies; smoothing it is left to the renderer.

A 2D quad mesh is extruded by its out-of-plane thickness first, so both
pipelines produce a solid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import meshio
import numpy as np

from .mesh_io import Mesh

#: The six faces of a hexahedron, corners ordered so the right-hand normal
#: points out of a positively oriented cell (VTK / Gmsh reference numbering).
HEX_FACES = np.array(
    [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
)


def extrude(mesh: Mesh, thickness: float) -> Mesh:
    """A quad mesh extruded along +z into one layer of hexahedra.

    Counter-clockwise quads become positively oriented hexes (bottom face at
    z = 0, top face at z = thickness), keeping one cell per element.
    """
    if mesh.cell_type != "quad":
        raise ValueError(f"only quad meshes can be extruded, not {mesh.cell_type!r}")
    n = mesh.n_nodes
    nodes = np.vstack(
        [
            np.column_stack([mesh.nodes, np.zeros(n)]),
            np.column_stack([mesh.nodes, np.full(n, float(thickness))]),
        ]
    )
    cells = np.hstack([mesh.cells, mesh.cells + n])
    return Mesh(nodes=nodes, cells=cells, node_sets={}, cell_type="hexahedron")


def boundary_faces(cells: np.ndarray) -> np.ndarray:
    """Outward quads bounding the union of ``cells`` (hexes), shape (k, 4)."""
    faces = cells[:, HEX_FACES].reshape(-1, 4)
    keys = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return faces[counts[inverse.ravel()] == 1]


def solid_surface(
    mesh: Mesh, densities: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of the thresholded design's closed outer surface."""
    solid = np.asarray(densities) >= threshold
    quads = boundary_faces(mesh.cells[solid])
    triangles = np.vstack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    # keep only the points the surface uses, renumbered
    used, tri = np.unique(triangles, return_inverse=True)
    return mesh.nodes[used], tri.reshape(triangles.shape)


def enclosed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    """Volume inside a closed, outward-oriented triangle surface (divergence theorem)."""
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def save_topology_stl(
    mesh: Mesh,
    densities: np.ndarray,
    path: Path,
    threshold: float = 0.5,
    thickness: float = 1.0,
) -> dict[str, Any]:
    """Write the thresholded design as a binary STL; return a short summary.

    ``thickness`` extrudes a 2D mesh and is ignored for a 3D one.
    """
    solid_mesh = extrude(mesh, thickness) if mesh.dim == 2 else mesh
    points, triangles = solid_surface(solid_mesh, densities, threshold)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write_points_cells(str(path), points, [("triangle", triangles)], binary=True)
    solid = np.asarray(densities) >= threshold
    return {
        "path": path,
        "threshold": threshold,
        "solid_elements": int(solid.sum()),
        "triangles": int(triangles.shape[0]),
        "volume": enclosed_volume(points, triangles) if triangles.size else 0.0,
        "solid_element_volume": float(solid_mesh.cell_measures()[solid].sum()),
    }
