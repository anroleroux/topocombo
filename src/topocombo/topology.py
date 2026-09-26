"""The optimized topology as a closed triangle surface (STL), for Blender & co.

Two surfaces are written from the density field:

* the **element surface** — every element with ``rho >= threshold`` solid, and
  the surface the set of solid-element faces no other solid element shares.
  Blocky (each step is one element) but exact: it encloses precisely the
  solid elements.
* the **smooth surface** — the contour ``rho = threshold`` of the density field
  made continuous.  Element densities are averaged onto the nodes (weighted by
  volume), every element is split into linear tetrahedra, and marching
  tetrahedra cuts each one where its linearly varying density crosses the
  threshold, so the surface runs *through* elements instead of along their
  faces.  Where the solid reaches the part's own boundary the boundary facets
  are clipped to the solid side, closing the surface.  Pieces of solid that
  touch no support are dropped (they float: nothing holds them), and a few
  passes of Taubin smoothing take out the remaining facet noise without
  shrinking the part.

A 2D quad mesh is extruded by its out-of-plane thickness first, so both
pipelines produce a solid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import meshio
import numpy as np

from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .elements import HEX8, TET4, boundary_facets, triangulate
from .mesh_io import Mesh

#: The six faces of a hexahedron, corners ordered so the right-hand normal
#: points out of a positively oriented cell (VTK / Gmsh reference numbering).
HEX_FACES = np.array(HEX8.facets)


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


def boundary_faces(cells: np.ndarray, el=HEX8) -> np.ndarray:
    """Outward facets bounding the union of ``cells`` (hexes by default)."""
    return boundary_facets(el, cells)


def solid_surface(
    mesh: Mesh, densities: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """(points, triangles) of the thresholded design's closed outer surface."""
    solid = np.asarray(densities) >= threshold
    triangles = triangulate(boundary_faces(mesh.cells[solid], mesh.element))
    # keep only the points the surface uses, renumbered
    used, tri = np.unique(triangles, return_inverse=True)
    return mesh.nodes[used], tri.reshape(triangles.shape)


def enclosed_volume(points: np.ndarray, triangles: np.ndarray) -> float:
    """Volume inside a closed, outward-oriented triangle surface (divergence theorem)."""
    a, b, c = (points[triangles[:, i]] for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# --------------------------------------------------------------------------
# the smooth surface: marching tetrahedra on the nodal density
# --------------------------------------------------------------------------
#: A 10-node tetrahedron as eight linear ones: a corner tet at each corner, and
#: the inner octahedron cut along the diagonal from mid-node 01 to mid-node 23
#: (mid-nodes 4..9 on edges 01, 12, 02, 03, 13, 23).
_TET10_SPLIT = np.array(
    [[0, 4, 6, 7], [4, 1, 5, 8], [6, 5, 2, 9], [7, 8, 9, 3],
     [4, 9, 5, 6], [4, 9, 6, 7], [4, 9, 7, 8], [4, 9, 8, 5]]
)
_TET_EDGES = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])


def nodal_densities(mesh: Mesh, densities: np.ndarray) -> np.ndarray:
    """Element densities averaged onto every node, weighted by element size."""
    weights = mesh.cell_measures()
    k = mesh.cells.shape[1]
    num = np.bincount(mesh.cells.ravel(), np.repeat(weights * densities, k), mesh.n_nodes)
    den = np.bincount(mesh.cells.ravel(), np.repeat(weights, k), mesh.n_nodes)
    return num / np.where(den > 0, den, 1.0)


def _signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = (points[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def linear_tets(mesh: Mesh, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(points, tets, values): ``mesh`` split into positively oriented linear
    tetrahedra that share faces with their neighbours.

    The mesh's own nodes come first, with their indices and values.  A T10
    splits into eight tets on its mid-nodes; a hexahedron gets a node at each
    face centre and at its centre (the mean of their corners' values), and
    splits into 24 tets — one per face quarter — so neighbouring hexes cut
    their shared face the same way whatever their orientation.
    """
    cell_type = mesh.cell_type
    if cell_type == "tetra":
        points, tets, vals = mesh.nodes, mesh.cells, values
    elif cell_type == "tetra10":
        points, vals = mesh.nodes, values
        tets = mesh.cells[:, _TET10_SPLIT].reshape(-1, 4)
    elif cell_type == "hexahedron":
        n, cells = mesh.n_nodes, mesh.cells
        faces = cells[:, HEX_FACES]  # (e, 6, 4), outward
        keys, face_id = np.unique(np.sort(faces.reshape(-1, 4), axis=1), axis=0, return_inverse=True)
        face_id = face_id.reshape(len(cells), 6) + n
        centre_id = np.arange(len(cells)) + n + len(keys)
        points = np.vstack([mesh.nodes, mesh.nodes[keys].mean(axis=1), mesh.nodes[cells].mean(axis=1)])
        vals = np.concatenate([values, values[keys].mean(axis=1), values[cells].mean(axis=1)])
        rolled = np.roll(faces, -1, axis=2)
        tets = np.stack(
            [faces, rolled, np.repeat(face_id[:, :, None], 4, axis=2),
             np.broadcast_to(centre_id[:, None, None], faces.shape)],
            axis=-1,
        ).reshape(-1, 4)
    else:
        raise ValueError(f"no tetrahedral split for {cell_type!r} cells")
    tets = np.array(tets)
    negative = _signed_volumes(points, tets) < 0
    tets[negative] = tets[negative][:, [0, 2, 1, 3]]
    return points, tets, vals


def _cut_points(points, values, iso, edges):
    """Where ``values`` crosses ``iso`` along each (i, j) edge (unique rows)."""
    vi, vj = values[edges[:, 0]], values[edges[:, 1]]
    t = np.clip((iso - vi) / np.where(vj != vi, vj - vi, 1.0), 0.0, 1.0)
    return points[edges[:, 0]] + t[:, None] * (points[edges[:, 1]] - points[edges[:, 0]])


def _orient(tri_pts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """True where triangle ``tri_pts`` (k, 3, 3) must flip to face ``direction``."""
    normal = np.cross(tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0])
    return np.einsum("ij,ij->i", normal, direction) < 0


def anchored(tets: np.ndarray, inside: np.ndarray, anchors: np.ndarray | None) -> np.ndarray:
    """``inside`` less the pieces that reach no anchor node.

    The solid region ``value >= iso`` of a linear tet is convex, so two solid
    nodes are in one piece exactly when a chain of tet edges joins them through
    solid nodes; a piece none of whose nodes is an anchor floats.  When no
    piece reaches an anchor (a design far from converged) all are kept."""
    if anchors is None or not inside[np.asarray(anchors, dtype=int)].any():
        return inside
    label = _pieces(tets, inside)
    anchors = np.asarray(anchors, dtype=int)
    held = np.unique(label[anchors[inside[anchors]]])
    return inside & np.isin(label, held)


def _pieces(tets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A piece label per node: nodes in ``mask`` joined by tet edges share one."""
    e = tets[:, _TET_EDGES].reshape(-1, 2)
    e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    n = len(mask)
    graph = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def taubin(points: np.ndarray, triangles: np.ndarray, fixed: np.ndarray,
           steps: int = 10, lam: float = 0.5, mu: float = -0.53) -> np.ndarray:
    """Taubin smoothing: a shrinking then an inflating Laplacian step, repeated,
    which smooths without the volume loss of plain Laplacian smoothing.
    ``fixed`` vertices stay put."""
    n = len(points)
    e = triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    adj = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    adj = ((adj + adj.T) > 0).astype(float)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    free = (~fixed & (degree > 0))[:, None]
    p = points.copy()
    for _ in range(steps):
        for factor in (lam, mu):
            mean = (adj @ p) / np.maximum(degree, 1.0)[:, None]
            p = np.where(free, p + factor * (mean - p), p)
    return p


def smooth_surface(
    mesh: Mesh,
    densities: np.ndarray,
    iso: float = 0.5,
    anchors: np.ndarray | None = None,
    smoothing: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """(points, triangles, facts) of the closed, outward surface ``rho = iso``.

    ``anchors`` are the mesh nodes that hold the part (the supports); pieces
    of solid touching none of them are dropped.  ``smoothing`` is the number
    of Taubin passes over the vertices inside the part (those on its
    boundary stay on it)."""
    values = nodal_densities(mesh, np.asarray(densities, dtype=float))
    points, tets, vals = linear_tets(mesh, values)
    vals = np.where(np.abs(vals - iso) < 1e-9, iso, vals)  # averages that round off iso
    solid = vals >= iso
    kept = anchored(tets, solid, anchors)
    vals = np.where(solid & ~kept, -1.0, vals)
    inside = vals >= iso

    # every tet edge that changes side carries one cut point, shared by all
    # the tets (and boundary triangles) around it
    all_edges = np.sort(tets[:, _TET_EDGES].reshape(-1, 2), axis=1)
    crossing = inside[all_edges[:, 0]] != inside[all_edges[:, 1]]
    cut_edges = np.unique(all_edges[crossing], axis=0)
    cuts = _cut_points(points, vals, iso, cut_edges)
    cut_base = len(points)
    # a cut at a node whose value is exactly ``iso`` is that node: one vertex,
    # not one per edge (duplicates would tear apart when smoothed)
    at_end = np.where(inside[cut_edges[:, 0]], cut_edges[:, 0], cut_edges[:, 1])
    vertex = np.where(vals[at_end] == iso, at_end, cut_base + np.arange(len(cut_edges)))

    def cut_index(i, j):
        key = np.sort(np.column_stack([i, j]), axis=1)
        hit = np.searchsorted(cut_edges[:, 0] * len(points) + cut_edges[:, 1],
                              key[:, 0] * len(points) + key[:, 1])
        return vertex[hit]

    all_points = np.vstack([points, cuts])
    tris = []

    # interior: marching tetrahedra, oriented down the density gradient
    s = inside[tets]
    k = s.sum(axis=1)
    order = np.argsort(~s, axis=1, kind="stable")  # inside corners first
    t = np.take_along_axis(tets, order, axis=1)
    grad = _gradients(points, tets, vals)
    for count in (1, 3):
        sel = k == count
        a = t[sel]
        if count == 1:  # corner 0 inside, 1 2 3 out
            tri = np.column_stack([cut_index(a[:, 0], a[:, i]) for i in (1, 2, 3)])
        else:  # corner 3 out
            tri = np.column_stack([cut_index(a[:, i], a[:, 3]) for i in (0, 1, 2)])
        tris.append((tri, grad[sel]))
    sel = k == 2
    a = t[sel]  # 0 1 in, 2 3 out: cut edges 02 03 13 12 go round the quad
    quad = np.column_stack([cut_index(a[:, 0], a[:, 2]), cut_index(a[:, 0], a[:, 3]),
                            cut_index(a[:, 1], a[:, 3]), cut_index(a[:, 1], a[:, 2])])
    g = grad[sel]
    tris += [(quad[:, [0, 1, 2]], g), (quad[:, [0, 2, 3]], g)]
    iso_tris = []
    for tri, g in tris:
        flip = _orient(all_points[tri], -g)  # outward: towards falling density
        tri[flip] = tri[flip][:, [0, 2, 1]]
        iso_tris.append(tri)

    # the part's own boundary, clipped to the solid side, closes the surface
    facets = boundary_facets(TET4, tets)  # outward
    fs = inside[facets]
    fk = fs.sum(axis=1)
    cap = [facets[fk == 3]]
    for count in (1, 2):
        f, odd = facets[fk == count], fs[fk == count]
        # rotate each triangle (keeping its orientation) so the odd corner is first
        shift = np.argmax(odd if count == 1 else ~odd, axis=1)
        f = np.take_along_axis(f, (shift[:, None] + np.arange(3)) % 3, axis=1)
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        if count == 1:  # a inside
            cap.append(np.column_stack([a, cut_index(a, b), cut_index(a, c)]))
        else:  # a outside, b c inside
            ab, ca = cut_index(a, b), cut_index(c, a)
            cap.append(np.column_stack([ab, b, c]))
            cap.append(np.column_stack([ab, c, ca]))

    triangles = np.vstack(iso_tris + cap)
    triangles = triangles[(triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                          & (triangles[:, 2] != triangles[:, 0])]
    # vertices on the part's boundary stay on it while the rest is smoothed
    on_boundary = np.zeros(len(all_points), dtype=bool)
    on_boundary[facets.ravel()] = True
    bedges = np.unique(np.sort(facets[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0)
    bkeys = bedges[:, 0] * len(points) + bedges[:, 1]
    ckeys = cut_edges[:, 0] * len(points) + cut_edges[:, 1]
    on_boundary[cut_base:] = np.isin(ckeys, bkeys)
    used, tri = np.unique(triangles, return_inverse=True)
    tri = tri.reshape(triangles.shape)
    pts, fixed = all_points[used], on_boundary[used]
    if smoothing:
        pts = taubin(pts, tri, fixed, steps=smoothing)
    floating = solid & ~kept
    dropped = int(np.unique(_pieces(tets, floating)[floating]).size)
    return pts, tri, {"floating_pieces_dropped": dropped}


def _gradients(points: np.ndarray, tets: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Gradient of the linear field ``values`` in each tet."""
    x = points[tets]  # (k, 4, 3)
    jac = x[:, 1:] - x[:, :1]  # rows: edges from corner 0
    dv = values[tets[:, 1:]] - values[tets[:, :1]]
    return np.linalg.solve(jac, dv[:, :, None])[:, :, 0]


def save_topology_stl(
    mesh: Mesh,
    densities: np.ndarray,
    path: Path,
    threshold: float = 0.5,
    thickness: float = 1.0,
    anchors: np.ndarray | None = None,
    smoothing: int = 10,
) -> dict[str, Any]:
    """Write the smooth surface to ``path`` and the element surface next to it
    (``<stem>_elements.stl``), as binary STL; return a short summary.

    ``thickness`` extrudes a 2D mesh and is ignored for a 3D one; ``anchors``
    are the supported nodes of ``mesh`` (see :func:`smooth_surface`).
    """
    densities = np.asarray(densities, dtype=float)
    solid_mesh = extrude(mesh, thickness) if mesh.dim == 2 else mesh
    if mesh.dim == 2 and anchors is not None:
        anchors = np.concatenate([anchors, np.asarray(anchors) + mesh.n_nodes])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    elements_path = path.with_name(f"{path.stem}_elements{path.suffix}")

    points, triangles = solid_surface(solid_mesh, densities, threshold)
    meshio.write_points_cells(str(elements_path), points, [("triangle", triangles)], binary=True)
    solid = densities >= threshold
    element_volume = enclosed_volume(points, triangles) if triangles.size else 0.0

    spoints, stris, facts = smooth_surface(solid_mesh, densities, threshold, anchors, smoothing)
    meshio.write_points_cells(str(path), spoints, [("triangle", stris)], binary=True)
    return {
        "path": path,
        "elements_path": elements_path,
        "threshold": threshold,
        "solid_elements": int(solid.sum()),
        "triangles": int(stris.shape[0]),
        "volume": enclosed_volume(spoints, stris) if stris.size else 0.0,
        "element_triangles": int(triangles.shape[0]),
        "element_volume": element_volume,
        "solid_element_volume": float(solid_mesh.cell_measures()[solid].sum()),
        "smoothing_passes": smoothing,
        **facts,
    }
