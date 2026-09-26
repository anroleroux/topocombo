"""Named regions: where the constraints and loads act, picked in the CAD.

(Solid regions are allowed too, for passive, non-design material only.)

A region is any CadQuery geometry on the part's boundary — a vertex, edges or
faces, taken from the part with a selector (``result.faces("<X")``) or built on
its own (a line across a face, where a load should act).  Nothing here knows
about cantilevers: the part script names its regions, the study refers to them
by name, and the pipeline turns each one into the node set of the mesh nodes
lying on it.

Selection works on mesh nodes, not on mesher entities, so it is the same for a
structured grid, a body-fitted mesh and — later — a tetrahedral one.  A load is
spread over its region by the region's dimension: all on one node (a vertex),
by tributary length (edges) or by tributary area (faces).
"""

from __future__ import annotations

from typing import Any

import numpy as np

import cadquery as cq

from .elements import boundary_facets, element

_KINDS = (cq.Vertex, cq.Edge, cq.Wire, cq.Face, cq.Shell, cq.Solid)


def as_shape(value: Any, name: str) -> cq.Shape:
    """One shape from a region value: a ``cq.Workplane`` selection, a
    ``cq.Shape`` or a list of them.  Solids are accepted for passive
    regions; loads and constraints act on the boundary, which the pipeline
    checks."""
    if isinstance(value, cq.Workplane):
        items = value.vals()
    elif isinstance(value, (list, tuple)):
        items = []
        for v in value:
            items.extend(v.vals() if isinstance(v, cq.Workplane) else [v])
    else:
        items = [value]
    shapes: list[cq.Shape] = []
    for item in items:
        if isinstance(item, cq.Compound):
            shapes.extend(item.Solids() or item.Faces() or item.Edges() or item.Vertices())
        elif isinstance(item, _KINDS):
            shapes.append(item)
        else:
            raise ValueError(
                f"region '{name}' must be vertices, edges, faces or solids, "
                f"not {type(item).__name__}"
            )
    if not shapes:
        raise ValueError(f"region '{name}' selects nothing")
    return shapes[0] if len(shapes) == 1 else cq.Compound.makeCompound(shapes)


def region_dim(shape: cq.Shape) -> int:
    """0 for vertices, 1 for edges, 2 for faces, 3 for solids."""
    if shape.Solids():
        return 3
    if shape.Faces():
        return 2
    if shape.Edges():
        return 1
    return 0


def region_vertices(shape: cq.Shape) -> np.ndarray:
    """(n, 3) corner points of a region — what a body-fitted mesh must contain."""
    return np.array([v.toTuple() for v in shape.Vertices()], dtype=float).reshape(-1, 3)


def region_nodes(nodes: np.ndarray, shape: cq.Shape, tol: float) -> np.ndarray:
    """Indices of the ``nodes`` (2D or 3D coordinates) lying on ``shape``."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    pts = np.column_stack([nodes, np.zeros((nodes.shape[0], 3 - nodes.shape[1]))])
    bb = shape.BoundingBox()
    lo = np.array([bb.xmin, bb.ymin, bb.zmin]) - tol
    hi = np.array([bb.xmax, bb.ymax, bb.zmax]) + tol
    if nodes.shape[1] == 2:  # a 2D mesh lives in z = 0
        lo[2], hi[2] = -np.inf, np.inf
    candidates = np.flatnonzero(np.all((pts >= lo) & (pts <= hi), axis=1))
    picked = []
    for i in candidates:
        vertex = BRepBuilderAPI_MakeVertex(gp_Pnt(*map(float, pts[i]))).Vertex()
        dist = BRepExtrema_DistShapeShape(vertex, shape.wrapped)
        if dist.IsDone() and dist.Value() <= tol:
            picked.append(i)
    return np.asarray(picked, dtype=int)


def node_shares(nodes: np.ndarray, cells: np.ndarray, cell_type: str,
                selected: np.ndarray, dim: int, name: str) -> np.ndarray:
    """Each selected node's share of a total force spread uniformly over the
    region: over facets of the mesh boundary (faces in 3D, edges in 2D) or
    along element edges (a line in 3D).  Each piece takes its length or area,
    split over its nodes by the element's load weights — evenly for linear
    elements, the consistent 1/6-2/3-1/6 along a quadratic edge and all on
    the mid-nodes of a 6-node triangle."""
    selected = np.asarray(selected, dtype=int)
    if selected.size == 0:
        raise ValueError(f"region '{name}' has no mesh nodes")
    if dim == 0 or selected.size == 1:
        return np.full(selected.size, 1.0 / selected.size)
    mesh_dim = nodes.shape[1]
    if dim == 2 and mesh_dim == 2:
        raise ValueError(f"region '{name}' is a face: a 2D part takes loads on edges or vertices")
    inside = np.zeros(nodes.shape[0], dtype=bool)
    inside[selected] = True
    el = element(cell_type)
    if dim == mesh_dim - 1:  # boundary facets: edges of a 2D mesh, faces of a 3D one
        pieces = boundary_facets(el, cells)
        weights = el.facet_load_weights
        corners = el.n_facet_corners
    else:  # a line inside a face or along an edge of a 3D mesh: element edges
        pieces = cells[:, np.array(el.line_pieces)].reshape(-1, len(el.line_pieces[0]))
        _, first = np.unique(np.sort(pieces, axis=1), axis=0, return_index=True)
        pieces = pieces[first]
        weights = np.asarray(el.edge_weights)
        corners = 2
    pieces = pieces[np.all(inside[pieces], axis=1)]
    if pieces.size == 0:
        raise ValueError(f"region '{name}': its nodes span no element edge or face")
    weight = np.zeros(nodes.shape[0])
    for piece in pieces:
        np.add.at(weight, piece, _measure(nodes[piece[:corners]]) * weights)
    shares = weight[selected]
    return shares / shares.sum()


def _measure(points: np.ndarray) -> float:
    """Length of a segment, or area of a triangle / (split) quadrilateral."""
    if len(points) == 2:
        return float(np.linalg.norm(points[1] - points[0]))
    p = np.column_stack([points, np.zeros((len(points), 3 - points.shape[1]))])
    area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
    if len(points) == 4:
        area += 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[3] - p[0]))
    return float(area)


def embed_points(regions: dict[str, cq.Shape], profile: cq.Face, tol: float) -> list[tuple[float, float]]:
    """The (x, y) corners of the regions that a mesh of ``profile`` would not
    have as vertices already — e.g. the ends of a load line across a face.

    Corners are projected onto z = 0 (a 3D mesh is the profile extruded, so a
    line along z comes from a point of the profile); corners off the profile
    are skipped.
    """
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_OUT

    existing = np.array([v.toTuple()[:2] for v in profile.Vertices()], dtype=float).reshape(-1, 2)
    classifier = BRepClass_FaceClassifier()
    out: list[tuple[float, float]] = []
    for shape in regions.values():
        for x, y, _ in region_vertices(shape):
            p = np.array([x, y])
            known = np.vstack([existing, np.array(out).reshape(-1, 2)])
            if known.size and np.min(np.linalg.norm(known - p, axis=1)) <= tol:
                continue
            classifier.Perform(profile.wrapped, gp_Pnt(float(x), float(y), 0.0), tol)
            if classifier.State() != TopAbs_OUT:
                out.append((float(x), float(y)))
    return out


def apply_regions(mesh: Any, regions: dict[str, cq.Shape], tol: float | None = None) -> None:
    """Store each region's nodes as ``mesh.node_sets[name]``."""
    if tol is None:
        tol = 1e-6 * float(np.ptp(mesh.nodes, axis=0).max())
    for name, shape in regions.items():
        mesh.node_sets[name] = region_nodes(mesh.nodes, shape, tol)
