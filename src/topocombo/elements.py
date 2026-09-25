"""Finite elements, described once.

Every element-specific fact the pipeline needs lives in one :class:`Element`
entry of :data:`ELEMENTS`, keyed by its meshio cell type:

* the reference cell — corner coordinates, shape-function derivatives, a
  quadrature rule (points and weights) and its centre;
* its topology — edges, outward-ordered boundary facets and the meshio type
  those facets have in a Gmsh file, and the node permutation that turns an
  inverted cell right way round;
* names — the study-facing ``name`` (``quad4``, ``hex8``) and a label.

The routines here work on any entry and on many elements at once: Jacobians,
shape-function gradients, strain-displacement (B) matrices, element stiffness
and cell measures.  :mod:`topocombo.fea`, :mod:`topocombo.mesh_io`,
:mod:`topocombo.regions`, :mod:`topocombo.topology` and the report read their
element facts from here, so a new element — tetrahedra next — is a new entry
with its shape functions and quadrature, not changes throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Element:
    """One element type: its reference cell, quadrature and topology."""

    name: str  # study-facing: "quad4", "hex8"
    cell_type: str  # meshio / VTK name: "quad", "hexahedron"
    label: str  # for logs and the report
    plural: str  # "quadrilaterals", "hexahedra"
    dim: int
    reference_nodes: np.ndarray  # (n, dim) corner coordinates, Gmsh / VTK order
    shape_derivatives: Callable[[np.ndarray], np.ndarray]  # point -> (n, dim) dN/dxi
    quadrature: tuple[np.ndarray, np.ndarray]  # (points (q, dim), weights (q,))
    centre: np.ndarray  # reference coordinates of the cell centre
    edges: tuple[tuple[int, int], ...]
    facets: tuple[tuple[int, ...], ...]  # boundary facets, right-hand normal outward
    facet_cell_type: str  # meshio type of a facet: "line", "quad", "triangle", "triangle6"
    flip: tuple[int, ...]  # node permutation mirroring the cell
    #: How a uniform load spreads over a facet's nodes (fractions of the facet's
    #: share; the integral of each shape function over the facet): even for
    #: linear facets, all on the mid-nodes of a 6-node triangle.  None: even.
    facet_weights: tuple[float, ...] | None = None
    #: Corner nodes per facet (listed first); they span its area.  None: all.
    facet_corners: int | None = None
    #: Nodes along each edge, ends first then any mid-node, and how a uniform
    #: line load spreads over them.  None: ``edges`` with even weights.
    load_edges: tuple[tuple[int, ...], ...] | None = None
    edge_weights: tuple[float, ...] = (0.5, 0.5)
    order: int = 1  # polynomial order of the shape functions

    @property
    def n_nodes(self) -> int:
        return int(self.reference_nodes.shape[0])

    @property
    def facet_load_weights(self) -> np.ndarray:
        n = len(self.facets[0])
        return np.full(n, 1.0 / n) if self.facet_weights is None else np.asarray(self.facet_weights)

    @property
    def n_facet_corners(self) -> int:
        return self.facet_corners or len(self.facets[0])

    @property
    def line_pieces(self) -> tuple[tuple[int, ...], ...]:
        return self.load_edges or self.edges

    @property
    def n_strains(self) -> int:
        """Voigt strain components: [exx, eyy, gxy] in 2D, six in 3D."""
        return 3 if self.dim == 2 else 6

    @property
    def measure_name(self) -> str:
        return "area" if self.dim == 2 else "volume"


# --------------------------------------------------------------------------
# tensor-product (bi/tri-linear) elements: quad4, hex8
# --------------------------------------------------------------------------
def _tensor_derivatives(reference: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """dN/dxi of ``N_a = prod_i (1 + xi_a,i xi_i) / 2^dim`` for corners ``reference``."""

    def derivatives(point: np.ndarray) -> np.ndarray:
        dim = reference.shape[1]
        factors = 1.0 + reference * np.asarray(point, dtype=float)  # (n, dim)
        out = np.empty_like(reference)
        for i in range(dim):
            others = np.prod(np.delete(factors, i, axis=1), axis=1)
            out[:, i] = reference[:, i] * others
        return out / 2.0**dim

    return derivatives


def gauss_legendre_2(dim: int) -> tuple[np.ndarray, np.ndarray]:
    """The 2-point Gauss rule per coordinate: 2^dim points, unit weights."""
    g = 1.0 / np.sqrt(3.0)
    grids = np.meshgrid(*([[-g, g]] * dim), indexing="ij")
    points = np.column_stack([grid.ravel() for grid in grids])
    return points, np.ones(points.shape[0])


_QUAD_NODES = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
_HEX_NODES = np.array(
    [
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ]
)

QUAD4 = Element(
    name="quad4",
    cell_type="quad",
    label="Q4 quadrilateral",
    plural="quadrilaterals",
    dim=2,
    reference_nodes=_QUAD_NODES,
    shape_derivatives=_tensor_derivatives(_QUAD_NODES),
    quadrature=gauss_legendre_2(2),
    centre=np.zeros(2),
    edges=((0, 1), (1, 2), (2, 3), (3, 0)),
    facets=((0, 1), (1, 2), (2, 3), (3, 0)),  # counter-clockwise: outward on the right
    facet_cell_type="line",
    flip=(3, 2, 1, 0),  # walk the quad backwards
)

HEX8 = Element(
    name="hex8",
    cell_type="hexahedron",
    label="H8 hexahedron",
    plural="hexahedra",
    dim=3,
    reference_nodes=_HEX_NODES,
    shape_derivatives=_tensor_derivatives(_HEX_NODES),
    quadrature=gauss_legendre_2(3),
    centre=np.zeros(3),
    edges=(
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ),
    facets=((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)),
    facet_cell_type="quad",
    flip=(0, 3, 2, 1, 4, 7, 6, 5),  # walk both faces backwards
)

# --------------------------------------------------------------------------
# simplex elements: tet4 (linear), tet10 (quadratic)
# --------------------------------------------------------------------------
_TET_CORNERS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
#: Mid-edge nodes of the quadratic tet in VTK / meshio order (Gmsh's own order
#: swaps the last two; meshio reorders on reading).
_TET10_MIDS = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
_TET_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))
#: Barycentric coordinates L = (1 - x - y - z, x, y, z) and their gradients.
_DL = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _barycentric(point: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(point, dtype=float)
    return np.array([1.0 - x - y - z, x, y, z])


def _tet4_derivatives(point: np.ndarray) -> np.ndarray:
    return _DL.copy()


def _tet10_derivatives(point: np.ndarray) -> np.ndarray:
    """Corners ``N_i = L_i (2 L_i - 1)``, mid-nodes ``N_ij = 4 L_i L_j``."""
    L = _barycentric(point)
    out = np.empty((10, 3))
    for i in range(4):
        out[i] = (4.0 * L[i] - 1.0) * _DL[i]
    for k, (i, j) in enumerate(_TET10_MIDS):
        out[4 + k] = 4.0 * (L[i] * _DL[j] + L[j] * _DL[i])
    return out


def _tet_rule_4() -> tuple[np.ndarray, np.ndarray]:
    """The 4-point, degree-2 rule on the unit tetrahedron (weights sum to 1/6)."""
    a, b = 0.5854101966249685, 0.1381966011250105
    points = np.array([[b, b, b], [a, b, b], [b, a, b], [b, b, a]])
    return points, np.full(4, 1.0 / 24.0)


#: Faces opposite vertex 3, 2, 1, 0, corners ordered with the normal outward.
_TET_FACES = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))


def _tet10_face(face: tuple[int, int, int]) -> tuple[int, ...]:
    """A face's three corners, then its mid-nodes on (a, b), (b, c), (c, a)."""
    a, b, c = face
    mid = {frozenset(p): 4 + k for k, p in enumerate(_TET10_MIDS)}
    return (a, b, c, mid[frozenset((a, b))], mid[frozenset((b, c))], mid[frozenset((c, a))])


TET4 = Element(
    name="tet4",
    cell_type="tetra",
    label="T4 tetrahedron",
    plural="tetrahedra",
    dim=3,
    reference_nodes=_TET_CORNERS,
    shape_derivatives=_tet4_derivatives,
    quadrature=(np.array([[0.25, 0.25, 0.25]]), np.array([1.0 / 6.0])),
    centre=np.full(3, 0.25),
    edges=_TET_EDGES,
    facets=_TET_FACES,
    facet_cell_type="triangle",
    flip=(0, 2, 1, 3),
)

TET10 = Element(
    name="tet10",
    cell_type="tetra10",
    label="T10 tetrahedron",
    plural="quadratic tetrahedra",
    dim=3,
    reference_nodes=np.vstack(
        [_TET_CORNERS] + [(_TET_CORNERS[i] + _TET_CORNERS[j]) / 2.0 for i, j in _TET10_MIDS]
    ),
    shape_derivatives=_tet10_derivatives,
    quadrature=_tet_rule_4(),
    centre=np.full(3, 0.25),
    edges=_TET_EDGES,
    facets=tuple(_tet10_face(f) for f in _TET_FACES),
    facet_cell_type="triangle6",
    # swapping corners 1 and 2 also swaps mid-nodes (0,1)<->(0,2) and (1,3)<->(2,3)
    flip=(0, 2, 1, 3, 6, 5, 4, 7, 9, 8),
    facet_weights=(0.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    facet_corners=3,
    load_edges=tuple((i, j, 4 + k) for k, (i, j) in enumerate(_TET10_MIDS)),
    edge_weights=(1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    order=2,
)

#: Every supported element, by meshio cell type.
ELEMENTS: dict[str, Element] = {e.cell_type: e for e in (QUAD4, HEX8, TET4, TET10)}

#: The same, by study-facing name.
BY_NAME: dict[str, Element] = {e.name: e for e in ELEMENTS.values()}


def element(cell_type: str) -> Element:
    """The element for a meshio cell type."""
    try:
        return ELEMENTS[cell_type]
    except KeyError:
        raise ValueError(
            f"unsupported cell type {cell_type!r}; supported: {', '.join(ELEMENTS)}"
        ) from None


# --------------------------------------------------------------------------
# element routines, vectorised over elements
# --------------------------------------------------------------------------
def jacobians(el: Element, coords: np.ndarray, point: np.ndarray) -> np.ndarray:
    """J of the isoparametric map of each element at reference ``point``,
    shape (m, dim, dim), with ``J[i, j] = d x_j / d xi_i``; ``coords`` is (m, n, dim)."""
    dn = el.shape_derivatives(np.asarray(point, dtype=float))  # (n, dim)
    return np.einsum("ni,mnj->mij", dn, coords)


def gradients(el: Element, coords: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dN/dx of shape (m, dim, n), det J of shape (m,)) at reference ``point``.

    Raises on a non-positive Jacobian: an inverted or degenerate element.
    """
    dn = el.shape_derivatives(np.asarray(point, dtype=float))  # (n, dim)
    jac = np.einsum("ni,mnj->mij", dn, coords)
    det = np.linalg.det(jac)
    if np.any(det <= 0.0):
        worst = float(det.min())
        raise ValueError(f"non-positive Jacobian ({worst:.3e}); element is inverted or degenerate")
    grads = np.linalg.solve(jac, np.broadcast_to(dn.T, (coords.shape[0], *dn.T.shape)))
    return grads, det


def strain_displacement(el: Element, coords: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B of shape (m, n_strains, dim * n), det J (m,)) at reference ``point``.

    Strains in Voigt order with engineering shears: [exx, eyy, gxy] in 2D,
    [exx, eyy, ezz, gxy, gyz, gzx] in 3D; DOFs node-major (x, y[, z] per node).
    """
    g, det = gradients(el, coords, point)  # (m, dim, n)
    m, dim, n = g.shape
    b = np.zeros((m, el.n_strains, dim * n))
    if dim == 2:
        b[:, 0, 0::2] = g[:, 0]
        b[:, 1, 1::2] = g[:, 1]
        b[:, 2, 0::2] = g[:, 1]
        b[:, 2, 1::2] = g[:, 0]
    else:
        b[:, 0, 0::3] = g[:, 0]
        b[:, 1, 1::3] = g[:, 1]
        b[:, 2, 2::3] = g[:, 2]
        b[:, 3, 0::3], b[:, 3, 1::3] = g[:, 1], g[:, 0]
        b[:, 4, 1::3], b[:, 4, 2::3] = g[:, 2], g[:, 1]
        b[:, 5, 2::3], b[:, 5, 0::3] = g[:, 0], g[:, 2]
    return b, det


def stiffness(el: Element, coords: np.ndarray, d: np.ndarray, thickness: float = 1.0,
              chunk: int = 4096) -> np.ndarray:
    """Element stiffness matrices, shape (m, dim * n, dim * n), by the
    element's quadrature.  ``thickness`` multiplies 2D (plane-stress) elements
    and is ignored in 3D.  Works in chunks of elements to bound memory."""
    coords = np.asarray(coords, dtype=float)
    factor = float(thickness) if el.dim == 2 else 1.0
    size = el.dim * el.n_nodes
    out = np.zeros((coords.shape[0], size, size))
    points, weights = el.quadrature
    for start in range(0, coords.shape[0], chunk):
        part = coords[start:start + chunk]
        ke = out[start:start + chunk]
        for point, weight in zip(points, weights):
            b, det = strain_displacement(el, part, point)
            ke += (factor * weight * det)[:, None, None] * (b.transpose(0, 2, 1) @ (d @ b))
    return out


def measures(el: Element, nodes: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Signed area (2D) or volume (3D) of every cell by the element's
    quadrature; negative for an inverted cell."""
    coords = nodes[cells]
    points, weights = el.quadrature
    return sum(w * np.linalg.det(jacobians(el, coords, p)) for p, w in zip(points, weights))


def boundary_facets(el: Element, cells: np.ndarray) -> np.ndarray:
    """Facets bounding the union of ``cells``, as node rows in the element's
    outward order, shape (k, nodes per facet): the facets only one cell has."""
    facets = cells[:, np.array(el.facets)].reshape(-1, len(el.facets[0]))
    keys = np.sort(facets, axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return facets[counts[inverse.ravel()] == 1]


def triangulate(facets: np.ndarray) -> np.ndarray:
    """Surface facets (triangles, quadrilaterals or 6-node triangles) as
    triangles, keeping their orientation."""
    if facets.shape[1] == 3:
        return facets
    if facets.shape[1] == 4:
        return np.vstack([facets[:, [0, 1, 2]], facets[:, [0, 2, 3]]])
    if facets.shape[1] == 6:  # corners a b c, mids ab bc ca: four sub-triangles
        return np.vstack([facets[:, [0, 3, 5]], facets[:, [3, 1, 4]],
                          facets[:, [5, 4, 2]], facets[:, [3, 4, 5]]])
    raise ValueError(f"cannot triangulate {facets.shape[1]}-node facets")
