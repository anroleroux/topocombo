"""Minimal linear-elastic FEA: Q4 plane-stress quads and H8 solid hexahedra.

Custom rather than external so the optimizer gets direct, in-memory access to
element stiffness matrices and the displacement field — the per-element
compliance ``u_e^T k_e u_e`` that comes out of :func:`solve` is exactly the
quantity SIMP sensitivities are built from.

Conventions: with ``d = mesh.dofs_per_node`` components per node, node ``n``
owns DOFs ``d*n`` (x), ``d*n + 1`` (y) and, in 3D, ``d*n + 2`` (z); units are
N and mm throughout, so stresses come out in MPa.  Element routines dispatch on
``mesh.cell_type``: bilinear quads (plane stress, with an out-of-plane
thickness) and trilinear hexahedra (full 3D, where the width is modelled and
``thickness`` is not used).  Both integrate with the 2-point Gauss rule.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh_io import Mesh, gauss_points, shape_derivatives

#: 2x2 Gauss-Legendre points and weights on the reference square.
_GAUSS = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
          (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]
_NODE_XI = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


@dataclass(frozen=True)
class Material:
    """Linear-elastic, isotropic material."""

    youngs_modulus: float = 210_000.0  # MPa
    poisson_ratio: float = 0.3

    def __post_init__(self) -> None:
        if self.youngs_modulus <= 0:
            raise ValueError("youngs_modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("poisson_ratio must lie in (-1, 0.5)")

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def constitutive_matrix(self) -> np.ndarray:
        """Plane-stress D matrix relating [sxx, syy, sxy] to [exx, eyy, gxy]."""
        e, nu = self.youngs_modulus, self.poisson_ratio
        return (e / (1.0 - nu**2)) * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )

    def constitutive_matrix_3d(self) -> np.ndarray:
        """3D D matrix relating [sxx, syy, szz, sxy, syz, szx] to engineering strains."""
        e, nu = self.youngs_modulus, self.poisson_ratio
        lam = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = self.shear_modulus
        d = np.zeros((6, 6))
        d[:3, :3] = lam
        d[:3, :3] += 2.0 * mu * np.eye(3)
        d[3:, 3:] = mu * np.eye(3)
        return d

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadCase:
    """A point load at one node, or one total load shared over several nodes.

    ``node`` is a single index or a tuple of indices; ``shares`` gives each
    node's fraction of the total (even split if omitted).  A line load is built
    with :meth:`along_line`, which uses consistent (tributary-length) shares.
    """

    node: int | tuple[int, ...]
    fy: float = -1000.0  # N, downwards — the total over all nodes
    fx: float = 0.0
    fz: float = 0.0  # only meaningful on a 3D mesh
    shares: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        n = self.nodes.size
        if n == 0:
            raise ValueError("a load needs at least one node")
        if self.shares is not None:
            if len(self.shares) != n:
                raise ValueError(f"{len(self.shares)} shares for {n} nodes")
            if abs(sum(self.shares) - 1.0) > 1e-9 or min(self.shares) < 0.0:
                raise ValueError("load shares must be non-negative and sum to 1")

    @classmethod
    def along_line(
        cls, mesh: Mesh, nodes: np.ndarray, fy: float = -1000.0, fx: float = 0.0, fz: float = 0.0
    ) -> "LoadCase":
        """A uniform line load through ``nodes`` (ordered along the line).

        Each node takes half of each segment it touches — the consistent nodal
        load for linear elements — so end nodes carry half an interior share.
        """
        nodes = np.asarray(nodes, dtype=int)
        if nodes.size == 1:
            return cls(node=int(nodes[0]), fy=fy, fx=fx, fz=fz)
        seg = np.linalg.norm(np.diff(mesh.nodes[nodes], axis=0), axis=1)
        tributary = np.zeros(nodes.size)
        tributary[:-1] += 0.5 * seg
        tributary[1:] += 0.5 * seg
        shares = tuple(float(v) for v in tributary / tributary.sum())
        return cls(node=tuple(int(n) for n in nodes), fy=fy, fx=fx, fz=fz, shares=shares)

    @property
    def nodes(self) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.node, dtype=int))

    def node_shares(self) -> np.ndarray:
        n = self.nodes.size
        return np.full(n, 1.0 / n) if self.shares is None else np.asarray(self.shares)

    def components(self, dim: int) -> np.ndarray:
        """The force vector with one entry per spatial dimension."""
        if dim == 2 and self.fz != 0.0:
            raise ValueError("a 2D mesh cannot carry an out-of-plane (fz) load")
        return np.array([self.fx, self.fy, self.fz][:dim])

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm([self.fx, self.fy, self.fz]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "node": self.node if isinstance(self.node, int) else list(self.node),
            "shares": self.node_shares().tolist(),
            "fx": self.fx,
            "fy": self.fy,
            "fz": self.fz,
            "magnitude": self.magnitude,
        }


@dataclass
class FEResult:
    """Displacements and the derived quantities the optimizer and report need."""

    u: np.ndarray  # (dofs_per_node * n_nodes,) displacements
    compliance: float  # F . U
    element_compliance: np.ndarray  # (n_elements,) u_e^T k_e u_e, density-scaled
    element_compliance_unscaled: np.ndarray  # (n_elements,) u_e^T k0_e u_e, the SIMP kernel
    von_mises: np.ndarray  # (n_elements,) centroid stress, MPa
    reactions: np.ndarray  # (dofs_per_node * n_nodes,) K U - F, non-zero on constrained DOFs
    equilibrium_residual: float  # ||K U - F|| on the free DOFs
    n_free_dofs: int
    dofs_per_node: int = 2

    @property
    def displacement_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.u.reshape(-1, self.dofs_per_node), axis=1)

    def component(self, axis: int, field: str = "u") -> np.ndarray:
        """One nodal component (0 = x, 1 = y, 2 = z) of ``u`` or ``reactions``."""
        return getattr(self, field)[axis :: self.dofs_per_node]

    def max_deflection(self) -> float:
        """Largest downward (negative-y) displacement, as a positive number."""
        return float(-self.component(1).min())


def _shape_gradients(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """dN/dx, dN/dy (2x4) and det(J) of the isoparametric map at (xi, eta)."""
    dn_dxi = 0.25 * np.column_stack(
        [
            _NODE_XI[:, 0] * (1.0 + _NODE_XI[:, 1] * eta),
            _NODE_XI[:, 1] * (1.0 + _NODE_XI[:, 0] * xi),
        ]
    )  # (4, 2)
    jac = dn_dxi.T @ coords  # (2, 2)
    det = float(np.linalg.det(jac))
    if det <= 0.0:
        raise ValueError(f"non-positive Jacobian ({det:.3e}); element is inverted or degenerate")
    return np.linalg.solve(jac, dn_dxi.T), det


def strain_displacement(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """B matrix (3x8) and det(J) for a Q4 element at (xi, eta)."""
    grads, det = _shape_gradients(coords, xi, eta)
    b = np.zeros((3, 8))
    b[0, 0::2] = grads[0]
    b[1, 1::2] = grads[1]
    b[2, 0::2] = grads[1]
    b[2, 1::2] = grads[0]
    return b, det


def element_stiffness(coords: np.ndarray, d: np.ndarray, thickness: float) -> np.ndarray:
    """Q4 plane-stress element stiffness (8x8) by 2x2 Gauss quadrature."""
    ke = np.zeros((8, 8))
    for xi, eta in _GAUSS:
        b, det = strain_displacement(coords, xi, eta)
        ke += thickness * det * (b.T @ d @ b)  # unit Gauss weights
    return ke


def hex_strain_displacement(coords: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, float]:
    """B matrix (6x24) and det(J) for an H8 element at reference ``point``.

    Strain order is [exx, eyy, ezz, gxy, gyz, gzx] (engineering shears), matching
    :meth:`Material.constitutive_matrix_3d`.
    """
    dn_dxi = shape_derivatives("hexahedron", point)  # (8, 3)
    jac = dn_dxi.T @ coords  # (3, 3)
    det = float(np.linalg.det(jac))
    if det <= 0.0:
        raise ValueError(f"non-positive Jacobian ({det:.3e}); element is inverted or degenerate")
    g = np.linalg.solve(jac, dn_dxi.T)  # (3, 8): dN/dx, dN/dy, dN/dz
    b = np.zeros((6, 24))
    b[0, 0::3] = g[0]
    b[1, 1::3] = g[1]
    b[2, 2::3] = g[2]
    b[3, 0::3], b[3, 1::3] = g[1], g[0]
    b[4, 1::3], b[4, 2::3] = g[2], g[1]
    b[5, 2::3], b[5, 0::3] = g[0], g[2]
    return b, det


def hex_element_stiffness(coords: np.ndarray, d: np.ndarray) -> np.ndarray:
    """H8 solid element stiffness (24x24) by 2x2x2 Gauss quadrature."""
    ke = np.zeros((24, 24))
    for point in gauss_points(3):
        b, det = hex_strain_displacement(coords, point)
        ke += det * (b.T @ d @ b)  # unit Gauss weights
    return ke


def element_dofs(cells: np.ndarray, dofs_per_node: int = 2) -> np.ndarray:
    """(n_elements, nodes_per_cell * dofs_per_node) DOF indices, node-major."""
    comps = np.arange(dofs_per_node)
    return (dofs_per_node * cells[:, :, None] + comps).reshape(cells.shape[0], -1)


def node_dofs(nodes: np.ndarray, dofs_per_node: int = 2) -> np.ndarray:
    """Every DOF of the given nodes, sorted."""
    comps = np.arange(dofs_per_node)
    return np.sort((dofs_per_node * np.asarray(nodes)[:, None] + comps).reshape(-1))


def _element_function(mesh: Mesh, material: Material, thickness: float):
    """The stiffness routine for the mesh's cell type, bound to its material."""
    if mesh.cell_type == "quad":
        d = material.constitutive_matrix()
        return lambda coords: element_stiffness(coords, d, thickness)
    if mesh.cell_type == "hexahedron":
        d = material.constitutive_matrix_3d()
        return lambda coords: hex_element_stiffness(coords, d)
    raise NotImplementedError(f"no element formulation for {mesh.cell_type!r} cells")  # pragma: no cover


def element_stiffnesses(mesh: Mesh, material: Material, thickness: float) -> np.ndarray:
    """Stiffness matrix of every element at full density, shape (n_elements, k, k).

    Stiffness depends only on an element's shape, not its position, so on a
    uniform structured grid one matrix is computed and shared (a read-only
    broadcast view) instead of repeating the integration per element.
    ``thickness`` applies to plane-stress quads only.
    """
    element = _element_function(mesh, material, thickness)
    xyz = mesh.nodes[mesh.cells]  # (m, k, dim)
    rel = xyz - xyz[:, :1]
    size = float(np.ptp(mesh.nodes, axis=0).max()) or 1.0
    if np.allclose(rel, rel[:1], rtol=0.0, atol=1e-12 * size):
        ke = element(rel[0])
        return np.broadcast_to(ke, (mesh.n_elements, *ke.shape))
    return np.array([element(coords) for coords in rel])


def simp_scaling(
    densities: np.ndarray, penal: float = 3.0, e_min: float = 1e-9
) -> np.ndarray:
    """SIMP interpolation ``E(x) = E_min + x^p (1 - E_min)`` as a stiffness factor."""
    return e_min + np.asarray(densities, dtype=float) ** penal * (1.0 - e_min)


def assemble_stiffness(
    mesh: Mesh,
    ke_all: np.ndarray,
    scale: np.ndarray | None = None,
) -> sp.csc_matrix:
    """Assemble the global stiffness matrix from per-element matrices."""
    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    n_edof = dofs.shape[1]
    values = ke_all if scale is None else ke_all * np.asarray(scale)[:, None, None]
    rows = np.repeat(dofs, n_edof, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, n_edof)).reshape(-1)
    n_dof = mesh.n_dofs
    return sp.coo_matrix(
        (values.reshape(-1), (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()


def load_vector(mesh: Mesh, load: LoadCase) -> np.ndarray:
    f = np.zeros(mesh.n_dofs)
    per_node = np.outer(load.node_shares(), load.components(mesh.dim))  # (n, dim)
    dofs = element_dofs(load.nodes[None, :], mesh.dofs_per_node)[0]  # node-major
    np.add.at(f, dofs, per_node.reshape(-1))
    return f


def fixed_dofs(mesh: Mesh, node_set: str) -> np.ndarray:
    """Every DOF of every node in ``node_set`` (a fully clamped boundary)."""
    return node_dofs(mesh.node_sets[node_set], mesh.dofs_per_node)


def _von_mises(stress: np.ndarray) -> np.ndarray:
    """Von Mises stress from rows of [sxx, syy, sxy] (plane stress) or 3D Voigt stress."""
    if stress.shape[1] == 3:
        sx, sy, txy = stress.T
        return np.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2)
    sx, sy, sz, txy, tyz, tzx = stress.T
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + tzx**2)
    )


def centroid_von_mises(
    mesh: Mesh, u: np.ndarray, material: Material, scale: np.ndarray | None = None
) -> np.ndarray:
    """Von Mises stress at each element centroid, MPa (plane stress for quads)."""
    if mesh.cell_type == "quad":
        d = material.constitutive_matrix()

        def b_at_centre(coords: np.ndarray) -> np.ndarray:
            return strain_displacement(coords, 0.0, 0.0)[0]
    elif mesh.cell_type == "hexahedron":
        d = material.constitutive_matrix_3d()

        def b_at_centre(coords: np.ndarray) -> np.ndarray:
            return hex_strain_displacement(coords, np.zeros(3))[0]
    else:  # pragma: no cover
        raise NotImplementedError(f"no element formulation for {mesh.cell_type!r} cells")

    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    b_all = np.array([b_at_centre(mesh.nodes[cell]) for cell in mesh.cells])
    strain = np.einsum("eij,ej->ei", b_all, u[dofs])
    stress = strain @ d.T
    if scale is not None:
        stress = stress * np.asarray(scale)[:, None]
    return _von_mises(stress)


def solve(
    mesh: Mesh,
    material: Material,
    thickness: float,
    load: LoadCase,
    fixed_node_set: str,
    densities: np.ndarray | None = None,
    penal: float = 3.0,
    ke_all: np.ndarray | None = None,
) -> FEResult:
    """Direct linear-elastic solve; ``densities`` (if given) applies SIMP scaling.

    Plane stress on a quad mesh (with ``thickness``), full 3D on a hex mesh.
    """
    if ke_all is None:
        ke_all = element_stiffnesses(mesh, material, thickness)
    scale = None if densities is None else simp_scaling(densities, penal)

    k = assemble_stiffness(mesh, ke_all, scale)
    f = load_vector(mesh, load)

    constrained = fixed_dofs(mesh, fixed_node_set)
    free = np.setdiff1d(np.arange(mesh.n_dofs), constrained, assume_unique=False)

    u = np.zeros(mesh.n_dofs)
    u[free] = spla.spsolve(k[free][:, free].tocsc(), f[free])

    residual = k @ u - f
    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    ue = u[dofs]  # (n_elements, nodes_per_cell * dofs_per_node)
    unscaled = np.einsum("ei,eij,ej->e", ue, ke_all, ue)
    element_compliance = unscaled if scale is None else unscaled * scale

    return FEResult(
        u=u,
        compliance=float(f @ u),
        element_compliance=element_compliance,
        element_compliance_unscaled=unscaled,
        von_mises=centroid_von_mises(mesh, u, material, scale),
        reactions=residual,
        equilibrium_residual=float(np.linalg.norm(residual[free])),
        n_free_dofs=int(free.size),
        dofs_per_node=mesh.dofs_per_node,
    )


def timoshenko_tip_deflection(
    length: float, height: float, thickness: float, material: Material, load: float
) -> dict[str, float]:
    """Cantilever tip deflection under an end load: bending + shear (k = 5/6)."""
    i = thickness * height**3 / 12.0
    area = thickness * height
    bending = abs(load) * length**3 / (3.0 * material.youngs_modulus * i)
    shear = abs(load) * length / ((5.0 / 6.0) * material.shear_modulus * area)
    return {"bending": bending, "shear": shear, "total": bending + shear}
