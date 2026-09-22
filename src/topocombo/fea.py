"""Minimal 2D plane-stress FEA for bilinear quadrilaterals.

Custom rather than external so the optimizer gets direct, in-memory access to
element stiffness matrices and the displacement field — the per-element
compliance ``u_e^T k_e u_e`` that comes out of :func:`solve` is exactly the
quantity SIMP sensitivities are built from.

Conventions: node ``n`` owns DOFs ``2n`` (x) and ``2n+1`` (y); units are
N and mm throughout, so stresses come out in MPa.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mesh_io import QuadMesh

#: 2x2 Gauss-Legendre points and weights on the reference square.
_GAUSS = [(-1 / np.sqrt(3), -1 / np.sqrt(3)), (1 / np.sqrt(3), -1 / np.sqrt(3)),
          (1 / np.sqrt(3), 1 / np.sqrt(3)), (-1 / np.sqrt(3), 1 / np.sqrt(3))]
_NODE_XI = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


@dataclass(frozen=True)
class Material:
    """Linear-elastic, isotropic material (plane stress)."""

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

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoadCase:
    """A single vertical point load, applied at one node."""

    node: int
    fy: float = -1000.0  # N, downwards
    fx: float = 0.0

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.fx, self.fy))

    def as_dict(self) -> dict[str, Any]:
        return {"node": self.node, "fx": self.fx, "fy": self.fy, "magnitude": self.magnitude}


@dataclass
class FEResult:
    """Displacements and the derived quantities the optimizer and report need."""

    u: np.ndarray  # (2 * n_nodes,) displacements
    compliance: float  # F . U
    element_compliance: np.ndarray  # (n_elements,) u_e^T k_e u_e
    von_mises: np.ndarray  # (n_elements,) centroid stress, MPa
    reactions: np.ndarray  # (2 * n_nodes,) K U - F, non-zero on constrained DOFs
    equilibrium_residual: float  # ||K U - F|| on the free DOFs
    n_free_dofs: int

    @property
    def displacement_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.u.reshape(-1, 2), axis=1)

    def max_deflection(self) -> float:
        """Largest downward (negative-y) displacement, as a positive number."""
        return float(-self.u[1::2].min())


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


def element_dofs(quads: np.ndarray) -> np.ndarray:
    """(n_elements, 8) DOF indices for every element."""
    dofs = np.empty((quads.shape[0], 8), dtype=int)
    dofs[:, 0::2] = 2 * quads
    dofs[:, 1::2] = 2 * quads + 1
    return dofs


def element_stiffnesses(mesh: QuadMesh, material: Material, thickness: float) -> np.ndarray:
    """Stiffness matrix of every element at full density, shape (n_elements, 8, 8)."""
    d = material.constitutive_matrix()
    return np.array([element_stiffness(mesh.nodes[quad], d, thickness) for quad in mesh.quads])


def simp_scaling(
    densities: np.ndarray, penal: float = 3.0, e_min: float = 1e-9
) -> np.ndarray:
    """SIMP interpolation ``E(x) = E_min + x^p (1 - E_min)`` as a stiffness factor."""
    return e_min + np.asarray(densities, dtype=float) ** penal * (1.0 - e_min)


def assemble_stiffness(
    mesh: QuadMesh,
    ke_all: np.ndarray,
    scale: np.ndarray | None = None,
) -> sp.csc_matrix:
    """Assemble the global stiffness matrix from per-element matrices."""
    dofs = element_dofs(mesh.quads)
    values = ke_all if scale is None else ke_all * np.asarray(scale)[:, None, None]
    rows = np.repeat(dofs, 8, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, 8)).reshape(-1)
    n_dof = 2 * mesh.n_nodes
    return sp.coo_matrix(
        (values.reshape(-1), (rows, cols)), shape=(n_dof, n_dof)
    ).tocsc()


def load_vector(mesh: QuadMesh, load: LoadCase) -> np.ndarray:
    f = np.zeros(2 * mesh.n_nodes)
    f[2 * load.node] = load.fx
    f[2 * load.node + 1] = load.fy
    return f


def fixed_dofs(mesh: QuadMesh, node_set: str) -> np.ndarray:
    """Both DOFs of every node in ``node_set`` (a fully clamped edge)."""
    nodes = mesh.node_sets[node_set]
    return np.sort(np.concatenate([2 * nodes, 2 * nodes + 1]))


def centroid_von_mises(
    mesh: QuadMesh, u: np.ndarray, material: Material, scale: np.ndarray | None = None
) -> np.ndarray:
    """Von Mises stress at each element centroid (plane stress), MPa."""
    d = material.constitutive_matrix()
    dofs = element_dofs(mesh.quads)
    out = np.empty(mesh.n_elements)
    for e, quad in enumerate(mesh.quads):
        b, _ = strain_displacement(mesh.nodes[quad], 0.0, 0.0)
        sx, sy, txy = d @ (b @ u[dofs[e]])
        if scale is not None:
            sx, sy, txy = np.array([sx, sy, txy]) * scale[e]
        out[e] = np.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * txy**2)
    return out


def solve(
    mesh: QuadMesh,
    material: Material,
    thickness: float,
    load: LoadCase,
    fixed_node_set: str,
    densities: np.ndarray | None = None,
    penal: float = 3.0,
    ke_all: np.ndarray | None = None,
) -> FEResult:
    """Direct plane-stress solve; ``densities`` (if given) applies SIMP scaling."""
    if ke_all is None:
        ke_all = element_stiffnesses(mesh, material, thickness)
    scale = None if densities is None else simp_scaling(densities, penal)

    k = assemble_stiffness(mesh, ke_all, scale)
    f = load_vector(mesh, load)

    constrained = fixed_dofs(mesh, fixed_node_set)
    free = np.setdiff1d(np.arange(2 * mesh.n_nodes), constrained, assume_unique=False)

    u = np.zeros(2 * mesh.n_nodes)
    u[free] = spla.spsolve(k[free][:, free].tocsc(), f[free])

    residual = k @ u - f
    dofs = element_dofs(mesh.quads)
    ue = u[dofs]  # (n_elements, 8)
    element_compliance = np.einsum("ei,eij,ej->e", ue, ke_all, ue)
    if scale is not None:
        element_compliance = element_compliance * scale

    return FEResult(
        u=u,
        compliance=float(f @ u),
        element_compliance=element_compliance,
        von_mises=centroid_von_mises(mesh, u, material, scale),
        reactions=residual,
        equilibrium_residual=float(np.linalg.norm(residual[free])),
        n_free_dofs=int(free.size),
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
