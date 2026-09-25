"""Minimal linear-elastic FEA: Q4 plane-stress quads and H8 solid hexahedra.

Custom rather than external so the optimizer gets direct, in-memory access to
element stiffness matrices and the displacement field — the per-element
compliance ``u_e^T k_e u_e`` that comes out of :func:`solve` is exactly the
quantity SIMP sensitivities are built from.

Conventions: with ``d = mesh.dofs_per_node`` components per node, node ``n``
owns DOFs ``d*n`` (x), ``d*n + 1`` (y) and, in 3D, ``d*n + 2`` (z); units are
N and mm throughout, so stresses come out in MPa.  Everything element-specific
comes from the mesh's :class:`~topocombo.elements.Element` (shape functions,
quadrature, B matrices); this module only picks the constitutive law by
dimension — plane stress in 2D (with an out-of-plane ``thickness``), full 3D
elasticity in 3D (where the width is modelled and ``thickness`` is not used).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from . import elements
from .elements import HEX8, QUAD4
from .mesh_io import Mesh


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

    def stiffness_matrix(self, dim: int) -> np.ndarray:
        """The D matrix for a ``dim``-dimensional analysis: plane stress in 2D."""
        return self.constitutive_matrix() if dim == 2 else self.constitutive_matrix_3d()

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
    solver: str = "direct"  # what solved it: "direct" or "amg-cg"
    solver_iterations: int = 0  # CG iterations (0 for a direct solve)

    @property
    def displacement_magnitude(self) -> np.ndarray:
        return np.linalg.norm(self.u.reshape(-1, self.dofs_per_node), axis=1)

    def component(self, axis: int, field: str = "u") -> np.ndarray:
        """One nodal component (0 = x, 1 = y, 2 = z) of ``u`` or ``reactions``."""
        return getattr(self, field)[axis :: self.dofs_per_node]

    def max_deflection(self) -> float:
        """Largest downward (negative-y) displacement, as a positive number."""
        return float(-self.component(1).min())


def strain_displacement(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    """B matrix (3x8) and det(J) for one Q4 element at (xi, eta)."""
    b, det = elements.strain_displacement(QUAD4, np.asarray(coords)[None], np.array([xi, eta]))
    return b[0], float(det[0])


def element_stiffness(coords: np.ndarray, d: np.ndarray, thickness: float) -> np.ndarray:
    """Q4 plane-stress element stiffness (8x8) of one element."""
    return elements.stiffness(QUAD4, np.asarray(coords)[None], d, thickness)[0]


def hex_strain_displacement(coords: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, float]:
    """B matrix (6x24) and det(J) for one H8 element at reference ``point``.

    Strain order is [exx, eyy, ezz, gxy, gyz, gzx] (engineering shears), matching
    :meth:`Material.constitutive_matrix_3d`.
    """
    b, det = elements.strain_displacement(HEX8, np.asarray(coords)[None], np.asarray(point))
    return b[0], float(det[0])


def hex_element_stiffness(coords: np.ndarray, d: np.ndarray) -> np.ndarray:
    """H8 solid element stiffness (24x24) of one element."""
    return elements.stiffness(HEX8, np.asarray(coords)[None], d)[0]


def element_dofs(cells: np.ndarray, dofs_per_node: int = 2) -> np.ndarray:
    """(n_elements, nodes_per_cell * dofs_per_node) DOF indices, node-major."""
    comps = np.arange(dofs_per_node)
    return (dofs_per_node * cells[:, :, None] + comps).reshape(cells.shape[0], -1)


def node_dofs(nodes: np.ndarray, dofs_per_node: int = 2) -> np.ndarray:
    """Every DOF of the given nodes, sorted."""
    comps = np.arange(dofs_per_node)
    return np.sort((dofs_per_node * np.asarray(nodes)[:, None] + comps).reshape(-1))


def element_stiffnesses(mesh: Mesh, material: Material, thickness: float) -> np.ndarray:
    """Stiffness matrix of every element at full density, shape (n_elements, k, k).

    Stiffness depends only on an element's shape, not its position, so on a
    uniform structured grid one matrix is computed and shared (a read-only
    broadcast view) instead of repeating the integration per element.
    ``thickness`` applies to plane-stress quads only.
    """
    el = mesh.element
    d = material.stiffness_matrix(el.dim)
    xyz = mesh.nodes[mesh.cells]  # (m, k, dim)
    rel = xyz - xyz[:, :1]
    size = float(np.ptp(mesh.nodes, axis=0).max()) or 1.0
    if np.allclose(rel, rel[:1], rtol=0.0, atol=1e-12 * size):
        ke = elements.stiffness(el, rel[:1], d, thickness)[0]
        return np.broadcast_to(ke, (mesh.n_elements, *ke.shape))
    return elements.stiffness(el, rel, d, thickness)


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
    """Assemble the global stiffness matrix from per-element matrices.

    The sparsity pattern and the map from element entries to it depend only on
    the mesh, so they are built once and cached on it; each assembly after that
    — one per SIMP iteration — is a weighted ``bincount``.
    """
    indptr, indices, slot = _assembly_plan(mesh)
    values = ke_all if scale is None else ke_all * np.asarray(scale)[:, None, None]
    data = np.bincount(slot, weights=values.reshape(-1), minlength=indices.size)
    n_dof = mesh.n_dofs
    return sp.csc_matrix((data, indices, indptr), shape=(n_dof, n_dof))


def _assembly_plan(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(indptr, row indices, slot of every element-matrix entry) of the global
    matrix in CSC form, cached on the mesh."""
    plan = getattr(mesh, "_assembly_plan", None)
    key = (mesh.n_nodes, mesh.cells.shape)
    if plan is not None and plan[0] == key:
        return plan[1]
    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    n_edof = dofs.shape[1]
    rows = np.repeat(dofs, n_edof, axis=1).reshape(-1)
    cols = np.tile(dofs, (1, n_edof)).reshape(-1)
    n_dof = mesh.n_dofs
    keys, slot = np.unique(cols.astype(np.int64) * n_dof + rows, return_inverse=True)
    indices = (keys % n_dof).astype(np.int32)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(keys // n_dof, minlength=n_dof))])
    result = (indptr, indices, slot.reshape(-1))
    mesh._assembly_plan = (key, result)
    return result


def load_vector(mesh: Mesh, load: LoadCase) -> np.ndarray:
    f = np.zeros(mesh.n_dofs)
    per_node = np.outer(load.node_shares(), load.components(mesh.dim))  # (n, dim)
    dofs = element_dofs(load.nodes[None, :], mesh.dofs_per_node)[0]  # node-major
    np.add.at(f, dofs, per_node.reshape(-1))
    return f


def fixed_dofs(mesh: Mesh, node_set: str | Sequence[str]) -> np.ndarray:
    """Every DOF of every node in ``node_set`` — one name or several (fully
    clamped boundaries)."""
    names = [node_set] if isinstance(node_set, str) else list(node_set)
    nodes = np.unique(np.concatenate([np.asarray(mesh.node_sets[n], dtype=int) for n in names]))
    return node_dofs(nodes, mesh.dofs_per_node)


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
    """Von Mises stress at each element's centre, MPa (plane stress in 2D)."""
    el = mesh.element
    d = material.stiffness_matrix(el.dim)
    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    b_all, _ = elements.strain_displacement(el, mesh.nodes[mesh.cells], el.centre)
    strain = np.einsum("eij,ej->ei", b_all, u[dofs])
    stress = strain @ d.T
    if scale is not None:
        stress = stress * np.asarray(scale)[:, None]
    return _von_mises(stress)


Load = "LoadCase | Sequence[LoadCase] | np.ndarray"


def force_vector(mesh: Mesh, load: Any) -> np.ndarray:
    """The global force vector of a load: one :class:`LoadCase`, several (their
    sum — the forces of one load case) or a ready vector."""
    if isinstance(load, np.ndarray):
        if load.shape != (mesh.n_dofs,):
            raise ValueError(f"a force vector needs {mesh.n_dofs} entries, not {load.shape}")
        return np.asarray(load, dtype=float)
    if isinstance(load, LoadCase):
        return load_vector(mesh, load)
    return sum((load_vector(mesh, part) for part in load), np.zeros(mesh.n_dofs))


def constrained_dofs(
    mesh: Mesh,
    fixed_node_set: str | Sequence[str] | None = None,
    prescribed: Mapping[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(constrained DOFs, their prescribed values): every DOF of the nodes in
    ``fixed_node_set`` held at zero, plus the ``prescribed`` DOF -> value map
    (a component held at a given displacement, zero or not)."""
    values: dict[int, float] = {}
    if fixed_node_set is not None:
        values.update((int(d), 0.0) for d in fixed_dofs(mesh, fixed_node_set))
    for dof, value in (prescribed or {}).items():
        if int(dof) in values and values[int(dof)] != float(value):
            raise ValueError(f"DOF {dof} is held at both {values[int(dof)]:g} and {value:g}")
        values[int(dof)] = float(value)
    if not values:
        raise ValueError("nothing is constrained: the part is free to move")
    dofs = np.array(sorted(values), dtype=int)
    return dofs, np.array([values[d] for d in dofs])


def rigid_body_free(mesh: Mesh, dofs: np.ndarray) -> int:
    """How many independent rigid-body motions the constrained ``dofs`` leave
    possible: 0 when the part is held, otherwise the number of translations
    and rotations it could still make without deforming."""
    modes = rigid_body_modes(mesh)[dofs]
    if modes.size == 0:
        return modes.shape[1]
    scale = np.linalg.norm(modes, axis=0)
    sv = np.linalg.svd(modes / np.maximum(scale, 1e-300), compute_uv=False)
    return int(modes.shape[1] - np.sum(sv > 1e-9 * sv.max()))


def solve(
    mesh: Mesh,
    material: Material,
    thickness: float,
    load: Any,
    fixed_node_set: str | Sequence[str] | None = None,
    densities: np.ndarray | None = None,
    penal: float = 3.0,
    ke_all: np.ndarray | None = None,
    solver: str = "auto",
    x0: np.ndarray | None = None,
    prescribed: Mapping[int, float] | None = None,
) -> FEResult:
    """Linear-elastic solve of one load case; see :func:`solve_cases`."""
    return solve_cases(
        mesh, material, thickness, [load], fixed_node_set, densities, penal, ke_all,
        solver, None if x0 is None else [x0], prescribed,
    )[0]


def solve_cases(
    mesh: Mesh,
    material: Material,
    thickness: float,
    loads: Sequence[Any],
    fixed_node_set: str | Sequence[str] | None = None,
    densities: np.ndarray | None = None,
    penal: float = 3.0,
    ke_all: np.ndarray | None = None,
    solver: str = "auto",
    x0: Sequence[np.ndarray | None] | None = None,
    prescribed: Mapping[int, float] | Sequence[Mapping[int, float] | None] | None = None,
    return_system: bool = False,
) -> list[FEResult] | tuple[list[FEResult], list["LinearSystem"]]:
    """Linear-elastic solves of several load cases.

    With ``return_system`` it also returns each case's :class:`LinearSystem`
    — the constrained stiffness and its factorisation — for adjoint solves.

    ``densities`` (if given) applies SIMP scaling.  Plane stress on a 2D mesh
    (with ``thickness``), full 3D elasticity on a 3D one.

    Constraints are every DOF of ``fixed_node_set`` held at zero plus the
    ``prescribed`` DOF -> displacement map: one map shared by all cases, or a
    sequence with one map per case (a support that settles in one case, a
    jack that pushes in another).  Cases held on the same DOFs share one
    constrained stiffness and one factorisation.  Each case is solved by
    partitioning, ``K_ff u_f = f_f - K_fp u_p``.  Its compliance is
    ``f . u - u_p . r_p`` — the work of the loads less that of the reactions
    at prescribed DOFs, i.e. minus twice the potential energy at equilibrium.
    It is ``f . u`` without prescribed displacements, and its sensitivity to
    an element's stiffness is ``-u_e^T dk_e u_e`` either way, so the SIMP
    loop needs no special case.

    ``solver`` is ``"direct"`` (sparse LU, factorised once per constrained
    set), ``"cg"`` (conjugate gradients preconditioned by smoothed-aggregation
    algebraic multigrid, built on the rigid-body modes) or ``"auto"``: direct
    up to :data:`DIRECT_MAX_DOFS` free DOFs, CG above.  ``x0`` (one entry per
    case) warm-starts CG — the SIMP loop passes the previous displacements.
    """
    if ke_all is None:
        ke_all = element_stiffnesses(mesh, material, thickness)
    scale = None if densities is None else simp_scaling(densities, penal)

    k = assemble_stiffness(mesh, ke_all, scale)
    per_case = _per_case(prescribed, len(loads))
    systems: dict[bytes, LinearSystem] = {}
    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    results, case_systems = [], []
    for index, load in enumerate(loads):
        fixed, values = constrained_dofs(mesh, fixed_node_set, per_case[index])
        system = systems.get(fixed.tobytes())
        if system is None:
            loose = rigid_body_free(mesh, fixed)
            if loose:
                raise ValueError(
                    f"the constraints leave {loose} rigid-body motion(s) free "
                    "(translations or rotations that deform nothing): hold more components"
                )
            free = np.setdiff1d(np.arange(mesh.n_dofs), fixed, assume_unique=True)
            system = LinearSystem(mesh=mesh, free=free, k_ff=k[free][:, free],
                                  solver=_choose_solver(solver, free.size), fixed=fixed)
            systems[fixed.tobytes()] = system
        free = system.free
        f = force_vector(mesh, load)
        u = np.zeros(mesh.n_dofs)
        u[fixed] = values
        rhs = f[free]
        if np.any(values):
            if system.k_fp is None:
                system.k_fp = k[free][:, fixed]
            rhs = rhs - system.k_fp @ values
        case_used, iterations = system.solver, 0
        if system.solver == "amg-cg":
            start = None if not x0 or x0[index] is None else np.asarray(x0[index])[free]
            solution, iterations = _amg_cg(mesh, free, system.k_ff, rhs, start)
            if solution is None:  # CG did not converge: fall back to the direct solve
                case_used = "direct (after CG failed)"
            else:
                u[free] = solution
        if case_used.startswith("direct"):
            if system.factor is None:
                system.factor = spla.splu(system.k_ff.tocsc())
            u[free] = system.factor.solve(rhs)

        residual = k @ u - f  # the reactions on constrained DOFs
        ue = u[dofs]  # (n_elements, nodes_per_cell * dofs_per_node)
        unscaled = np.einsum("ei,eij,ej->e", ue, ke_all, ue)
        element_compliance = unscaled if scale is None else unscaled * scale
        results.append(FEResult(
            u=u,
            compliance=float(f @ u - values @ residual[fixed]),
            element_compliance=element_compliance,
            element_compliance_unscaled=unscaled,
            von_mises=centroid_von_mises(mesh, u, material, scale),
            reactions=residual,
            equilibrium_residual=float(np.linalg.norm(residual[free])),
            n_free_dofs=int(free.size),
            dofs_per_node=mesh.dofs_per_node,
            solver=case_used,
            solver_iterations=iterations,
        ))
        case_systems.append(system)
    if return_system:
        return results, case_systems
    return results


def _per_case(
    prescribed: Mapping[int, float] | Sequence[Mapping[int, float] | None] | None, n: int,
) -> list[Mapping[int, float] | None]:
    """One prescribed-displacement map per case, from a shared map or a list."""
    if prescribed is None or isinstance(prescribed, Mapping):
        return [prescribed] * n
    maps = list(prescribed)
    if len(maps) != n:
        raise ValueError(f"prescribed has {len(maps)} maps for {n} load cases")
    return maps


@dataclass
class LinearSystem:
    """The constrained stiffness of one solve, kept for adjoint solves: the
    same matrix with any right-hand side, zero at the constrained DOFs."""

    mesh: Mesh
    free: np.ndarray
    k_ff: sp.spmatrix
    factor: Any = None
    solver: str = "direct"
    fixed: np.ndarray | None = None
    k_fp: sp.spmatrix | None = None

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """``K lambda = rhs`` with lambda = 0 on constrained DOFs (the adjoint
        of a response whose derivative with respect to u is ``rhs``)."""
        out = np.zeros(self.mesh.n_dofs)
        r = np.asarray(rhs)[self.free]
        if not np.any(r):
            return out
        if self.solver == "amg-cg":
            solution, _ = _amg_cg(self.mesh, self.free, self.k_ff, r, None)
            if solution is not None:
                out[self.free] = solution
                return out
        if self.factor is None:
            self.factor = spla.splu(self.k_ff.tocsc())
        out[self.free] = self.factor.solve(r)
        return out


#: Above this many free DOFs, ``solver="auto"`` switches from the direct sparse
#: LU to multigrid-preconditioned CG (a quadratic-tet mesh of the published
#: part at 1 mm has ~55k: LU ~7 s per solve, warm-started CG a fraction).
DIRECT_MAX_DOFS = 30_000
SOLVERS = ("auto", "direct", "cg")
#: CG stops at this relative residual; compliance is then good to ~1e-10.
CG_RTOL = 1e-9


def _choose_solver(solver: str, n_free: int) -> str:
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {', '.join(SOLVERS)}, not {solver!r}")
    if solver == "cg" or (solver == "auto" and n_free > DIRECT_MAX_DOFS):
        try:
            import pyamg  # noqa: F401
        except ImportError:  # pragma: no cover - pyamg is a dependency
            return "direct"
        return "amg-cg"
    return "direct"


def rigid_body_modes(mesh: Mesh) -> np.ndarray:
    """The rigid-body displacement fields of the mesh, (n_dofs, 3 in 2D or 6 in 3D):
    translations and infinitesimal rotations — the near-null space multigrid
    needs to coarsen an elasticity problem well."""
    d, n = mesh.dofs_per_node, mesh.n_nodes
    x = mesh.nodes - mesh.nodes.mean(axis=0)
    modes = np.zeros((d * n, 3 if d == 2 else 6))
    for i in range(d):
        modes[i::d, i] = 1.0
    if d == 2:
        modes[0::2, 2], modes[1::2, 2] = -x[:, 1], x[:, 0]
    else:
        for col, (a, b) in zip((3, 4, 5), ((0, 1), (1, 2), (2, 0))):
            modes[a::3, col], modes[b::3, col] = -x[:, b], x[:, a]
    return modes


def _amg_cg(mesh: Mesh, free: np.ndarray, k: sp.spmatrix, f: np.ndarray,
            x0: np.ndarray | None) -> tuple[np.ndarray | None, int]:
    """CG on ``k u = f`` preconditioned by smoothed-aggregation AMG built on
    the rigid-body modes; (u, iterations), or (None, iterations) if it did not
    converge.  The hierarchy is rebuilt for every solve: reusing one from an
    earlier SIMP iteration was measured slower, the density changes making CG
    need several times the iterations."""
    import pyamg

    k = k.tocsr()
    ml = pyamg.smoothed_aggregation_solver(k, B=rigid_body_modes(mesh)[free], max_coarse=500)
    count = [0]

    def tick(_):
        count[0] += 1

    u, info = spla.cg(k, f, x0=x0, rtol=CG_RTOL, maxiter=2000,
                      M=ml.aspreconditioner(), callback=tick)
    return (u if info == 0 else None), count[0]


def timoshenko_tip_deflection(
    length: float, height: float, thickness: float, material: Material, load: float
) -> dict[str, float]:
    """Cantilever tip deflection under an end load: bending + shear (k = 5/6)."""
    i = thickness * height**3 / 12.0
    area = thickness * height
    bending = abs(load) * length**3 / (3.0 * material.youngs_modulus * i)
    shear = abs(load) * length / ((5.0 / 6.0) * material.shear_modulus * area)
    return {"bending": bending, "shear": shear, "total": bending + shear}
