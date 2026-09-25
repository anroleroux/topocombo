"""SIMP compliance minimisation with an Optimality Criteria update.

One iteration is: solve → compliance and sensitivities → filter → density
update → convergence check. The FEA solver is called directly (in memory), so
the element stiffness matrices are assembled once at full density and reused
every iteration; only the density scaling changes.

The loop prints per-iteration metrics and appends to a CSV. It writes density
snapshots periodically, and never plots — the report is built afterwards from
what lands on disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .fea import Material, element_stiffnesses, solve_cases
from .mesh_io import Mesh, vtk_points


@dataclass(frozen=True)
class SimpParams:
    """Settings of the SIMP loop."""

    volume_fraction: float = 0.5
    penal: float = 3.0
    filter_radius: float = 1.5  # mm; ~1.5 element sizes on the default mesh
    move_limit: float = 0.2
    max_iterations: int = 80
    tolerance: float = 0.01  # max density change that counts as converged
    e_min: float = 1e-9
    filter_type: str = "sensitivity"  # "sensitivity" (Sigmund) or "density"
    #: "oc" (Optimality Criteria: minimum compliance under the volume only),
    #: "mma" (Method of Moving Asymptotes: any objective and limits; density
    #: filter), "nlopt" (NLopt's MMA) or "auto": OC where it applies, MMA
    #: otherwise
    optimizer: str = "auto"

    def __post_init__(self) -> None:
        if not 0.0 < self.volume_fraction <= 1.0:
            raise ValueError("volume_fraction must lie in (0, 1]")
        if self.penal < 1.0:
            raise ValueError("penal must be >= 1")
        if self.filter_radius <= 0.0:
            raise ValueError("filter_radius must be positive")
        if self.filter_type not in ("density", "sensitivity"):
            raise ValueError("filter_type must be 'density' or 'sensitivity'")
        if self.optimizer not in ("auto", "oc", "mma", "nlopt"):
            raise ValueError("optimizer must be 'auto', 'oc', 'mma' or 'nlopt'")
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be a whole number >= 1")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OptResult:
    """Final design plus the per-iteration history."""

    densities: np.ndarray  # (n_elements,) physical densities in [0, 1]
    compliance: float
    volume_fraction: float
    iterations: int
    converged: bool
    history: list[dict[str, float]] = field(default_factory=list)
    case_compliances: list[float] = field(default_factory=list)  # per load case, last iteration
    limits: list[dict[str, Any]] = field(default_factory=list)  # each limit's final value (MMA)
    optimizer: str = "oc"
    objective: str = "compliance"

    def measure_of_discreteness(self) -> float:
        """Mnd (%): 0 = fully black-and-white, 100 = every element at 0.5."""
        x = self.densities
        return float(np.mean(4.0 * x * (1.0 - x)) * 100.0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "compliance": self.compliance,
            "volume_fraction": self.volume_fraction,
            "iterations": self.iterations,
            "converged": self.converged,
            "case_compliances": list(self.case_compliances),
            "optimizer": self.optimizer,
            "objective": self.objective,
            "limits": list(self.limits),
            "measure_of_discreteness": self.measure_of_discreteness(),
            "density_min": float(self.densities.min()),
            "density_max": float(self.densities.max()),
            "solid_fraction": float(np.mean(self.densities > 0.9)),
            "void_fraction": float(np.mean(self.densities < 0.1)),
        }


def element_centroids(mesh: Mesh) -> np.ndarray:
    return mesh.nodes[mesh.cells].mean(axis=1)


def build_filter(mesh: Mesh, radius: float) -> tuple[sp.csr_matrix, np.ndarray]:
    """Cone-shaped neighbourhood weights ``H`` and their row sums ``Hs``.

    Filtering couples neighbouring elements, which is what stops the
    checkerboard patterns that the element-wise density field would otherwise
    converge to.  On a mesh of unequal elements (body-fitted) each neighbour is
    also weighted by its size, so a patch of small elements does not outvote a
    large one; on a uniform grid that factor is constant and left out.
    """
    centroids = element_centroids(mesh)
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(radius, output_type="ndarray")

    # every element is its own neighbour, with the maximum weight
    rows = [np.arange(mesh.n_elements)]
    cols = [np.arange(mesh.n_elements)]
    weights = [np.full(mesh.n_elements, radius)]

    if pairs.size:
        dist = np.linalg.norm(centroids[pairs[:, 0]] - centroids[pairs[:, 1]], axis=1)
        w = radius - dist
        rows += [pairs[:, 0], pairs[:, 1]]
        cols += [pairs[:, 1], pairs[:, 0]]
        weights += [w, w]

    h = sp.coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(mesh.n_elements, mesh.n_elements),
    ).tocsr()
    measures = mesh.cell_measures()
    if np.ptp(measures) > 1e-9 * measures.mean():
        h = (h @ sp.diags(measures / measures.mean())).tocsr()
    hs = np.asarray(h.sum(axis=1)).ravel()
    return h, hs


def oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    dv: np.ndarray,
    volume_fraction: float,
    move: float,
    volume_of: Callable[[np.ndarray], float],
    l2_start: float = 1e9,
    passive: np.ndarray | None = None,
    solid: np.ndarray | None = None,
) -> np.ndarray:
    """Optimality Criteria step: bisect the Lagrange multiplier onto the volume.

    Elements flagged ``passive`` are held at zero density, ``solid`` at one.
    """
    l1, l2 = 0.0, l2_start
    x_new = x
    while (l2 - l1) / max(l1 + l2, 1e-30) > 1e-9:
        lmid = 0.5 * (l1 + l2)
        # OC: x * sqrt(-dc / (lambda dv)); dc < 0 for compliance, so the root is real
        ratio = np.sqrt(np.maximum(-dc / (lmid * dv), 0.0))
        x_new = np.clip(np.clip(x * ratio, x - move, x + move), 0.0, 1.0)
        if passive is not None:
            x_new[passive] = 0.0
        if solid is not None:
            x_new[solid] = 1.0
        if volume_of(x_new) > volume_fraction:
            l1 = lmid
        else:
            l2 = lmid
    return x_new


def optimize(
    mesh: Mesh,
    material: Material,
    thickness: float,
    load: Any,
    fixed_node_set: str | Sequence[str] | None,
    params: SimpParams | None = None,
    on_iteration: Callable[[dict[str, float], np.ndarray], None] | None = None,
    passive: np.ndarray | None = None,
    solver: str = "auto",
    cases: Sequence[tuple[float, Any]] | None = None,
    prescribed: Mapping[int, float] | None = None,
    solid: np.ndarray | None = None,
) -> OptResult:
    """Minimise compliance subject to a volume constraint, returning the design.

    The objective is ``load``'s compliance, or with ``cases`` — (weight, load)
    pairs, ``load`` then ignored — the weighted sum of the cases' compliances;
    each case's sensitivity is the same ``-p x^(p-1) u_e^T k0_e u_e``, so the
    sum's is the weighted sum.  Constraints are ``fixed_node_set`` (clamped)
    and ``prescribed`` (DOF -> displacement), see :func:`topocombo.fea.solve_cases`.

    ``passive`` (a boolean mask) marks non-design elements held void — the
    cutouts of the CAD model, or keep-out regions — and ``solid`` those held
    solid.  The volume fraction stays relative to the whole meshed envelope,
    and counts the solid ones.  ``solver`` picks the linear solver; an
    iterative one starts each solve from the previous iteration's
    displacements.
    """
    params = params or SimpParams()
    if passive is not None and not np.any(passive):
        passive = None
    if solid is not None and not np.any(solid):
        solid = None
    if passive is not None and solid is not None and np.any(passive & solid):
        raise ValueError("an element cannot be held both void and solid")
    cases = list(cases) if cases else [(1.0, load)]
    weights = np.array([float(w) for w, _ in cases])
    if np.any(weights <= 0):
        raise ValueError("load case weights must be positive")

    ke_all = element_stiffnesses(mesh, material, thickness)
    measures = mesh.cell_measures()
    measure_fraction = measures / measures.sum()
    h, hs = build_filter(mesh, params.filter_radius)

    x = np.full(mesh.n_elements, params.volume_fraction)
    if passive is not None:
        x[passive] = 0.0
    if solid is not None:
        held = float(measure_fraction[solid].sum())
        if held >= params.volume_fraction:
            raise ValueError(
                f"the regions held solid take {held:.3f} of the volume, no less than the "
                f"target fraction {params.volume_fraction:g}: nothing is left to design"
            )
        x[solid] = 1.0
    history: list[dict[str, float]] = []
    change = float("inf")
    converged = False
    iteration = 0
    x_phys = x.copy()
    compliance = float("nan")

    def physical(design: np.ndarray) -> np.ndarray:
        if params.filter_type == "density":
            design = np.asarray(h @ design).ravel() / hs
            if passive is not None:
                design[passive] = 0.0
            if solid is not None:
                design[solid] = 1.0
        return design

    def volume_of(design: np.ndarray) -> float:
        return float(measure_fraction @ physical(design))

    u_prev: list[np.ndarray | None] = [None] * len(cases)
    case_compliances: list[float] = []
    while iteration < params.max_iterations:
        iteration += 1
        t0 = time.perf_counter()

        x_phys = physical(x)
        results = solve_cases(
            mesh=mesh,
            material=material,
            thickness=thickness,
            loads=[case for _, case in cases],
            fixed_node_set=fixed_node_set,
            densities=x_phys,
            penal=params.penal,
            ke_all=ke_all,
            solver=solver,
            x0=u_prev,
            prescribed=prescribed,
        )
        u_prev = [r.u for r in results]
        case_compliances = [r.compliance for r in results]
        compliance = float(weights @ case_compliances)

        # dc/dx_phys = -p x^(p-1) (1 - Emin) sum_i w_i u_e,i^T k0 u_e,i
        energy = sum(w * r.element_compliance_unscaled for w, r in zip(weights, results))
        dc = (
            -params.penal
            * np.power(np.maximum(x_phys, 1e-12), params.penal - 1.0)
            * (1.0 - params.e_min)
            * energy
        )
        dv = measure_fraction.copy()

        if params.filter_type == "density":
            dc = np.asarray(h @ (dc / hs)).ravel()
            dv = np.asarray(h @ (dv / hs)).ravel()
        else:  # sensitivity filtering acts on dc only
            dc = np.asarray(h @ (x * dc)).ravel() / hs / np.maximum(1e-3, x)

        x_new = oc_update(
            x=x,
            dc=dc,
            dv=dv,
            volume_fraction=params.volume_fraction,
            move=params.move_limit,
            volume_of=volume_of,
            passive=passive,
            solid=solid,
        )
        change = float(np.abs(x_new - x).max())
        x = x_new
        x_phys = physical(x)

        record = {
            "iteration": iteration,
            "compliance": float(compliance),
            "volume_fraction": float(measure_fraction @ x_phys),
            "change": change,
            "measure_of_discreteness": float(np.mean(4.0 * x_phys * (1.0 - x_phys)) * 100.0),
            "seconds": time.perf_counter() - t0,
            "solver_iterations": sum(r.solver_iterations for r in results),
        }
        history.append(record)
        if on_iteration is not None:
            on_iteration(record, x_phys)

        if change < params.tolerance:
            converged = True
            break

    return OptResult(
        densities=x_phys,
        compliance=float(compliance),
        volume_fraction=float(measure_fraction @ x_phys),
        iterations=iteration,
        converged=converged,
        history=history,
        case_compliances=case_compliances,
    )


def save_history(history: list[dict[str, float]], path: Path) -> Path:
    """Write the scalar log as CSV — the format the README asks the loop for."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["iteration", "compliance", "volume_fraction", "change",
               "measure_of_discreteness", "seconds", "solver_iterations"]
    lines = [",".join(columns)]
    for row in history:
        lines.append(",".join(f"{row.get(c, 0):.10g}" for c in columns))
    path.write_text("\n".join(lines) + "\n")
    return path


def save_density_field(mesh: Mesh, densities: np.ndarray, path: Path) -> Path:
    """Write one density field as a .vtu — used for the periodic snapshots."""
    import meshio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meshio.write_points_cells(
        str(path),
        vtk_points(mesh),
        [(mesh.cell_type, mesh.cells)],
        cell_data={"density": [np.asarray(densities, dtype=float)]},
    )
    return path


def save_design(mesh: Mesh, result: OptResult, out_dir: Path) -> dict[str, Path]:
    """Write the final density field for the solver side and for PyVista."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = out_dir / "density.npz"
    np.savez_compressed(
        npz,
        densities=result.densities,
        compliance=np.asarray([result.compliance]),
        volume_fraction=np.asarray([result.volume_fraction]),
        iterations=np.asarray([result.iterations]),
    )
    vtu = save_density_field(mesh, result.densities, out_dir / "density.vtu")
    return {"npz": npz, "vtu": vtu}
