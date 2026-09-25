"""What the optimizer can minimise or limit, with exact gradients.

Every response here is a function of the physical (filtered) densities ``x``
through the SIMP stiffness ``E(x) = E_min + x^p (1 - E_min)``, and returns
its value and its gradient with respect to ``x``:

* compliance of each load case (and their weighted sum) — self-adjoint;
* volume fraction — explicit;
* a displacement: the mean of one component over a region's nodes — one
  adjoint solve, ``K lambda = L``;
* stress: a p-norm of the elements' von Mises stresses, relaxed by ``x^q``
  (the "qp" approach: a void element's stress vanishes with its density, so
  the optimizer can remove material without the constraint blocking it) —
  one adjoint solve, with the adjoint load assembled from ``d(norm)/du``.

Adjoint solves reuse the factorisation of the state solve
(:class:`topocombo.fea.LinearSystem`).  With ``dK/dx_e = E'(x_e) k0_e``, a
response ``f(u(x), x)`` has ``df/dx_e = (explicit) - E'(x_e) lambda_e^T k0_e
u_e``.  Each gradient is checked against central differences in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import elements
from .fea import FEResult, LinearSystem, Material, element_dofs
from .mesh_io import Mesh

#: Von Mises quadratic forms: sigma_vm^2 = s^T V s, in Voigt order.
VON_MISES = {
    2: np.array([[1.0, -0.5, 0.0], [-0.5, 1.0, 0.0], [0.0, 0.0, 3.0]]),  # plane stress
    3: np.block([
        [np.array([[1.0, -0.5, -0.5], [-0.5, 1.0, -0.5], [-0.5, -0.5, 1.0]]), np.zeros((3, 3))],
        [np.zeros((3, 3)), 3.0 * np.eye(3)],
    ]),
}


def stiffness_slope(x: np.ndarray, penal: float, e_min: float = 1e-9) -> np.ndarray:
    """dE/dx of the SIMP interpolation ``E_min + x^p (1 - E_min)``."""
    return penal * np.power(np.maximum(x, 1e-12), penal - 1.0) * (1.0 - e_min)


def _implicit(mesh: Mesh, ke_all: np.ndarray, u: np.ndarray, lam: np.ndarray,
              x: np.ndarray, penal: float) -> np.ndarray:
    """-E'(x_e) lambda_e^T k0_e u_e for every element."""
    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    return -stiffness_slope(x, penal) * np.einsum("ei,eij,ej->e", lam[dofs], ke_all, u[dofs])


def compliance(results: Sequence[FEResult], weights: Sequence[float], x: np.ndarray,
               penal: float) -> tuple[float, np.ndarray, list[float]]:
    """(weighted compliance, its gradient, each case's compliance).

    Compliance ``f.u - u_p.r_p`` is self-adjoint: ``dC/dx_e = -E'(x_e)
    u_e^T k0_e u_e`` (see :func:`topocombo.fea.solve_cases`).
    """
    slope = stiffness_slope(x, penal)
    values = [r.compliance for r in results]
    grad = -slope * sum(w * r.element_compliance_unscaled for w, r in zip(weights, results))
    return float(np.dot(weights, values)), grad, values


def volume(measure_fraction: np.ndarray, x: np.ndarray) -> tuple[float, np.ndarray]:
    """(volume fraction, its gradient): the measure-weighted mean density."""
    return float(measure_fraction @ x), measure_fraction.copy()


def displacement(mesh: Mesh, system: LinearSystem, result: FEResult, ke_all: np.ndarray,
                 selector: np.ndarray, x: np.ndarray, penal: float) -> tuple[float, np.ndarray]:
    """(``selector . u``, its gradient): e.g. the mean of one component over
    a region's nodes, with ``selector`` holding 1/n at those DOFs."""
    value = float(selector @ result.u)
    lam = system.solve(selector)
    return value, _implicit(mesh, ke_all, result.u, lam, x, penal)


def displacement_selector(mesh: Mesh, nodes: np.ndarray, component: int) -> np.ndarray:
    """The DOF weights that average one displacement component over ``nodes``."""
    sel = np.zeros(mesh.n_dofs)
    sel[mesh.dofs_per_node * np.asarray(nodes, dtype=int) + component] = 1.0 / len(nodes)
    return sel


@dataclass
class StressState:
    """Per-element stress quantities at the element centres."""

    von_mises: np.ndarray  # unrelaxed sigma_vm of the full-stiffness material, MPa
    relaxed: np.ndarray  # x^q sigma_vm — what the norm aggregates
    pnorm: float


def stress_pnorm(mesh: Mesh, material: Material, system: LinearSystem, result: FEResult,
                 ke_all: np.ndarray, x: np.ndarray, penal: float, q: float = 0.5,
                 p: float = 8.0) -> tuple[float, np.ndarray, StressState]:
    """(p-norm of the relaxed von Mises stress, its gradient, the stresses).

    ``s_e = x_e^q sigma_vm(D B_e u_e)`` at each element's centre, and
    ``PN = (sum_e s_e^p)^(1/p)``: a smooth stand-in for the largest stress,
    never below it (it tends to the maximum as p grows).  ``q < 1`` is the
    qp relaxation.
    """
    el = mesh.element
    d = material.stiffness_matrix(el.dim)
    v = VON_MISES[el.dim]
    b, _ = elements.strain_displacement(el, mesh.nodes[mesh.cells], el.centre)  # (m, ns, nd)
    dofs = element_dofs(mesh.cells, mesh.dofs_per_node)
    sigma = np.einsum("ij,mjk,mk->mi", d, b, result.u[dofs])  # (m, ns)
    s0 = np.sqrt(np.maximum(np.einsum("mi,ij,mj->m", sigma, v, sigma), 0.0))
    xs = np.maximum(x, 1e-9)
    relaxed = xs**q * s0
    pn = float(np.sum(relaxed**p) ** (1.0 / p))
    state = StressState(von_mises=s0, relaxed=relaxed, pnorm=pn)
    if pn == 0.0:
        return 0.0, np.zeros_like(x), state
    dpn_ds = (relaxed / pn) ** (p - 1.0)  # d PN / d s_e
    explicit = dpn_ds * q * xs ** (q - 1.0) * s0
    # d s_e / d u_e = x^q (B^T D^T V sigma) / s0; assembled into the adjoint load
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(s0 > 0, dpn_ds * xs**q / s0, 0.0)
    local = np.einsum("m,mki,kl,lj,mj->mi", scale, b, d.T, v, sigma)  # (m, nd)
    rhs = np.zeros(mesh.n_dofs)
    np.add.at(rhs, dofs, local)
    lam = system.solve(rhs)
    return pn, explicit + _implicit(mesh, ke_all, result.u, lam, x, penal), state

