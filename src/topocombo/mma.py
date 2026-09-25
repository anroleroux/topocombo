"""Gradient-based optimisation with the Method of Moving Asymptotes: any
objective, any limits.

The Optimality Criteria update (:func:`topocombo.optimize.oc_update`) is
built for one problem — minimum compliance under a volume constraint.  The
Method of Moving Asymptotes (MMA, Svanberg 1987) takes any differentiable
objective and any number of inequality constraints, so it opens the problems
engineers actually pose:

* minimum compliance with the volume and, say, a displacement or stress limit;
* minimum volume (the lightest part) that keeps compliance, a displacement or
  the stress within limits.

Every response comes with its exact gradient (:mod:`topocombo.responses`).
MMA needs true gradients, so the densities are smoothed with the *density*
filter — ``x_phys = H x / Hs`` — and gradients are carried back through it
(``dF/dx = H^T (dF/dx_phys / Hs)``); the sensitivity filter that OC uses is a
heuristic that no longer is a gradient.  Elements held void or solid are
fixed through their bounds.  Constraints are normalised to ``g / limit - 1 <=
0``, a compliance objective to its starting value.

Two drivers:

``mma`` (the default): MMA implemented here from Svanberg's paper, as used in
    topology optimisation — moving asymptotes that widen while the design
    keeps going one way and close in when it oscillates, the same move limit
    as OC on every step, and elastic constraints (a slack with a large
    penalty) so an infeasible iterate never makes the subproblem empty.  The
    convex, separable subproblem is solved through its dual, one variable per
    limit, with L-BFGS-B.  One state solve (plus one adjoint per displacement
    or stress limit) per iteration; it stops when the design changes by less
    than the tolerance with every limit met.
``nlopt``: NLopt's MMA.  Good on smooth problems (minimum compliance), but it
    has no move limit and starts with cautious steps, which let stress-limited
    designs run away and stop minimum-volume runs from a full start at once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import responses
from .fea import Material, element_stiffnesses, solve_cases
from .mesh_io import Mesh
from .optimize import OptResult, SimpParams, build_filter

OBJECTIVES = ("compliance", "volume")
DRIVERS = ("mma", "nlopt")


@dataclass
class Limit:
    """One inequality constraint of the optimisation, ``value <= bound``.

    ``kind`` is ``volume``, ``compliance``, ``displacement`` or ``stress``;
    ``case`` the load-case index it reads (None: the weighted sum, for
    compliance).  A displacement limit bounds ``|selector . u|``; a stress
    limit the p-norm of the relaxed von Mises stress.
    """

    kind: str
    bound: float
    label: str
    case: int | None = None
    selector: np.ndarray | None = None
    p: float = 8.0
    q: float = 0.5
    values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in ("volume", "compliance", "displacement", "stress"):
            raise ValueError(f"unknown limit kind {self.kind!r}")
        if not self.bound > 0:
            raise ValueError(f"limit '{self.label}' needs a positive bound")
        if self.kind in ("displacement", "stress") and self.case is None:
            raise ValueError(f"limit '{self.label}' needs a load case")


# --------------------------------------------------------------------------
# the MMA step (Svanberg 1987)
# --------------------------------------------------------------------------
class MMAState:
    """Moving asymptotes and the last two iterates, for :func:`mma_step`."""

    def __init__(self, n: int, asymptote_init: float = 0.5,
                 asymptote_grow: float = 1.2, asymptote_shrink: float = 0.7) -> None:
        self.iteration = 0
        self.x1: np.ndarray | None = None
        self.x2: np.ndarray | None = None
        self.low = np.zeros(n)
        self.upp = np.zeros(n)
        self.init, self.grow, self.shrink = asymptote_init, asymptote_grow, asymptote_shrink


def mma_step(state: MMAState, x: np.ndarray, f0: float, df0: np.ndarray,
             g: np.ndarray, dg: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
             move: float, c: float = 1000.0) -> np.ndarray:
    """One MMA iteration: the next design from values and gradients at ``x``.

    Minimises the convex approximation of ``f0`` subject to the approximations
    of ``g <= 0`` (made elastic: each may be exceeded at a cost ``c`` per
    unit), within the move limit and the asymptotes.  ``g`` has shape (m,),
    ``dg`` (m, n).
    """
    m = g.size
    span = np.maximum(xmax - xmin, 1e-12)
    state.iteration += 1
    # asymptotes: wide at first, then widened while the design keeps moving
    # one way and pulled in when it oscillates
    if state.iteration <= 2:
        low, upp = x - state.init * span, x + state.init * span
    else:
        trend = (x - state.x1) * (state.x1 - state.x2)
        factor = np.where(trend > 0, state.grow, np.where(trend < 0, state.shrink, 1.0))
        low = x - factor * (state.x1 - state.low)
        upp = x + factor * (state.upp - state.x1)
        low = np.clip(low, x - 10.0 * span, x - 0.01 * span)
        upp = np.clip(upp, x + 0.01 * span, x + 10.0 * span)
    state.low, state.upp = low, upp
    state.x2, state.x1 = state.x1, x.copy()
    # the step's box: bounds, move limit, and a margin inside the asymptotes
    alpha = np.maximum.reduce([xmin, x - move * span, low + 0.1 * (x - low)])
    beta = np.minimum.reduce([xmax, x + move * span, upp - 0.1 * (upp - x)])
    # convex approximations r + sum p / (U - y) + q / (y - L) of f0 and each g
    grads = np.vstack([df0[None, :], dg])  # (m + 1, n)
    values = np.concatenate([[f0], g])
    ux, xl = upp - x, x - low
    pos, neg = np.maximum(grads, 0.0), np.maximum(-grads, 0.0)
    reg = 1e-5 / span
    p = ux**2 * (1.001 * pos + 0.001 * neg + reg)
    q = xl**2 * (0.001 * pos + 1.001 * neg + reg)
    r = values - (p / ux + q / xl).sum(axis=1)

    def primal(lam: np.ndarray) -> np.ndarray:
        weights = np.concatenate([[1.0], lam])
        sp, sq = np.sqrt(weights @ p), np.sqrt(weights @ q)
        y = (sp * low + sq * upp) / np.maximum(sp + sq, 1e-300)
        return np.clip(y, alpha, beta)

    def neg_dual(lam: np.ndarray) -> tuple[float, np.ndarray]:
        y = primal(lam)
        approx = r + (p / (upp - y) + q / (y - low)).sum(axis=1)  # each function at y
        return -(approx[0] + lam @ approx[1:]), -approx[1:]

    if m == 0:
        return primal(np.zeros(0))
    from scipy.optimize import minimize

    sol = minimize(neg_dual, np.ones(m), jac=True, method="L-BFGS-B",
                   bounds=[(0.0, c)] * m, options={"maxiter": 500, "gtol": 1e-10})
    return primal(sol.x)


# --------------------------------------------------------------------------
# the problem, and the two drivers
# --------------------------------------------------------------------------
def optimize_mma(
    mesh: Mesh,
    material: Material,
    thickness: float,
    cases: Sequence[tuple[float, Any]],
    params: SimpParams,
    objective: str = "compliance",
    limits: Sequence[Limit] = (),
    fixed_node_set: Any = None,
    prescribed: Mapping[int, float] | None = None,
    passive: np.ndarray | None = None,
    solid: np.ndarray | None = None,
    solver: str = "auto",
    on_iteration: Callable[[dict[str, float], np.ndarray], None] | None = None,
    start: float | None = None,
    driver: str = "mma",
) -> OptResult:
    """Minimise ``objective`` (``compliance``: the weighted sum over
    ``cases``; ``volume``: the volume fraction) subject to ``limits``.

    With a compliance objective the volume fraction of ``params`` is a limit
    too, as for OC.  ``start`` is the initial density (default: the volume
    fraction for compliance, the full part for volume).  ``driver`` is
    ``"mma"`` or ``"nlopt"``.  Returns the design as :class:`OptResult`,
    whose ``limits`` hold each limit's final value.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}, not {objective!r}")
    if driver not in DRIVERS:
        raise ValueError(f"driver is one of {', '.join(DRIVERS)}, not {driver!r}")
    limits = list(limits)
    if objective == "compliance":
        limits.insert(0, Limit("volume", params.volume_fraction, "volume fraction"))
    elif not any(lim.kind != "volume" for lim in limits):
        raise ValueError("minimum volume needs a limit on compliance, a displacement or stress")
    n = mesh.n_elements
    passive = np.zeros(n, bool) if passive is None else np.asarray(passive, bool)
    solid = np.zeros(n, bool) if solid is None else np.asarray(solid, bool)
    weights = np.array([float(w) for w, _ in cases])
    loads = [load for _, load in cases]

    ke_all = element_stiffnesses(mesh, material, thickness)
    measures = mesh.cell_measures()
    measure_fraction = measures / measures.sum()
    h, hs = build_filter(mesh, params.filter_radius)
    ht = h.T.tocsr()
    fixed = passive | solid
    lower, upper = np.zeros(n), np.ones(n)
    lower[solid] = 1.0
    upper[passive] = 0.0

    def physical(x: np.ndarray) -> np.ndarray:
        x_phys = np.asarray(h @ x).ravel() / hs
        x_phys[passive], x_phys[solid] = 0.0, 1.0
        return x_phys

    def to_design(grad_phys: np.ndarray) -> np.ndarray:
        g = np.where(fixed, 0.0, grad_phys)
        return np.asarray(ht @ (g / hs)).ravel()

    cache: dict[str, Any] = {"key": None}
    u_prev: list[np.ndarray | None] = [None] * len(loads)

    def evaluate(x: np.ndarray) -> dict[str, Any]:
        key = x.tobytes()
        if cache["key"] == key:
            return cache["value"]
        x_phys = physical(x)
        results, system = solve_cases(
            mesh, material, thickness, loads, fixed_node_set, densities=x_phys,
            penal=params.penal, ke_all=ke_all, solver=solver, x0=u_prev,
            prescribed=prescribed, return_system=True,
        )
        u_prev[:] = [r.u for r in results]
        comp, comp_grad, case_values = responses.compliance(results, weights, x_phys, params.penal)
        vol, vol_grad = responses.volume(measure_fraction, x_phys)
        out = {"x_phys": x_phys, "compliance": comp, "case_compliances": case_values,
               "volume": vol, "limits": [],
               "solver_iterations": sum(r.solver_iterations for r in results)}
        if objective == "compliance":
            out["objective"], out["objective_grad"] = comp, to_design(comp_grad)
        else:
            out["objective"], out["objective_grad"] = vol, to_design(vol_grad)
        for lim in limits:
            if lim.kind == "volume":
                value, grad = vol, vol_grad
            elif lim.kind == "compliance":
                if lim.case is None:
                    value, grad = comp, comp_grad
                else:
                    value, grad, _ = responses.compliance(
                        [results[lim.case]], [1.0], x_phys, params.penal
                    )
            elif lim.kind == "displacement":
                d, dd = responses.displacement(
                    mesh, system, results[lim.case], ke_all, lim.selector, x_phys, params.penal
                )
                value, grad = abs(d), np.sign(d) * dd
            else:
                value, grad, _ = responses.stress_pnorm(
                    mesh, material, system, results[lim.case], ke_all, x_phys,
                    params.penal, q=lim.q, p=lim.p,
                )
            # normalised: value / bound - 1 <= 0
            out["limits"].append((value, value / lim.bound - 1.0, to_design(grad) / lim.bound))
        cache["key"], cache["value"] = key, out
        return out

    history: list[dict[str, float]] = []
    clock: dict[str, Any] = {"t0": time.perf_counter(), "x_last": None, "scale": None}

    def log(x: np.ndarray, ev: dict[str, Any]) -> tuple[float, float]:
        """Record an evaluation; return (design change, worst normalised limit)."""
        change = (float(np.abs(x - clock["x_last"]).max())
                  if clock["x_last"] is not None else 1.0)
        clock["x_last"] = x.copy()
        violation = max((g for _, g, _ in ev["limits"]), default=-1.0)
        x_phys = ev["x_phys"]
        record = {
            "iteration": len(history) + 1,
            "compliance": ev["compliance"],
            "volume_fraction": ev["volume"],
            "change": change,
            "measure_of_discreteness": float(np.mean(4.0 * x_phys * (1.0 - x_phys)) * 100.0),
            "seconds": time.perf_counter() - clock["t0"],
            "solver_iterations": ev["solver_iterations"],
            "max_violation": float(violation),
        }
        clock["t0"] = time.perf_counter()
        history.append(record)
        if on_iteration is not None:
            on_iteration(record, x_phys)
        return change, violation

    def scaled_objective(ev: dict[str, Any]) -> tuple[float, np.ndarray]:
        if clock["scale"] is None:
            clock["scale"] = (abs(ev["objective"]) or 1.0) if objective == "compliance" else 1.0
        return ev["objective"] / clock["scale"], ev["objective_grad"] / clock["scale"]

    if start is None:
        start = params.volume_fraction if objective == "compliance" else 1.0
    x = np.clip(np.full(n, float(start)), lower, upper)
    converged = False
    if driver == "mma":
        mma = MMAState(n)
        for _ in range(params.max_iterations):
            ev = evaluate(x)
            change, violation = log(x, ev)
            if len(history) > 1 and change < params.tolerance and violation <= 1e-3:
                converged = True
                break
            f0, df0 = scaled_objective(ev)
            g = np.array([gv for _, gv, _ in ev["limits"]])
            dg = np.array([gd for _, _, gd in ev["limits"]]).reshape(len(g), n)
            x = mma_step(mma, x, f0, df0, g, dg, lower, upper, params.move_limit)
    else:
        x, converged = _run_nlopt(x, lower, upper, evaluate, log, scaled_objective, params,
                                  len(limits))
    final = evaluate(x)
    for lim, (value, _, _) in zip(limits, final["limits"]):
        lim.values.append(value)
    return OptResult(
        densities=final["x_phys"],
        compliance=float(final["compliance"]),
        volume_fraction=float(final["volume"]),
        iterations=len(history),
        converged=converged,
        history=history,
        case_compliances=list(final["case_compliances"]),
        limits=[
            {"kind": lim.kind, "label": lim.label, "bound": lim.bound, "value": value,
             "satisfied": bool(value <= lim.bound * (1 + 1e-3))}
            for lim, (value, _, _) in zip(limits, final["limits"])
        ],
        optimizer=driver,
        objective=objective,
    )


def _run_nlopt(x0, lower, upper, evaluate, log, scaled_objective, params, n_limits):
    """NLopt's MMA on the same problem; returns (design, converged)."""
    import nlopt

    def objective_fn(x: np.ndarray, grad: np.ndarray) -> float:
        ev = evaluate(x)
        log(x, ev)
        f0, df0 = scaled_objective(ev)
        if grad.size:
            grad[:] = df0
        return f0

    def constraint_fn(index: int):
        def g(x: np.ndarray, grad: np.ndarray) -> float:
            _, value, gradient = evaluate(x)["limits"][index]
            if grad.size:
                grad[:] = gradient
            return value
        return g

    opt = nlopt.opt(nlopt.LD_MMA, x0.size)
    opt.set_lower_bounds(lower)
    opt.set_upper_bounds(upper)
    opt.set_min_objective(objective_fn)
    for i in range(n_limits):
        opt.add_inequality_constraint(constraint_fn(i), 1e-6)
    opt.set_maxeval(params.max_iterations)
    # NLopt's MMA takes cautious first steps, so a design-change test on
    # successive evaluations would stop it at once; its own tests compare
    # accepted iterates
    opt.set_xtol_abs(params.tolerance * 0.1)
    opt.set_ftol_rel(1e-6)
    x = opt.optimize(x0)
    code = opt.last_optimize_result()
    ok = code in (nlopt.XTOL_REACHED, nlopt.FTOL_REACHED, nlopt.SUCCESS)
    return x, ok and all(g <= 1e-3 for _, g, _ in evaluate(x)["limits"])
