"""The study: the lightest L-bracket whose stress stays below a limit.

    python -m topocombo.cli all --study examples/lbracket/study.py

Minimum volume subject to a von Mises stress limit — a problem Optimality
Criteria cannot pose, so it runs on MMA (the Method of Moving Asymptotes).
The stress is bounded through a p-norm of the element stresses, relaxed by
density ("qp": a void element's stress fades with it), a smooth bound on the
largest stress.
"""
from topocombo.study import Fix, Force, Material, Mesh, SimpParams, Study, StressLimit

study = Study(
    part="part.py",
    mesh=Mesh(mode="body-fitted", size=1.0),  # quads following the L
    material=Material(youngs_modulus=210_000.0, poisson_ratio=0.3),
    thickness=5.0,  # mm, plane stress
    constraints=[Fix("top")],
    loads=[Force("tip", (0.0, -1500.0))],  # N, spread over the arm's end
    objective="volume",
    limits=[StressLimit(max=350.0, p=8.0)],  # MPa, on the p-norm
    optimize=SimpParams(optimizer="mma", filter_radius=2.0, max_iterations=300),
    crosscheck=True,  # solve again in CalculiX and compare
)
