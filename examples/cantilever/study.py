"""The study: how part.py is meshed, held, loaded and optimized.

    python -m topocombo.cli all --study examples/cantilever/study.py

Constraints and loads refer to the regions part.py names.  Everything here is
ordinary Python, so values can be computed, looped over or imported.
"""
from topocombo.study import Fix, Force, Material, Mesh, SimpParams, Study

study = Study(
    part="part.py",
    # body-fitted: the mesh follows the notch; 1 mm quads, one hex layer
    mesh=Mesh(mode="body-fitted", size=1.0, layers=1),
    material=Material(youngs_modulus=210_000.0, poisson_ratio=0.3),  # steel, MPa
    constraints=[Fix("wall")],  # clamped: every displacement component zero
    loads=[Force("tip", (0.0, -1000.0, 0.0))],  # N, total, spread along the line
    optimize=SimpParams(volume_fraction=0.5, penal=3.0, filter_radius=1.5),
)
