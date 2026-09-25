"""The study: the half MBB beam, held by symmetry and a roller.

    python -m topocombo.cli all --study examples/mbb/study.py

Neither support is a clamp: the symmetry edge holds only ux (points on the
mid-span line may move vertically, not across it) and the roller holds only
uy (it slides along x).  Together they stop every rigid-body motion.  The
load is half the full beam's, since half the beam carries it.
"""
from topocombo.study import Fix, Force, Material, Mesh, SimpParams, Study

study = Study(
    part="part.py",
    mesh=Mesh(mode="structured", nelx=60, nely=20),  # the benchmark's 60 x 20 grid
    material=Material(youngs_modulus=210_000.0, poisson_ratio=0.3),
    constraints=[
        Fix("symmetry", dofs="x"),  # ux = 0 on the symmetry line
        Fix("roller", dofs="y"),  # uy = 0 at the support
    ],
    loads=[Force("load", (0.0, -1000.0))],  # N, half the full beam's 2 kN
    optimize=SimpParams(volume_fraction=0.5, penal=3.0, filter_radius=1.5, max_iterations=150),
    crosscheck=True,  # solve again in CalculiX and compare
)
