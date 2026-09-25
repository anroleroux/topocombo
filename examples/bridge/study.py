"""The study: a bridge deck stiff under traffic, with or without its pier.

    python -m topocombo.cli all --study examples/bridge/study.py

Two load cases on the same deck load.  In 'traffic' the middle pier holds;
in 'settled' it has sunk 0.05 mm — a constraint of that case alone, so the
design must also carry the deck across the settled pier (the support then
takes less, and does work on the deck).  The optimizer minimises the sum of
the two compliances.

MMA with Heaviside projection: the density filter's grey edges are pushed
to solid or void as the projection sharpens (beta 1 -> 16).
"""
from topocombo.study import (
    Displace, Fix, Force, LoadCase, Material, Mesh, SimpParams, Study,
)

traffic = Force("deck", (0.0, -2000.0))  # N, spread along the deck

study = Study(
    part="part.py",
    mesh=Mesh(mode="structured", nelx=120, nely=20),  # 1 mm squares
    material=Material(youngs_modulus=210_000.0, poisson_ratio=0.3),
    thickness=5.0,  # mm, plane stress
    constraints=[Fix("left"), Fix("right", dofs="y")],  # in every case
    load_cases=[
        LoadCase("traffic", [traffic], constraints=[Fix("middle", dofs="y")]),
        LoadCase("settled", [traffic], constraints=[Displace("middle", uy=-0.05)]),
    ],
    optimize=SimpParams(
        volume_fraction=0.4, filter_radius=2.0, optimizer="mma",
        projection=16.0, max_iterations=400,
    ),
)
