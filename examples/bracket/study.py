"""The study: the bracket bolted to a wall, carrying a pin load two ways.

    python -m topocombo.cli all --study examples/bracket/study.py

The part is not a prism, so it is meshed with quadratic tetrahedra; the
filter radius is 1.5 element sizes, as SIMP wants.

Two load cases — the pin pulled down, and pushed sideways (along z) — are
optimised together: the design minimises the sum of their compliances, so it
must be stiff both ways.  Two passive regions stay solid whatever the
optimizer does: a ring round the pin's bore (material to bear on) and a pad
against the wall (material to bolt through).
"""
from topocombo.study import Fix, Force, LoadCase, Material, Mesh, Passive, SimpParams, Study

size = 2.5  # mm, tetrahedron edge

study = Study(
    part="part.py",
    mesh=Mesh(element="tet10", size=size),
    material=Material(youngs_modulus=210_000.0, poisson_ratio=0.3),  # steel, MPa
    constraints=[Fix("wall")],  # bolted flat to the wall: clamped
    load_cases=[
        LoadCase("hang", [Force("pin", (0.0, -1000.0, 0.0))]),  # N, spread over the bore
        LoadCase("sway", [Force("pin", (0.0, 0.0, -300.0))]),
    ],
    passive=[
        Passive("pin", state="solid", within=2.0),  # a 2 mm ring round the bore
        Passive("wall", state="solid", within=1.5),  # a 1.5 mm mounting pad
    ],
    optimize=SimpParams(volume_fraction=0.3, penal=3.0, filter_radius=1.5 * size),
)
