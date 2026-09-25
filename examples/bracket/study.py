"""The study: the bracket bolted to a wall, carrying 1 kN down through its pin.

    python -m topocombo.cli all --study examples/bracket/study.py

The part is not a prism, so it is meshed with quadratic tetrahedra; the
filter radius is 1.5 element sizes, as SIMP wants.
"""
from topocombo.study import Fix, Force, Material, Mesh, SimpParams, Study

size = 2.5  # mm, tetrahedron edge

study = Study(
    part="part.py",
    mesh=Mesh(element="tet10", size=size),
    material=Material(youngs_modulus=210_000.0, poisson_ratio=0.3),  # steel, MPa
    constraints=[Fix("wall")],  # bolted flat to the wall: clamped
    loads=[Force("pin", (0.0, -1000.0, 0.0))],  # N, total, spread over the bore
    optimize=SimpParams(volume_fraction=0.3, penal=3.0, filter_radius=1.5 * size),
)
