"""topocombo — topology optimization from decoupled open-source components.

Currently implemented: the geometry (CadQuery) and meshing (Gmsh) stages of the
main flow described in the README.  The FEA solver and the SIMP optimizer are
future work; this package deliberately stops at a validated quadrilateral mesh
plus the boundary-condition sets the solver will need.
"""

__version__ = "0.1.0"
