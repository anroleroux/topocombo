"""The part: a 60 x 20 x 1 mm cantilever with a notch in its bottom edge.

Runs as-is in CQ-editor.  The pipeline reads two names from it:

  result   the part — a face in the x-y plane (2D) or a solid prismatic
           along z (3D), with its bounding box starting at the origin
  regions  named places on the part where the study (study.py) puts its
           constraints and loads: vertices, edges or faces, picked from the
           part with selectors or built on their own
"""
import cadquery as cq

# parameters, in mm
length = 60.0  # along x
height = 20.0  # along y
width = 1.0  # along z

# circular cutouts through z, as (x, y, diameter); this one sits on the bottom
# edge and bites a half-circle notch out of the beam
holes = [(20.0, 0.0, 10.0)]

result = cq.Workplane("XY").box(length, height, width, centered=False)
for x, y, d in holes:
    cutter = cq.Workplane("XY").center(x, y).circle(d / 2).extrude(width + 1.0, both=True)
    result = result.cut(cutter)

regions = {
    # the whole end face at x = 0
    "wall": result.faces("<X"),
    # a line across the free end at mid-height, where the tip load acts; it is
    # not an edge of the part, so it is built here and the mesh gets nodes on it
    "tip": cq.Edge.makeLine(cq.Vector(length, height / 2, 0), cq.Vector(length, height / 2, width)),
}

if "show_object" in globals():  # CQ-editor
    show_object(result, name="part")
    for name, region in regions.items():
        show_object(region, name=name)
