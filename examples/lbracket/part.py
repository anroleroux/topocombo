"""The part: the L-bracket of stress-constrained topology optimisation.

A 100 x 100 mm square with its upper-right 60 x 60 mm corner cut away: a
vertical leg clamped along its top edge and a horizontal arm loaded at its
tip.  The re-entrant corner where the two meet concentrates stress —
infinitely, for a sharp corner — which is what makes it the standard test of
stress constraints: a minimum-compliance design keeps the sharp corner, a
stress-limited one rounds it off.  A planar face (2D, plane stress); runs
as-is in CQ-editor.
"""
import cadquery as cq

size = 100.0  # the square, mm
leg = 40.0  # width of the leg and the arm
tip = 10.0  # length of the loaded stretch at the arm's end, from its top

outline = [(0, 0), (size, 0), (size, leg), (leg, leg), (leg, size), (0, size)]
result = cq.Workplane("XY").polyline(outline).close().extrude(1.0).faces("<Z").val()

regions = {
    "top": cq.Workplane("XY").add(result).edges(">Y"),  # clamped to the ceiling
    "tip": cq.Edge.makeLine(cq.Vector(size, leg - tip, 0), cq.Vector(size, leg, 0)),
}

if "show_object" in globals():  # CQ-editor
    show_object(result, name="part")
