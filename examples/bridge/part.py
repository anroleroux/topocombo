"""The part: a two-span bridge deck, 120 x 20 mm, on three supports.

Pinned at the left end, on rollers at the right end and at mid-span.  With
the middle support the deck is statically indeterminate, so a support that
settles strains it: the study's second load case.  A planar face (2D, plane
stress); runs as-is in CQ-editor.
"""
import cadquery as cq

span = 60.0  # each of the two spans, mm
depth = 20.0

result = cq.Workplane("XY").rect(2 * span, depth, centered=False).extrude(1.0).faces("<Z").val()

regions = {
    "left": cq.Vertex.makeVertex(0, 0, 0),  # pinned
    "middle": cq.Vertex.makeVertex(span, 0, 0),  # the pier that may settle
    "right": cq.Vertex.makeVertex(2 * span, 0, 0),  # on rollers
    "deck": cq.Workplane("XY").add(result).edges(">Y"),  # the road surface
}

if "show_object" in globals():  # CQ-editor
    show_object(result, name="part")
