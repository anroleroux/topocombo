"""The part: half of an MBB beam — the classic topology-optimisation benchmark.

The full beam (120 x 20 mm) rests on a pin and a roller at its bottom corners
and carries a load at the middle of its top edge.  It is symmetric about
x = 60, so only the right half is modelled, shifted to start at the origin:
the symmetry line is the edge x = 0.  A planar face, so the study is 2D
(plane stress).  Runs as-is in CQ-editor.
"""
import cadquery as cq

length = 60.0  # half span, mm
height = 20.0

result = cq.Workplane("XY").rect(length, height, centered=False).extrude(1.0).faces("<Z").val()

regions = {
    "symmetry": cq.Workplane("XY").add(result).edges("<X"),  # the cut through mid-span
    "roller": cq.Vertex.makeVertex(length, 0.0, 0.0),  # the support, bottom corner
    "load": cq.Vertex.makeVertex(0.0, height, 0.0),  # mid-span, top
}

if "show_object" in globals():  # CQ-editor
    show_object(result, name="part")
