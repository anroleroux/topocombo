"""Parametric design domain for the 2D cantilever beam (CadQuery).

The design domain is a planar rectangle in the XY plane with its lower-left
corner at the origin:

    y
    ^
    H +-----------------------------+
      |                             |  <- load region (right edge, mid height)
      |        design domain        |
    0 +-----------------------------+--> x
      0                             L
    ^^^ fixed (left edge, fully clamped)

Only the geometry lives here.  Meshing, boundary conditions and the solve are
downstream stages that consume the exported CAD file, so the geometry can be
re-parameterised without touching them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cadquery as cq


@dataclass(frozen=True)
class BeamDomain:
    """Design domain of a 2D cantilever beam (plane stress).

    Lengths are in millimetres; ``thickness`` is the out-of-plane thickness used
    later by the plane-stress solver, not a modelled dimension.
    """

    length: float = 60.0
    height: float = 20.0
    thickness: float = 1.0

    def __post_init__(self) -> None:
        if self.length <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError("length, height and thickness must be positive")

    @property
    def aspect_ratio(self) -> float:
        return self.length / self.height

    @property
    def area(self) -> float:
        return self.length * self.height

    @property
    def load_point(self) -> tuple[float, float]:
        """Where the tip load is applied: mid-height of the free (right) edge."""
        return (self.length, self.height / 2.0)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            aspect_ratio=self.aspect_ratio,
            area=self.area,
            load_point=list(self.load_point),
        )
        return d

    def face(self) -> cq.Face:
        """The planar face representing the design domain."""
        wire = (
            cq.Workplane("XY")
            .polyline(
                [
                    (0.0, 0.0),
                    (self.length, 0.0),
                    (self.length, self.height),
                    (0.0, self.height),
                ]
            )
            .close()
            .wire()
            .val()
        )
        return cq.Face.makeFromWires(wire)

    def workplane(self) -> cq.Workplane:
        return cq.Workplane("XY").newObject([self.face()])


def export_domain(domain: BeamDomain, out_dir: Path) -> dict[str, Path]:
    """Export the design domain as BREP (for Gmsh) and STEP (for exchange)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    face = domain.face()

    brep = out_dir / "design_domain.brep"
    face.exportBrep(str(brep))

    step = out_dir / "design_domain.step"
    cq.exporters.export(domain.workplane(), str(step))

    return {"brep": brep, "step": step}
