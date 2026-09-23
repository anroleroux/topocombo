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

:class:`BeamDomain3D` is the same beam as a solid box, with the current
out-of-plane thickness modelled as a real ``width`` along z:

    z = 0 face is the front, z = W the back; x and y are as above.
    The tip load is a line at mid-height of the free end (x = L, y = H/2),
    running across the full width.

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


@dataclass(frozen=True)
class BeamDomain3D:
    """Design domain of a 3D cantilever beam: a solid box, lengths in mm."""

    length: float = 60.0
    height: float = 20.0
    width: float = 1.0

    def __post_init__(self) -> None:
        if self.length <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("length, height and width must be positive")

    @property
    def aspect_ratio(self) -> float:
        return self.length / self.height

    @property
    def volume(self) -> float:
        return self.length * self.height * self.width

    @property
    def load_line(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """End points of the tip load: mid-height of the free end, across the width."""
        return (self.length, self.height / 2.0, 0.0), (self.length, self.height / 2.0, self.width)

    @property
    def load_point(self) -> tuple[float, float, float]:
        """Centre of the load line — the resultant's point of application."""
        return (self.length, self.height / 2.0, self.width / 2.0)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            aspect_ratio=self.aspect_ratio,
            volume=self.volume,
            load_line=[list(p) for p in self.load_line],
            load_point=list(self.load_point),
        )
        return d

    def solid(self) -> cq.Solid:
        """The box representing the design domain, lower corner at the origin."""
        return cq.Solid.makeBox(self.length, self.height, self.width)

    def workplane(self) -> cq.Workplane:
        return cq.Workplane("XY").newObject([self.solid()])


def _shape(domain: BeamDomain | BeamDomain3D) -> cq.Shape:
    return domain.solid() if isinstance(domain, BeamDomain3D) else domain.face()


def export_domain(domain: BeamDomain | BeamDomain3D, out_dir: Path) -> dict[str, Path]:
    """Export the design domain as BREP (for Gmsh) and STEP (for exchange)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shape = _shape(domain)

    brep = out_dir / "design_domain.brep"
    shape.exportBrep(str(brep))

    step = out_dir / "design_domain.step"
    cq.exporters.export(domain.workplane(), str(step))

    return {"brep": brep, "step": step}
