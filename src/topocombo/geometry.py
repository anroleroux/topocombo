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

Both domains take optional circular cutouts through z, as (x, y, diameter)
``holes``.  They are real features of the CAD model (the script cuts them), but
the structured mesh still covers the whole L x H envelope — Gmsh meshes
:meth:`BeamDomain.envelope` — and the elements whose centroids fall inside a
cutout are held void by the optimizer (:meth:`BeamDomain.void_mask`), the usual
SIMP treatment of a non-design region.

Only the geometry lives here.  Meshing, boundary conditions and the solve are
downstream stages that consume the exported CAD file, so the geometry can be
re-parameterised without touching them.

The CadQuery input is explicit: each domain writes itself out as a standalone
CadQuery script (:meth:`BeamDomain.cadquery_script`), and the shape is built by
running exactly that script.  The script is exported next to the BREP/STEP
files and shown in the report, so what went into CadQuery is never implicit —
and it opens as-is in CQ-editor.

:class:`CadDomain` takes the CadQuery input as code: any script that assigns
the part to ``result`` — a planar face in the x-y plane (2D) or a solid
prismatic along z (3D).  Nothing downstream reads parameters from it: the
extent, the material area or volume, the profile Gmsh meshes and which grid
cells lie outside the part all come from the shape the script built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
from typing import Any

import numpy as np

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
    holes: tuple[tuple[float, float, float], ...] = field(default=())

    def __post_init__(self) -> None:
        if self.length <= 0 or self.height <= 0 or self.thickness <= 0:
            raise ValueError("length, height and thickness must be positive")
        object.__setattr__(self, "holes", _check_holes(self.holes, self.length, self.height))

    dim = 2

    @property
    def has_cutouts(self) -> bool:
        return bool(self.holes)

    @property
    def aspect_ratio(self) -> float:
        return self.length / self.height

    @property
    def area(self) -> float:
        """Area of the L x H envelope — the meshed region, cutouts included."""
        return self.length * self.height

    @property
    def material_area(self) -> float:
        """Area of the CAD face: the envelope less the cutouts."""
        return self.area - _holes_area(self.holes)

    def envelope(self) -> BeamDomain:
        """The same domain without cutouts: what Gmsh meshes."""
        return replace(self, holes=())

    def void_mask(self, centroids: np.ndarray) -> np.ndarray:
        """True for elements whose centroid lies inside a cutout."""
        return _in_holes(self.holes, centroids)

    @property
    def load_point(self) -> tuple[float, float]:
        """Where the tip load is applied: mid-height of the free (right) edge."""
        return (self.length, self.height / 2.0)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            aspect_ratio=self.aspect_ratio,
            area=self.area,
            material_area=self.material_area,
            load_point=list(self.load_point),
            holes=[list(h) for h in self.holes],
        )
        return d

    def cadquery_script(self) -> str:
        """The standalone CadQuery script that builds this domain as ``result``."""
        if self.holes:
            face = _FACE_WITH_HOLES_2D.format(holes=_holes_literal(self.holes))
        else:
            face = _FACE_2D
        return _SCRIPT_2D.format(
            length=float(self.length),
            height=float(self.height),
            thickness=float(self.thickness),
            face=face,
        )

    def workplane(self) -> cq.Workplane:
        """The domain as built by running :meth:`cadquery_script`."""
        return run_cadquery_script(self.cadquery_script())

    def face(self) -> cq.Face:
        """The planar face representing the design domain."""
        face = self.workplane().val()
        if not isinstance(face, cq.Face):  # pragma: no cover - script invariant
            raise TypeError(f"the 2D CadQuery script built a {type(face).__name__}, not a Face")
        return face


@dataclass(frozen=True)
class BeamDomain3D:
    """Design domain of a 3D cantilever beam: a solid box, lengths in mm."""

    length: float = 60.0
    height: float = 20.0
    width: float = 1.0
    holes: tuple[tuple[float, float, float], ...] = field(default=())

    def __post_init__(self) -> None:
        if self.length <= 0 or self.height <= 0 or self.width <= 0:
            raise ValueError("length, height and width must be positive")
        object.__setattr__(self, "holes", _check_holes(self.holes, self.length, self.height))

    dim = 3

    @property
    def has_cutouts(self) -> bool:
        return bool(self.holes)

    @property
    def aspect_ratio(self) -> float:
        return self.length / self.height

    @property
    def volume(self) -> float:
        """Volume of the L x H x W envelope — the meshed region, cutouts included."""
        return self.length * self.height * self.width

    @property
    def material_volume(self) -> float:
        """Volume of the CAD solid: the envelope less the cutouts."""
        return self.volume - _holes_area(self.holes) * self.width

    def envelope(self) -> BeamDomain3D:
        """The same domain without cutouts: what Gmsh meshes."""
        return replace(self, holes=())

    def void_mask(self, centroids: np.ndarray) -> np.ndarray:
        """True for elements whose centroid lies inside a cutout (through z)."""
        return _in_holes(self.holes, centroids)

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
            material_volume=self.material_volume,
            load_line=[list(p) for p in self.load_line],
            load_point=list(self.load_point),
            holes=[list(h) for h in self.holes],
        )
        return d

    def cadquery_script(self) -> str:
        """The standalone CadQuery script that builds this domain as ``result``."""
        cutouts = _CUTOUTS_3D.format(holes=_holes_literal(self.holes)) if self.holes else ""
        return _SCRIPT_3D.format(
            length=float(self.length),
            height=float(self.height),
            width=float(self.width),
            cutouts=cutouts,
        )

    def workplane(self) -> cq.Workplane:
        """The domain as built by running :meth:`cadquery_script`."""
        return run_cadquery_script(self.cadquery_script())

    def solid(self) -> cq.Solid:
        """The box representing the design domain, lower corner at the origin."""
        solid = self.workplane().val()
        if not isinstance(solid, cq.Solid):  # pragma: no cover - script invariant
            raise TypeError(f"the 3D CadQuery script built a {type(solid).__name__}, not a Solid")
        return solid

    def profile(self) -> cq.Face:
        """The x-y cross-section at z = 0 — what the body-fitted mesh extrudes."""
        return BeamDomain(length=self.length, height=self.height, holes=self.holes).face()


#: Relative tolerance on the part's placement and on the prism check.
_CAD_RTOL = 1e-6


@dataclass(frozen=True)
class CadDomain:
    """A design domain given as code: any CadQuery script that assigns the part
    to ``result``.

    The shape decides the dimension: a planar face in the x-y plane is a 2D
    plane-stress domain (``thickness`` is the solver's out-of-plane size), a
    solid is a 3D domain, which must be a prism along z because the hex mesh
    is an extrusion of its x-y profile.  The part's bounding box must start at
    the origin; the cantilever conventions then hold as for :class:`BeamDomain`:
    the part's boundary at x = 0 is clamped and the load acts at mid-height of
    x = L, where the part must have an edge (2D) or face (3D).

    ``length``, ``height`` and ``width`` are the bounding box; ``area`` /
    ``volume`` are the envelope's and ``material_area`` / ``material_volume``
    the shape's own, so whatever the script cut away counts as a cutout.
    """

    source: str
    path: str | None = None
    thickness: float = 1.0
    _shape: cq.Shape = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.thickness > 0:
            raise ValueError("thickness must be positive")
        where = self.path or "the CadQuery script"
        shape = _single_shape(run_cadquery_script(self.source, self.path or "design_domain.py"), where)
        bb = shape.BoundingBox()
        size = max(bb.xlen, bb.ylen, bb.zlen)
        if max(abs(bb.xmin), abs(bb.ymin), abs(bb.zmin)) > _CAD_RTOL * size:
            raise ValueError(
                f"{where}: the part's bounding box must start at the origin, not at "
                f"({bb.xmin:g}, {bb.ymin:g}, {bb.zmin:g}); move it with .translate()"
            )
        object.__setattr__(self, "_shape", shape)
        if isinstance(shape, cq.Face):
            if bb.zlen > _CAD_RTOL * size:
                raise ValueError(f"{where}: a 2D part must be a face in the x-y plane (z = 0)")
        profile = self.profile()
        if isinstance(shape, cq.Solid):
            prism = profile.Area() * bb.zlen
            if abs(prism - shape.Volume()) > _CAD_RTOL * prism:
                raise ValueError(
                    f"{where}: a 3D part must be a prism along z (its z = 0 face swept "
                    f"through the width): the profile gives {prism:.6g} mm^3, the solid "
                    f"has {shape.Volume():.6g} mm^3"
                )
        _check_cantilever_edges(profile, bb.xlen, bb.ylen, where)

    @property
    def dim(self) -> int:
        return 3 if isinstance(self._shape, cq.Solid) else 2

    @property
    def length(self) -> float:
        return self._shape.BoundingBox().xlen

    @property
    def height(self) -> float:
        return self._shape.BoundingBox().ylen

    @property
    def width(self) -> float:
        """Extent along z; 0 for a 2D part."""
        return self._shape.BoundingBox().zlen if self.dim == 3 else 0.0

    @property
    def holes(self) -> tuple[tuple[float, float, float], ...]:
        """No parametric cutouts: whatever the script removed is in the shape."""
        return ()

    @property
    def aspect_ratio(self) -> float:
        return self.length / self.height

    @property
    def area(self) -> float:
        """Area of the L x H envelope."""
        return self.length * self.height

    @property
    def material_area(self) -> float:
        """Area of the part's x-y profile."""
        return self.profile().Area()

    @property
    def volume(self) -> float:
        return self.area * self.width

    @property
    def material_volume(self) -> float:
        return self._shape.Volume() if self.dim == 3 else 0.0

    @property
    def has_cutouts(self) -> bool:
        """True when the part does not fill its bounding box."""
        return self.material_area < self.area * (1 - _CAD_RTOL)

    def envelope(self) -> BeamDomain | BeamDomain3D:
        """The bounding box as a plain beam: what the structured grid covers."""
        if self.dim == 3:
            return BeamDomain3D(length=self.length, height=self.height, width=self.width)
        return BeamDomain(length=self.length, height=self.height, thickness=self.thickness)

    def void_mask(self, centroids: np.ndarray) -> np.ndarray:
        """True for elements whose centroid lies outside the part's x-y profile."""
        return ~_in_face(self.profile(), centroids)

    @property
    def load_point(self) -> tuple[float, ...]:
        if self.dim == 3:
            return (self.length, self.height / 2.0, self.width / 2.0)
        return (self.length, self.height / 2.0)

    @property
    def load_line(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (self.length, self.height / 2.0, 0.0), (self.length, self.height / 2.0, self.width)

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "source": "script",
            "script_path": self.path,
            "length": self.length,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "area": self.area,
            "material_area": self.material_area,
            "load_point": list(self.load_point),
            "holes": [],
        }
        if self.dim == 3:
            d.update(
                width=self.width,
                volume=self.volume,
                material_volume=self.material_volume,
                load_line=[list(p) for p in self.load_line],
            )
        else:
            d["thickness"] = self.thickness
        return d

    def cadquery_script(self) -> str:
        return self.source

    def workplane(self) -> cq.Workplane:
        return cq.Workplane("XY").add(self._shape)

    def face(self) -> cq.Face:
        if self.dim != 2:
            raise TypeError("a 3D part has no face(); use solid() or profile()")
        return self._shape

    def solid(self) -> cq.Solid:
        if self.dim != 3:
            raise TypeError("a 2D part has no solid(); use face()")
        return self._shape

    def profile(self) -> cq.Face:
        """The part's x-y cross-section at z = 0 (the face itself in 2D)."""
        if isinstance(self._shape, cq.Face):
            return self._shape
        faces = [
            f for f in self._shape.Faces()
            if f.BoundingBox().zlen <= _CAD_RTOL * self.length and abs(f.Center().z) <= _CAD_RTOL * self.length
        ]
        if len(faces) != 1:
            raise ValueError(
                f"{self.path or 'the CadQuery script'}: expected one planar face at z = 0, "
                f"found {len(faces)}"
            )
        return faces[0]

    @classmethod
    def from_file(cls, path: Path, thickness: float = 1.0) -> CadDomain:
        path = Path(path)
        return cls(source=path.read_text(), path=str(path), thickness=thickness)


def _single_shape(result: cq.Workplane, where: str) -> cq.Face | cq.Solid:
    """The one face or solid a script built, unwrapped from any compound."""
    shapes: list[cq.Shape] = []
    for obj in result.vals():
        if isinstance(obj, cq.Compound):
            shapes.extend(obj.Solids() or obj.Faces())
        elif isinstance(obj, cq.Shape):
            shapes.append(obj)
    if len(shapes) != 1 or not isinstance(shapes[0], (cq.Face, cq.Solid)):
        kinds = ", ".join(type(s).__name__ for s in shapes) or "nothing"
        raise ValueError(
            f"{where}: `result` must be one connected face (2D) or solid (3D), not {kinds}"
        )
    return shapes[0]


def _check_cantilever_edges(profile: cq.Face, length: float, height: float, where: str) -> None:
    """The profile needs a boundary at x = 0 to clamp and one at x = L that
    passes through mid-height, where the load acts."""
    tol = _CAD_RTOL * max(length, height)
    at_x = {"x = 0": 0.0, "x = L": length}
    found = {name: [] for name in at_x}
    for edge in profile.Edges():
        bb = edge.BoundingBox()
        for name, x in at_x.items():
            if abs(bb.xmin - x) <= tol and abs(bb.xmax - x) <= tol:
                found[name].append((bb.ymin, bb.ymax))
    if not found["x = 0"]:
        raise ValueError(f"{where}: the part has no straight edge at x = 0 to clamp")
    if not any(y0 - tol <= height / 2 <= y1 + tol for y0, y1 in found["x = L"]):
        raise ValueError(
            f"{where}: the part has no edge at x = L = {length:g} through the load point "
            f"at mid-height y = {height / 2:g}"
        )


def _in_face(face: cq.Face, points: np.ndarray) -> np.ndarray:
    """True for the (x, y[, z]) points whose x-y position lies inside ``face``."""
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN

    points = np.asarray(points, dtype=float)
    classifier = BRepClass_FaceClassifier()
    inside = np.zeros(points.shape[0], dtype=bool)
    for i, (x, y) in enumerate(points[:, :2]):
        classifier.Perform(face.wrapped, gp_Pnt(float(x), float(y), 0.0), 1e-9)
        inside[i] = classifier.State() == TopAbs_IN
    return inside


_SCRIPT_2D = '''"""Design domain of a 2D cantilever beam (plane stress), written by topocombo.

Runs as-is in CQ-editor or with plain Python; the pipeline builds the domain by
running this script; Gmsh meshes its L x H envelope and the optimizer holds
any cutouts void.
"""
import cadquery as cq

# parameters, in mm
length = {length!r}
height = {height!r}
thickness = {thickness!r}  # plane-stress thickness: used by the solver, not modelled

# a planar rectangle in the XY plane, lower-left corner at the origin;
# the x = 0 edge is clamped, the load acts at (length, height / 2)
outline = (
    cq.Workplane("XY")
    .polyline([(0.0, 0.0), (length, 0.0), (length, height), (0.0, height)])
    .close()
    .wire()
    .val()
)
{face}
if "show_object" in globals():  # CQ-editor
    show_object(result, name="design_domain")
'''

_SCRIPT_3D = '''"""Design domain of a 3D cantilever beam (solid box), written by topocombo.

Runs as-is in CQ-editor or with plain Python; the pipeline builds the domain by
running this script; Gmsh meshes its L x H envelope and the optimizer holds
any cutouts void.
"""
import cadquery as cq

# parameters, in mm
length = {length!r}  # along x
height = {height!r}  # along y
width = {width!r}  # along z

# a box with its lower corner at the origin; the x = 0 face is clamped, the
# load acts along the line (length, height / 2, 0 .. width)
result = cq.Workplane("XY").box(length, height, width, centered=False)
{cutouts}
if "show_object" in globals():  # CQ-editor
    show_object(result, name="design_domain")
'''


_FACE_2D = '''result = cq.Workplane("XY").add(cq.Face.makeFromWires(outline))
'''

_FACE_WITH_HOLES_2D = '''
# circular cutouts through the thickness, as (x, y, diameter)
holes = {holes}
circles = [
    cq.Wire.makeCircle(d / 2, cq.Vector(x, y, 0), cq.Vector(0, 0, 1)) for x, y, d in holes
]
result = cq.Workplane("XY").add(cq.Face.makeFromWires(outline, circles))
'''

_CUTOUTS_3D = '''
# circular cutouts through the width (along z), as (x, y, diameter)
holes = {holes}
for x, y, d in holes:
    cutter = cq.Workplane("XY").center(x, y).circle(d / 2).extrude(width + 1.0, both=True)
    result = result.cut(cutter)
result = result.solids()
'''


def _holes_literal(holes: tuple[tuple[float, float, float], ...]) -> str:
    return "[" + ", ".join(f"({x!r}, {y!r}, {d!r})" for x, y, d in holes) + "]"


def _check_holes(holes: Any, length: float, height: float) -> tuple[tuple[float, float, float], ...]:
    """Normalise cutouts to float triples; each must sit inside the domain and
    clear of the others, so the CAD face or solid stays one connected piece."""
    out: list[tuple[float, float, float]] = []
    for hole in holes:
        if len(hole) != 3:
            raise ValueError(f"a hole is (x, y, diameter), not {hole!r}")
        x, y, d = (float(v) for v in hole)
        if d <= 0:
            raise ValueError(f"hole diameter must be positive, not {d:g}")
        r = d / 2
        if not (r < x < length - r and r < y < height - r):
            raise ValueError(
                f"hole at ({x:g}, {y:g}) with diameter {d:g} does not fit strictly inside "
                f"the {length:g} x {height:g} domain"
            )
        for ox, oy, od in out:
            if math.hypot(x - ox, y - oy) <= r + od / 2:
                raise ValueError(f"holes at ({x:g}, {y:g}) and ({ox:g}, {oy:g}) overlap")
        out.append((x, y, d))
    return tuple(out)


def _holes_area(holes: tuple[tuple[float, float, float], ...]) -> float:
    return sum(math.pi * d * d / 4 for _, _, d in holes)


def _in_holes(holes: tuple[tuple[float, float, float], ...], centroids: np.ndarray) -> np.ndarray:
    centroids = np.asarray(centroids, dtype=float)
    mask = np.zeros(centroids.shape[0], dtype=bool)
    for x, y, d in holes:
        mask |= np.hypot(centroids[:, 0] - x, centroids[:, 1] - y) < d / 2
    return mask


def run_cadquery_script(source: str, filename: str = "design_domain.py") -> cq.Workplane:
    """Execute a CadQuery script and return the ``result`` it defines (a
    ``cq.Shape`` is wrapped in a workplane)."""
    namespace: dict[str, Any] = {"__name__": "__cadquery_script__"}
    exec(compile(source, filename, "exec"), namespace)
    result = namespace.get("result")
    if isinstance(result, cq.Shape):
        result = cq.Workplane("XY").add(result)
    if not isinstance(result, cq.Workplane):
        raise ValueError("a CadQuery script must assign a cq.Workplane or cq.Shape to `result`")
    return result


#: Anything the pipeline can mesh.
Domain = BeamDomain | BeamDomain3D | CadDomain


def export_domain(domain: Domain, out_dir: Path) -> dict[str, Path]:
    """Export the design domain as its CadQuery script, BREP and STEP.

    With cutouts, the envelope Gmsh meshes is written too, as
    ``design_envelope.brep`` (key ``envelope``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = out_dir / "design_domain.py"
    script.write_text(domain.cadquery_script())

    workplane = domain.workplane()
    brep = out_dir / "design_domain.brep"
    workplane.val().exportBrep(str(brep))

    step = out_dir / "design_domain.step"
    cq.exporters.export(workplane, str(step))

    exported = {"script": script, "brep": brep, "step": step}
    if domain.has_cutouts:
        envelope = out_dir / "design_envelope.brep"
        domain.envelope().workplane().val().exportBrep(str(envelope))
        exported["envelope"] = envelope
    return exported
