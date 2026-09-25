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
and it opens as-is in CQ-editor.  :func:`load_cad_config` reads the domain
parameters from a JSON or TOML file, rejecting any key it does not know.
"""

from __future__ import annotations

import json
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


def run_cadquery_script(source: str) -> cq.Workplane:
    """Execute a CadQuery script and return the ``result`` it defines."""
    namespace: dict[str, Any] = {"__name__": "__cadquery_script__"}
    exec(compile(source, "design_domain.py", "exec"), namespace)
    result = namespace.get("result")
    if not isinstance(result, cq.Workplane):
        raise ValueError("a CadQuery script must assign a cq.Workplane to `result`")
    return result


def export_domain(domain: BeamDomain | BeamDomain3D, out_dir: Path) -> dict[str, Path]:
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
    if domain.holes:
        envelope = out_dir / "design_envelope.brep"
        domain.envelope().workplane().val().exportBrep(str(envelope))
        exported["envelope"] = envelope
    return exported


# --------------------------------------------------------------------------
# CAD parameters from a file
# --------------------------------------------------------------------------
#: Keys a CAD config file may set; everything else is rejected so a typo never
#: silently falls back to a default.
CAD_CONFIG_KEYS = ("dim", "length", "height", "thickness", "width", "holes")


def load_cad_config(path: Path) -> dict[str, Any]:
    """Read CadQuery domain parameters from a ``.json`` or ``.toml`` file.

    The file is flat, or keeps the parameters under a ``[cad]`` table /
    ``"cad"`` object.  Lengths must be positive numbers and ``dim`` 2 or 3;
    unknown keys raise :class:`ValueError`, naming the ones that are allowed.
    ``holes`` is a list of ``{x, y, diameter}`` tables (``[[cad.holes]]`` in
    TOML) or ``[x, y, diameter]`` triples; whether they fit is checked when the
    domain is built.
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - Python 3.10
            raise ValueError(f"{path}: TOML needs Python 3.11+; use a .json file") from None
        data = tomllib.loads(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"{path}: CAD config must be a .json or .toml file")
    if isinstance(data, dict) and isinstance(data.get("cad"), dict):
        data = data["cad"]
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a table of CAD parameters")

    unknown = sorted(set(data) - set(CAD_CONFIG_KEYS))
    if unknown:
        raise ValueError(
            f"{path}: unknown CAD parameter(s) {', '.join(unknown)}; "
            f"allowed: {', '.join(CAD_CONFIG_KEYS)}"
        )
    config: dict[str, Any] = {}
    for key, value in data.items():
        if key == "holes":
            config[key] = _config_holes(path, value)
            continue
        if key == "dim":
            if value not in (2, 3) or isinstance(value, bool):
                raise ValueError(f"{path}: dim must be 2 or 3, not {value!r}")
            config[key] = int(value)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: {key} must be a number, not {value!r}")
        if not value > 0:
            raise ValueError(f"{path}: {key} must be positive, not {value!r}")
        config[key] = float(value)
    return config


def _config_holes(path: Path, value: Any) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: holes must be a list, not {value!r}")
    holes = []
    for item in value:
        if isinstance(item, dict):
            unknown = sorted(set(item) - {"x", "y", "diameter"})
            missing = sorted({"x", "y", "diameter"} - set(item))
            if unknown or missing:
                raise ValueError(
                    f"{path}: a hole takes x, y and diameter"
                    + (f"; unknown: {', '.join(unknown)}" if unknown else "")
                    + (f"; missing: {', '.join(missing)}" if missing else "")
                )
            item = [item["x"], item["y"], item["diameter"]]
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"{path}: a hole is {{x, y, diameter}} or [x, y, diameter], not {item!r}")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in item):
            raise ValueError(f"{path}: hole values must be numbers, not {item!r}")
        holes.append(tuple(float(v) for v in item))
    return tuple(holes)
