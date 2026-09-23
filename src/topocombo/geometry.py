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

The CadQuery input is explicit: each domain writes itself out as a standalone
CadQuery script (:meth:`BeamDomain.cadquery_script`), and the shape is built by
running exactly that script.  The script is exported next to the BREP/STEP
files and shown in the report, so what went into CadQuery is never implicit —
and it opens as-is in CQ-editor.  :func:`load_cad_config` reads the domain
parameters from a JSON or TOML file, rejecting any key it does not know.
"""

from __future__ import annotations

import json
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

    def cadquery_script(self) -> str:
        """The standalone CadQuery script that builds this domain as ``result``."""
        return _SCRIPT_2D.format(
            length=float(self.length), height=float(self.height), thickness=float(self.thickness)
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

    def cadquery_script(self) -> str:
        """The standalone CadQuery script that builds this domain as ``result``."""
        return _SCRIPT_3D.format(
            length=float(self.length), height=float(self.height), width=float(self.width)
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
running this script and meshes ``result``.
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
result = cq.Workplane("XY").add(cq.Face.makeFromWires(outline))

if "show_object" in globals():  # CQ-editor
    show_object(result, name="design_domain")
'''

_SCRIPT_3D = '''"""Design domain of a 3D cantilever beam (solid box), written by topocombo.

Runs as-is in CQ-editor or with plain Python; the pipeline builds the domain by
running this script and meshes ``result``.
"""
import cadquery as cq

# parameters, in mm
length = {length!r}  # along x
height = {height!r}  # along y
width = {width!r}  # along z

# a box with its lower corner at the origin; the x = 0 face is clamped, the
# load acts along the line (length, height / 2, 0 .. width)
result = cq.Workplane("XY").box(length, height, width, centered=False)

if "show_object" in globals():  # CQ-editor
    show_object(result, name="design_domain")
'''


def run_cadquery_script(source: str) -> cq.Workplane:
    """Execute a CadQuery script and return the ``result`` it defines."""
    namespace: dict[str, Any] = {"__name__": "__cadquery_script__"}
    exec(compile(source, "design_domain.py", "exec"), namespace)
    result = namespace.get("result")
    if not isinstance(result, cq.Workplane):
        raise ValueError("a CadQuery script must assign a cq.Workplane to `result`")
    return result


def export_domain(domain: BeamDomain | BeamDomain3D, out_dir: Path) -> dict[str, Path]:
    """Export the design domain as its CadQuery script, BREP (for Gmsh) and STEP."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    script = out_dir / "design_domain.py"
    script.write_text(domain.cadquery_script())

    workplane = domain.workplane()
    brep = out_dir / "design_domain.brep"
    workplane.val().exportBrep(str(brep))

    step = out_dir / "design_domain.step"
    cq.exporters.export(workplane, str(step))

    return {"script": script, "brep": brep, "step": step}


# --------------------------------------------------------------------------
# CAD parameters from a file
# --------------------------------------------------------------------------
#: Keys a CAD config file may set; everything else is rejected so a typo never
#: silently falls back to a default.
CAD_CONFIG_KEYS = ("dim", "length", "height", "thickness", "width")


def load_cad_config(path: Path) -> dict[str, Any]:
    """Read CadQuery domain parameters from a ``.json`` or ``.toml`` file.

    The file is flat, or keeps the parameters under a ``[cad]`` table /
    ``"cad"`` object.  Lengths must be positive numbers and ``dim`` 2 or 3;
    unknown keys raise :class:`ValueError`, naming the ones that are allowed.
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

