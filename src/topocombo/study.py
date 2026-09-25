"""A study: everything around the part — mesh, material, constraints, loads
and the optimizer — written as a Python script.

A design is two scripts side by side::

    examples/cantilever/
      part.py     # CadQuery: `result` (the part) and `regions` (named places on it)
      study.py    # `study = Study(part="part.py", ...)`, referring to regions by name

``study.py`` builds a :class:`Study` from the small dataclasses below; each
checks its own fields when constructed, so a mistake fails with the field
named.  :func:`load_study` runs the script, builds the part, and checks that
every region the study names exists on it.  Being Python, a study can loop,
compute and import helpers like any script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .fea import Material
from .geometry import CadDomain
from .meshing import MESH_MODES, MeshSpec, MeshSpec3D
from .optimize import SimpParams

__all__ = ["Study", "Mesh", "Fix", "Force", "Material", "SimpParams", "load_study", "run_study"]

#: Element types per dimension.  Hexahedra are extruded from the x-y profile,
#: so a 3D part must be a prism along z; tetrahedra will lift that.
ELEMENTS = {2: ("quad4",), 3: ("hex8",)}


@dataclass(frozen=True)
class Mesh:
    """How to mesh the part.

    ``body-fitted`` (the default) meshes the real profile at element edge
    ``size`` (default ``min(L / nelx, H / nely)``); ``structured`` lays an
    ``nelx`` x ``nely`` grid over the bounding box and holds the cells outside
    the part void.  In 3D, ``layers`` hexahedra go through the width.
    ``element`` defaults to ``quad4`` in 2D and ``hex8`` in 3D.
    """

    mode: str = "body-fitted"
    size: float | None = None
    nelx: int = 60
    nely: int = 20
    layers: int = 1
    element: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in MESH_MODES:
            raise ValueError(f"Mesh.mode must be one of {', '.join(MESH_MODES)}, not {self.mode!r}")
        if self.size is not None and not self.size > 0:
            raise ValueError(f"Mesh.size must be positive, not {self.size!r}")
        for name in ("nelx", "nely", "layers"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"Mesh.{name} must be a whole number >= 1")
        known = {e for names in ELEMENTS.values() for e in names}
        if self.element is not None and self.element not in known:
            raise ValueError(f"Mesh.element must be one of {', '.join(sorted(known))}")

    def spec(self, dim: int) -> MeshSpec | MeshSpec3D:
        element = self.element or ELEMENTS[dim][0]
        if element not in ELEMENTS[dim]:
            raise ValueError(f"a {dim}D part takes {', '.join(ELEMENTS[dim])} elements, not {element}")
        common = {"nelx": self.nelx, "nely": self.nely, "mode": self.mode, "size": self.size}
        if dim == 3:
            return MeshSpec3D(nelz=self.layers, **common)
        return MeshSpec(**common)


@dataclass(frozen=True)
class Fix:
    """Clamp a region: every displacement component of its nodes is zero."""

    region: str

    def __post_init__(self) -> None:
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("Fix.region must be a region name")

    def as_dict(self) -> dict[str, Any]:
        return {"region": self.region, "type": "clamped"}


@dataclass(frozen=True)
class Force:
    """A total force (N) spread uniformly over a region: on its one node (a
    vertex), by tributary length (edges) or by tributary area (faces)."""

    region: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.region, str) or not self.region:
            raise ValueError("Force.region must be a region name")
        vector = tuple(float(v) for v in self.vector)
        if len(vector) not in (2, 3):
            raise ValueError(f"Force.vector is (Fx, Fy) or (Fx, Fy, Fz), not {self.vector!r}")
        object.__setattr__(self, "vector", vector)

    def as_dict(self) -> dict[str, Any]:
        return {"region": self.region, "vector": list(self.vector)}


@dataclass(frozen=True)
class Study:
    """Everything about a run except the geometry, which ``part`` points to
    (a CadQuery script, relative to the study file)."""

    part: str
    constraints: tuple[Fix, ...]
    loads: tuple[Force, ...]
    mesh: Mesh = field(default_factory=Mesh)
    material: Material = field(default_factory=Material)
    optimize: SimpParams | None = field(default_factory=SimpParams)
    thickness: float = 1.0  # 2D only: plane-stress out-of-plane size, mm

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "loads", tuple(self.loads))
        if not self.constraints or not all(isinstance(c, Fix) for c in self.constraints):
            raise ValueError("Study.constraints needs at least one Fix(...)")
        if not all(isinstance(f, Force) for f in self.loads):
            raise ValueError("Study.loads takes Force(...) entries")
        if len(self.loads) != 1:
            raise ValueError("Study.loads takes exactly one Force(...) for now")
        for name, cls in (("mesh", Mesh), ("material", Material)):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"Study.{name} must be a {cls.__name__}(...)")
        if self.optimize is not None and not isinstance(self.optimize, SimpParams):
            raise ValueError("Study.optimize must be a SimpParams(...) or None")
        if not self.thickness > 0:
            raise ValueError("Study.thickness must be positive")

    @property
    def regions_used(self) -> list[str]:
        names = [c.region for c in self.constraints] + [f.region for f in self.loads]
        return list(dict.fromkeys(names))


@dataclass(frozen=True)
class LoadedStudy:
    """A study with its part built: what the pipeline runs."""

    study: Study
    domain: CadDomain
    path: Path
    source: str

    @property
    def spec(self) -> MeshSpec | MeshSpec3D:
        return self.study.mesh.spec(self.domain.dim)


def load_study(path: Path) -> LoadedStudy:
    """Run ``study.py``, build its part and check the two agree."""
    path = Path(path)
    source = path.read_text()
    namespace: dict[str, Any] = {"__name__": "__study__", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    study = namespace.get("study")
    if not isinstance(study, Study):
        raise ValueError(f"{path}: a study script must assign a Study(...) to `study`")
    part = (path.parent / study.part) if not Path(study.part).is_absolute() else Path(study.part)
    domain = CadDomain.from_file(part, thickness=study.thickness)
    regions = domain.regions()
    missing = [name for name in study.regions_used if name not in regions]
    if missing:
        raise ValueError(
            f"{path}: region(s) {', '.join(missing)} not defined in {part}; "
            f"it defines: {', '.join(sorted(regions)) or 'none'}"
        )
    for force in study.loads:
        if len(force.vector) != domain.dim:
            raise ValueError(
                f"{path}: Force on '{force.region}' has {len(force.vector)} components "
                f"for a {domain.dim}D part"
            )
    study.mesh.spec(domain.dim)  # element type fits the dimension
    return LoadedStudy(study=study, domain=domain, path=path, source=source)


def run_study(path: Path, out_dir: Path, echo: bool = True) -> tuple[Any, dict[str, Any]]:
    """Load ``study.py`` and run the pipeline on it."""
    from .pipeline import run

    loaded = load_study(path)
    s = loaded.study
    return run(
        domain=loaded.domain,
        spec=loaded.spec,
        out_dir=out_dir,
        material=s.material,
        simp=s.optimize or SimpParams(),
        optimize_design=s.optimize is not None,
        constraints=s.constraints,
        loads=s.loads,
        study=loaded,
        echo=echo,
    )
