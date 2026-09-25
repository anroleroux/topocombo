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

from .elements import BY_NAME as _BY_NAME, ELEMENTS as _REGISTRY
from .fea import SOLVERS, Material
from .geometry import CadDomain
from .meshing import MESH_MODES, MeshSpec, MeshSpec3D
from .optimize import SimpParams

__all__ = [
    "Study", "Mesh", "Fix", "Displace", "Force", "LoadCase", "Passive", "Material", "SimpParams",
    "ComplianceLimit", "DisplacementLimit", "StressLimit",
    "load_study", "run_study", "run_loaded",
]

#: Element names per dimension, from the element registry.  Hexahedra are
#: extruded from the x-y profile, so a 3D part must be a prism along z;
#: tetrahedra will lift that.
ELEMENTS = {
    dim: tuple(e.name for e in _REGISTRY.values() if e.dim == dim) for dim in (2, 3)
}


@dataclass(frozen=True)
class Mesh:
    """How to mesh the part.

    ``body-fitted`` (the default) meshes the real profile at element edge
    ``size`` (default ``min(L / nelx, H / nely)``); ``structured`` lays an
    ``nelx`` x ``nely`` grid over the bounding box and holds the cells outside
    the part void.  ``element`` defaults to ``quad4`` in 2D and ``hex8`` in
    3D, where ``layers`` hexahedra go through the width of a prismatic part.
    ``tet10`` (quadratic, recommended) or ``tet4`` (linear, stiff in bending)
    mesh any 3D part with tetrahedra of edge ``size``, body-fitted only.
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
            return MeshSpec3D(nelz=self.layers, element=_BY_NAME[element], **common)
        return MeshSpec(**common)


def _region_name(cls: str, region: Any) -> None:
    if not isinstance(region, str) or not region:
        raise ValueError(f"{cls}.region must be a region name")


@dataclass(frozen=True)
class Fix:
    """Hold displacement components of a region's nodes at zero.

    ``dofs`` picks the components — ``"xyz"`` (the default; in 2D, ``"xy"``)
    clamps; ``"x"`` alone is a symmetry plane normal to x, ``"y"`` a roller
    that slides along x, and so on.
    """

    region: str
    dofs: str = "xyz"

    def __post_init__(self) -> None:
        _region_name("Fix", self.region)
        if not self.dofs or set(self.dofs) - set("xyz") or len(set(self.dofs)) != len(self.dofs):
            raise ValueError(f"Fix.dofs takes distinct letters from 'xyz', not {self.dofs!r}")

    def components(self, dim: int) -> dict[int, float]:
        """Component index -> held value (zero) for a ``dim``-D part."""
        axes = "xyz"[:dim]
        if self.dofs != "xyz" and set(self.dofs) - set(axes):
            raise ValueError(f"Fix('{self.region}', dofs='{self.dofs}'): a 2D part has no z")
        return {axes.index(a): 0.0 for a in axes if a in self.dofs}

    def as_dict(self) -> dict[str, Any]:
        return {"region": self.region, "type": "clamped" if self.dofs == "xyz" else "held",
                "dofs": self.dofs}


@dataclass(frozen=True)
class Displace:
    """Prescribe displacement components of a region's nodes (mm); components
    left as None are free.  A support that settles, a part pushed into shape."""

    region: str
    ux: float | None = None
    uy: float | None = None
    uz: float | None = None

    def __post_init__(self) -> None:
        _region_name("Displace", self.region)
        if self.ux is None and self.uy is None and self.uz is None:
            raise ValueError("Displace needs at least one of ux, uy, uz")

    def components(self, dim: int) -> dict[int, float]:
        values = {0: self.ux, 1: self.uy, 2: self.uz}
        if dim == 2 and self.uz is not None:
            raise ValueError(f"Displace('{self.region}', uz=...): a 2D part has no z")
        return {i: float(v) for i, v in values.items() if v is not None and i < dim}

    def as_dict(self) -> dict[str, Any]:
        return {"region": self.region, "type": "prescribed",
                "dofs": "".join(a for a, v in zip("xyz", (self.ux, self.uy, self.uz)) if v is not None)}


@dataclass(frozen=True)
class Force:
    """A total force (N) spread uniformly over a region: on its one node (a
    vertex), by tributary length (edges) or by tributary area (faces)."""

    region: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        _region_name("Force", self.region)
        vector = tuple(float(v) for v in self.vector)
        if len(vector) not in (2, 3):
            raise ValueError(f"Force.vector is (Fx, Fy) or (Fx, Fy, Fz), not {self.vector!r}")
        object.__setattr__(self, "vector", vector)

    def as_dict(self) -> dict[str, Any]:
        return {"region": self.region, "vector": list(self.vector)}


@dataclass(frozen=True)
class LoadCase:
    """Forces that act together.  With several cases the optimizer minimises
    the weighted sum of their compliances: a design stiff for each.

    ``constraints`` (:class:`Fix`, :class:`Displace`) hold in this case only,
    on top of the study's: a support that settles, a jack that pushes, a
    part held one way in service and another in handling.  On a component
    the study also holds, the case's value wins.  A case may have no forces
    when a :class:`Displace` in it moves something.
    """

    name: str
    loads: tuple[Force, ...] = ()
    weight: float = 1.0
    constraints: tuple[Fix | Displace, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "loads", tuple(self.loads))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("LoadCase.name must be a non-empty string")
        if not all(isinstance(f, Force) for f in self.loads):
            raise ValueError(f"LoadCase '{self.name}': loads takes Force(...) entries")
        if not all(isinstance(c, (Fix, Displace)) for c in self.constraints):
            raise ValueError(f"LoadCase '{self.name}': constraints takes Fix(...) or Displace(...)")
        moves = any(
            isinstance(c, Displace) and any(v for v in (c.ux, c.uy, c.uz) if v is not None)
            for c in self.constraints
        )
        if not self.loads and not moves:
            raise ValueError(
                f"LoadCase '{self.name}' needs a Force(...), or a Displace(...) that moves something"
            )
        if not self.weight > 0:
            raise ValueError(f"LoadCase '{self.name}': weight must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight,
                "forces": [f.as_dict() for f in self.loads],
                "constraints": [c.as_dict() for c in self.constraints]}


@dataclass(frozen=True)
class Passive:
    """Take elements out of the design: held ``"solid"`` (keep-in: bosses,
    bearing pads, flanges) or ``"void"`` (keep-out: clearance).

    Elements count when their centre lies within ``within`` mm of the region:
    inside a solid region with the default 0, or a layer ``within`` thick
    around a face, edge or point (a ring round a bore: the bore's face and the
    ring's width).
    """

    region: str
    state: str = "solid"
    within: float = 0.0

    def __post_init__(self) -> None:
        _region_name("Passive", self.region)
        if self.state not in ("solid", "void"):
            raise ValueError(f"Passive.state is 'solid' or 'void', not {self.state!r}")
        if self.within < 0:
            raise ValueError("Passive.within must not be negative")

    def as_dict(self) -> dict[str, Any]:
        return {"region": self.region, "state": self.state, "within": self.within}


@dataclass(frozen=True)
class ComplianceLimit:
    """Compliance (N·mm) at most ``max``: of one load case, or with ``case``
    None the weighted sum over the cases."""

    max: float
    case: str | None = None

    def __post_init__(self) -> None:
        if not self.max > 0:
            raise ValueError("ComplianceLimit.max must be positive")


@dataclass(frozen=True)
class DisplacementLimit:
    """The mean ``component`` ("x", "y" or "z") displacement over a region's
    nodes at most ``max`` mm in size, in one load case or (``case`` None) in
    each."""

    region: str
    component: str
    max: float
    case: str | None = None

    def __post_init__(self) -> None:
        _region_name("DisplacementLimit", self.region)
        if self.component not in ("x", "y", "z"):
            raise ValueError("DisplacementLimit.component is 'x', 'y' or 'z'")
        if not self.max > 0:
            raise ValueError("DisplacementLimit.max must be positive")


@dataclass(frozen=True)
class StressLimit:
    """Von Mises stress (MPa) at most ``max``, in one load case or (``case``
    None) in each — through a p-norm of the element stresses relaxed by
    ``x^q``, a smooth bound on the largest one (see
    :func:`topocombo.responses.stress_pnorm`)."""

    max: float
    case: str | None = None
    p: float = 8.0
    q: float = 0.5

    def __post_init__(self) -> None:
        if not self.max > 0:
            raise ValueError("StressLimit.max must be positive")
        if not self.p >= 1 or not 0 < self.q <= 1:
            raise ValueError("StressLimit takes p >= 1 and 0 < q <= 1")


LIMITS = (ComplianceLimit, DisplacementLimit, StressLimit)


@dataclass(frozen=True)
class Study:
    """Everything about a run except the geometry, which ``part`` points to
    (a CadQuery script, relative to the study file).

    Loads come as ``loads`` — the forces of a single case — or as
    ``load_cases``, several weighted :class:`LoadCase` s; give one or the
    other.  ``constraints`` (:class:`Fix`, :class:`Displace`) hold in every
    case, and a case may add its own; ``passive`` regions are kept out of
    the design.
    """

    part: str
    constraints: tuple[Fix | Displace, ...] = ()
    loads: tuple[Force, ...] = ()
    mesh: Mesh = field(default_factory=Mesh)
    material: Material = field(default_factory=Material)
    optimize: SimpParams | None = field(default_factory=SimpParams)
    thickness: float = 1.0  # 2D only: plane-stress out-of-plane size, mm
    #: linear solver: "auto" (direct up to 30k free DOFs, multigrid CG above),
    #: "direct" or "cg"
    solver: str = "auto"
    load_cases: tuple[LoadCase, ...] = ()
    passive: tuple[Passive, ...] = ()
    #: what the optimizer minimises: "compliance" (under the volume fraction
    #: and any limits) or "volume" (the lightest design within the limits)
    objective: str = "compliance"
    limits: tuple[ComplianceLimit | DisplacementLimit | StressLimit, ...] = ()

    def __post_init__(self) -> None:
        for name in ("constraints", "loads", "load_cases", "passive", "limits"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not all(isinstance(c, (Fix, Displace)) for c in self.constraints):
            raise ValueError("Study.constraints takes Fix(...) or Displace(...) entries")
        if not all(isinstance(f, Force) for f in self.loads):
            raise ValueError("Study.loads takes Force(...) entries")
        if not all(isinstance(c, LoadCase) for c in self.load_cases):
            raise ValueError("Study.load_cases takes LoadCase(...) entries")
        if bool(self.loads) == bool(self.load_cases):
            raise ValueError("give Study either loads=[Force(...)] or load_cases=[LoadCase(...)]")
        names = [c.name for c in self.load_cases]
        if len(set(names)) != len(names):
            raise ValueError("Study.load_cases need distinct names")
        if not self.constraints and not all(c.constraints for c in self.cases):
            raise ValueError(
                "Study.constraints needs at least one Fix(...) or Displace(...) "
                "(or every load case its own)"
            )
        if not all(isinstance(p, Passive) for p in self.passive):
            raise ValueError("Study.passive takes Passive(...) entries")
        for name, cls in (("mesh", Mesh), ("material", Material)):
            if not isinstance(getattr(self, name), cls):
                raise ValueError(f"Study.{name} must be a {cls.__name__}(...)")
        if self.optimize is not None and not isinstance(self.optimize, SimpParams):
            raise ValueError("Study.optimize must be a SimpParams(...) or None")
        if not self.thickness > 0:
            raise ValueError("Study.thickness must be positive")
        if self.solver not in SOLVERS:
            raise ValueError(f"Study.solver must be one of {', '.join(SOLVERS)}")
        if self.objective not in ("compliance", "volume"):
            raise ValueError("Study.objective is 'compliance' or 'volume'")
        if not all(isinstance(lim, LIMITS) for lim in self.limits):
            raise ValueError(
                "Study.limits takes ComplianceLimit, DisplacementLimit or StressLimit entries"
            )
        if self.objective == "volume" and not self.limits:
            raise ValueError("objective='volume' needs limits: the lightest design within what?")
        names = {c.name for c in self.cases}
        unknown = [lim.case for lim in self.limits if lim.case is not None and lim.case not in names]
        if unknown:
            raise ValueError(f"Study.limits name unknown load case(s): {', '.join(unknown)}")
        if (self.limits or self.objective != "compliance") and self.optimize is not None \
                and self.optimize.optimizer == "oc":
            raise ValueError("limits and a volume objective need optimizer='mma' (or 'auto'); "
                             "OC takes minimum compliance under the volume only")

    @property
    def cases(self) -> tuple[LoadCase, ...]:
        """The load cases: ``load_cases``, or ``loads`` as one case named "load"."""
        return self.load_cases or (LoadCase("load", self.loads),)

    @property
    def forces(self) -> list[Force]:
        return [f for case in self.cases for f in case.loads]

    @property
    def regions_used(self) -> list[str]:
        names = (
            [c.region for c in self.constraints] + [f.region for f in self.forces]
            + [c.region for case in self.cases for c in case.constraints]
            + [p.region for p in self.passive]
            + [lim.region for lim in self.limits if isinstance(lim, DisplacementLimit)]
        )
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
    for force in study.forces:
        if len(force.vector) != domain.dim:
            raise ValueError(
                f"{path}: Force on '{force.region}' has {len(force.vector)} components "
                f"for a {domain.dim}D part"
            )
    study.mesh.spec(domain.dim)  # element type fits the dimension
    return LoadedStudy(study=study, domain=domain, path=path, source=source)


def run_study(path: Path, out_dir: Path, echo: bool = True) -> tuple[Any, dict[str, Any]]:
    """Load ``study.py`` and run the pipeline on it."""
    return run_loaded(load_study(path), out_dir, echo=echo)


def run_loaded(loaded: LoadedStudy, out_dir: Path, echo: bool = True,
               optimize: bool = True) -> tuple[Any, dict[str, Any]]:
    """Run the pipeline on a loaded study (``optimize=False`` stops after the
    full-density solve)."""
    from .pipeline import run

    s = loaded.study
    return run(
        domain=loaded.domain,
        spec=loaded.spec,
        out_dir=out_dir,
        material=s.material,
        simp=s.optimize or SimpParams(),
        optimize_design=optimize and s.optimize is not None,
        constraints=s.constraints,
        load_cases=s.cases,
        passive_regions=s.passive,
        objective=s.objective,
        limits=s.limits,
        study=loaded,
        echo=echo,
        solver=s.solver,
    )
