"""Quadrilateral meshing of the design domain with Gmsh.

The CAD face exported by :mod:`topocombo.geometry` is imported through the OCC
kernel and meshed with a *transfinite* (structured) quad grid, which is what the
SIMP loop wants: one design variable per element, uniform element size, and a
predictable element ordering.

Boundary regions are tagged as physical groups so the downstream solver never
has to re-derive them from coordinates:

``design_domain``  the meshed surface (all quads)
``fixed``          left edge — clamped
``load_edge``      right edge — carries the tip load

The 3D domain is meshed the same way into a structured hexahedral grid
(:func:`generate_hex_mesh`); the groups keep their names but tag the volume and
the clamped / loaded *faces* (x = 0 and x = L) instead of edges.  By default the
grid is one element through the width, which keeps the CI run light.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import gmsh

from .geometry import BeamDomain, BeamDomain3D

#: Physical group names written into the .msh file.
PHYS_DOMAIN = "design_domain"
PHYS_FIXED = "fixed"
PHYS_LOAD = "load_edge"

_TOL = 1e-6


@dataclass(frozen=True)
class MeshSpec:
    """Discretisation of the design domain."""

    nelx: int = 60
    nely: int = 20

    def __post_init__(self) -> None:
        if self.nelx < 1 or self.nely < 1:
            raise ValueError("nelx and nely must be >= 1")

    @property
    def n_elements(self) -> int:
        return self.nelx * self.nely

    @property
    def n_nodes(self) -> int:
        return (self.nelx + 1) * (self.nely + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nelx": self.nelx,
            "nely": self.nely,
            "expected_elements": self.n_elements,
            "expected_nodes": self.n_nodes,
        }


@dataclass(frozen=True)
class MeshSpec3D:
    """Discretisation of the 3D design domain; one element through the width by default."""

    nelx: int = 60
    nely: int = 20
    nelz: int = 1

    def __post_init__(self) -> None:
        if self.nelx < 1 or self.nely < 1 or self.nelz < 1:
            raise ValueError("nelx, nely and nelz must be >= 1")

    @property
    def n_elements(self) -> int:
        return self.nelx * self.nely * self.nelz

    @property
    def n_nodes(self) -> int:
        return (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "nelx": self.nelx,
            "nely": self.nely,
            "nelz": self.nelz,
            "expected_elements": self.n_elements,
            "expected_nodes": self.n_nodes,
        }


def _classify_curves(domain: BeamDomain) -> dict[str, list[int]]:
    """Sort the boundary curves of the imported face into named edges."""
    groups: dict[str, list[int]] = {"left": [], "right": [], "bottom": [], "top": []}
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(dim, tag)
        if abs(xmin) < _TOL and abs(xmax) < _TOL:
            groups["left"].append(tag)
        elif abs(xmin - domain.length) < _TOL and abs(xmax - domain.length) < _TOL:
            groups["right"].append(tag)
        elif abs(ymin) < _TOL and abs(ymax) < _TOL:
            groups["bottom"].append(tag)
        elif abs(ymin - domain.height) < _TOL and abs(ymax - domain.height) < _TOL:
            groups["top"].append(tag)
        else:  # pragma: no cover - only reachable for non-rectangular domains
            raise RuntimeError(f"curve {tag} is not on a domain boundary")
    missing = [name for name, tags in groups.items() if not tags]
    if missing:  # pragma: no cover - defensive
        raise RuntimeError(f"no boundary curve found for: {', '.join(missing)}")
    return groups


@contextmanager
def _gmsh_session(model: str, log: Any = None) -> Iterator[list[str]]:
    """Initialise Gmsh with its logger on; always finalise, and echo the log.

    Yields the list that receives Gmsh's own log lines once the block exits.
    """
    gmsh.initialize()
    gmsh_log: list[str] = []
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()
        gmsh.model.add(model)
        yield gmsh_log
        gmsh_log.extend(gmsh.logger.get())
        if log is not None:
            log.tool("gmsh", gmsh.option.getString("General.Version"))
            for line in gmsh_log:
                log.log(f"[gmsh] {line}")
    finally:
        try:
            gmsh.logger.stop()
        except Exception:  # pragma: no cover - logger may not be started
            pass
        gmsh.finalize()


def _import_single(brep_path: Path, dim: int) -> int:
    """Import ``brep_path`` and return the tag of its one entity of dimension ``dim``."""
    gmsh.model.occ.importShapes(str(brep_path))
    gmsh.model.occ.synchronize()
    entities = gmsh.model.getEntities(dim)
    if len(entities) != 1:  # pragma: no cover - defensive
        raise RuntimeError(f"expected one entity of dim {dim} in {brep_path}, got {len(entities)}")
    return entities[0][1]


def _write_msh(msh_path: Path) -> None:
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(msh_path))


def generate_quad_mesh(
    domain: BeamDomain,
    spec: MeshSpec,
    brep_path: Path,
    out_dir: Path,
    log: Any = None,
) -> tuple[Path, list[str]]:
    """Mesh ``brep_path`` into a structured quad grid; return (msh path, gmsh log)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / "beam.msh"

    with _gmsh_session("cantilever_beam", log) as gmsh_log:
        surf_tag = _import_single(brep_path, 2)

        edges = _classify_curves(domain)
        if log is not None:
            log.log(f"imported {brep_path.name}: 1 surface, {len(gmsh.model.getEntities(1))} curves")
            log.log(
                "boundary curves — "
                + ", ".join(f"{name}: {tags}" for name, tags in edges.items())
            )

        # Structured grid: nelx divisions along x, nely along y.
        for tag in edges["bottom"] + edges["top"]:
            gmsh.model.mesh.setTransfiniteCurve(tag, spec.nelx + 1)
        for tag in edges["left"] + edges["right"]:
            gmsh.model.mesh.setTransfiniteCurve(tag, spec.nely + 1)
        gmsh.model.mesh.setTransfiniteSurface(surf_tag)
        gmsh.model.mesh.setRecombine(2, surf_tag)

        # Physical groups carry the boundary conditions downstream.
        gmsh.model.addPhysicalGroup(2, [surf_tag], name=PHYS_DOMAIN)
        gmsh.model.addPhysicalGroup(1, edges["left"], name=PHYS_FIXED)
        gmsh.model.addPhysicalGroup(1, edges["right"], name=PHYS_LOAD)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        _write_msh(msh_path)

    return msh_path, gmsh_log


def _classify_box(domain: BeamDomain3D) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    """Sort the 12 edges of the box by direction and its 6 faces by position.

    Edges are keyed ``x``/``y``/``z`` (the axis they run along); faces are keyed
    ``xmin``/``xmax``/``ymin``/``ymax``/``zmin``/``zmax`` (the plane they lie in).
    """
    extent = (domain.length, domain.height, domain.width)
    axes = "xyz"

    edges: dict[str, list[int]] = {a: [] for a in axes}
    for _, tag in gmsh.model.getEntities(1):
        box = gmsh.model.getBoundingBox(1, tag)
        spans = [box[i + 3] - box[i] for i in range(3)]
        along = [i for i, span in enumerate(spans) if span > _TOL]
        if len(along) != 1:  # pragma: no cover - only reachable for non-box domains
            raise RuntimeError(f"curve {tag} is not parallel to an axis")
        edges[axes[along[0]]].append(tag)

    faces: dict[str, list[int]] = {f"{a}{end}": [] for a in axes for end in ("min", "max")}
    for _, tag in gmsh.model.getEntities(2):
        box = gmsh.model.getBoundingBox(2, tag)
        flat = [i for i in range(3) if box[i + 3] - box[i] < _TOL]
        if len(flat) != 1:  # pragma: no cover - only reachable for non-box domains
            raise RuntimeError(f"surface {tag} is not normal to an axis")
        i = flat[0]
        if abs(box[i]) < _TOL:
            faces[f"{axes[i]}min"].append(tag)
        elif abs(box[i] - extent[i]) < _TOL:
            faces[f"{axes[i]}max"].append(tag)
        else:  # pragma: no cover - only reachable for non-box domains
            raise RuntimeError(f"surface {tag} is not on a domain boundary")

    missing = [name for name, tags in {**edges, **faces}.items() if not tags]
    if missing or any(len(tags) != 4 for tags in edges.values()):  # pragma: no cover
        raise RuntimeError(f"box topology not recognised (missing: {', '.join(missing)})")
    return edges, faces


def generate_hex_mesh(
    domain: BeamDomain3D,
    spec: MeshSpec3D,
    brep_path: Path,
    out_dir: Path,
    log: Any = None,
) -> tuple[Path, list[str]]:
    """Mesh the box in ``brep_path`` into a structured hex grid; return (msh path, gmsh log)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / "beam.msh"

    with _gmsh_session("cantilever_beam_3d", log) as gmsh_log:
        vol_tag = _import_single(brep_path, 3)

        edges, faces = _classify_box(domain)
        if log is not None:
            log.log(
                f"imported {brep_path.name}: 1 volume, {len(gmsh.model.getEntities(2))} "
                f"surfaces, {len(gmsh.model.getEntities(1))} curves"
            )
            log.log("edges by direction — " + ", ".join(f"{a}: {t}" for a, t in edges.items()))
            log.log("faces by position — " + ", ".join(f"{f}: {t}" for f, t in faces.items()))

        # Structured grid: nelx divisions along x, nely along y, nelz along z.
        divisions = {"x": spec.nelx, "y": spec.nely, "z": spec.nelz}
        for axis, tags in edges.items():
            for tag in tags:
                gmsh.model.mesh.setTransfiniteCurve(tag, divisions[axis] + 1)
        for tags in faces.values():
            for tag in tags:
                gmsh.model.mesh.setTransfiniteSurface(tag)
                gmsh.model.mesh.setRecombine(2, tag)  # quad faces -> hexahedra
        gmsh.model.mesh.setTransfiniteVolume(vol_tag)

        # Physical groups carry the boundary conditions downstream.
        gmsh.model.addPhysicalGroup(3, [vol_tag], name=PHYS_DOMAIN)
        gmsh.model.addPhysicalGroup(2, faces["xmin"], name=PHYS_FIXED)
        gmsh.model.addPhysicalGroup(2, faces["xmax"], name=PHYS_LOAD)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        _write_msh(msh_path)

    return msh_path, gmsh_log
