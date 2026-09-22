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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gmsh

from .geometry import BeamDomain

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

    gmsh.initialize()
    gmsh_log: list[str] = []
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.logger.start()
        gmsh.model.add("cantilever_beam")

        gmsh.model.occ.importShapes(str(brep_path))
        gmsh.model.occ.synchronize()

        surfaces = gmsh.model.getEntities(2)
        if len(surfaces) != 1:  # pragma: no cover - defensive
            raise RuntimeError(f"expected one surface in {brep_path}, got {len(surfaces)}")
        surf_tag = surfaces[0][1]

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

        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(msh_path))

        gmsh_log = list(gmsh.logger.get())
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

    return msh_path, gmsh_log
