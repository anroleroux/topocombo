"""Quadrilateral meshing of the design domain with Gmsh.

The CAD face exported by :mod:`topocombo.geometry` is imported through the OCC
kernel and meshed with a *transfinite* (structured) quad grid, which is what the
SIMP loop wants: one design variable per element, uniform element size, and a
predictable element ordering.

The meshed surface or volume is tagged ``design_domain``.  Nothing here knows
where the part is held or loaded: that is the named regions' job
(:mod:`topocombo.regions`), which pick their nodes from the finished mesh.

The 3D domain is meshed the same way into a structured hexahedral grid
(:func:`generate_hex_mesh`).  By default the grid is one element through the
width, which keeps the CI run light.

The *body-fitted* mode (:func:`generate_fitted_mesh`) meshes the real CAD
profile, cutouts included: an unstructured all-quad mesh of the x-y face
(frontal-Delaunay triangles recombined into quads), extruded through the width
into hexahedra in 3D.  The corners of the named regions are embedded in the
profile first, so a region that is not a whole CAD edge or face (a load line
across the free end, say) still lands on real nodes.  The structured grid stays the default and the
reference; the body-fitted one follows curved boundaries exactly (to within the
chordal error of the element edges) at the cost of a non-uniform mesh.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import gmsh

from .elements import HEX8, QUAD4, Element
from .geometry import Domain

#: Physical group name of the meshed surface / volume in the .msh file.
PHYS_DOMAIN = "design_domain"

_TOL = 1e-6

#: ``structured``: a transfinite grid over the L x H envelope (cutouts held
#: void by the optimizer); ``body-fitted``: an unstructured quad / extruded-hex
#: mesh of the actual CAD profile, cutouts included.
MESH_MODES = ("structured", "body-fitted")


def _check_mode(mode: str, size: float | None) -> None:
    if mode not in MESH_MODES:
        raise ValueError(f"mesh mode must be one of {', '.join(MESH_MODES)}, not {mode!r}")
    if size is not None and not size > 0:
        raise ValueError(f"mesh size must be positive, not {size!r}")


@dataclass(frozen=True)
class MeshSpec:
    """Discretisation of the design domain.

    In body-fitted mode ``size`` is the target element edge length; left
    unset, it is the structured grid's cell size, ``min(L / nelx, H / nely)``.
    """

    nelx: int = 60
    nely: int = 20
    mode: str = "structured"
    size: float | None = None

    def __post_init__(self) -> None:
        if self.nelx < 1 or self.nely < 1:
            raise ValueError("nelx and nely must be >= 1")
        _check_mode(self.mode, self.size)

    #: the element this spec meshes into
    element: Element = field(default=QUAD4, init=False, repr=False, compare=False)

    @property
    def structured(self) -> bool:
        return self.mode == "structured"

    @property
    def n_elements(self) -> int | None:
        """Element count of the structured grid; None when body-fitted."""
        return self.nelx * self.nely if self.structured else None

    @property
    def n_nodes(self) -> int | None:
        return (self.nelx + 1) * (self.nely + 1) if self.structured else None

    def element_size(self, domain: Domain) -> float:
        """The target edge length of a body-fitted mesh."""
        return self.size or min(domain.length / self.nelx, domain.height / self.nely)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nelx": self.nelx,
            "nely": self.nely,
            "size": self.size,
            "expected_elements": self.n_elements,
            "expected_nodes": self.n_nodes,
        }


@dataclass(frozen=True)
class MeshSpec3D:
    """Discretisation of the 3D design domain.

    Hexahedra (the default) come from a structured grid or from the x-y
    profile extruded into ``nelz`` layers, so the part must be a prism along
    z.  Tetrahedra (``TET4``, ``TET10``) are meshed by Gmsh from the solid
    itself at edge length ``size`` — any 3D part, body-fitted only; ``nelz``
    does not apply to them.
    """

    nelx: int = 60
    nely: int = 20
    nelz: int = 1
    mode: str = "structured"
    size: float | None = None
    #: the element this spec meshes into
    element: Element = field(default=HEX8, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.nelx < 1 or self.nely < 1 or self.nelz < 1:
            raise ValueError("nelx, nely and nelz must be >= 1")
        _check_mode(self.mode, self.size)
        if self.element.dim != 3:
            raise ValueError(f"a 3D mesh takes 3D elements, not {self.element.name}")
        if self.tetrahedral and self.structured:
            raise ValueError(
                f"{self.element.name} elements are meshed body-fitted; "
                "the structured grid is hexahedral"
            )

    @property
    def tetrahedral(self) -> bool:
        return self.element.cell_type.startswith("tetra")

    @property
    def structured(self) -> bool:
        return self.mode == "structured"

    @property
    def n_elements(self) -> int | None:
        """Element count of the structured grid; None when body-fitted."""
        return self.nelx * self.nely * self.nelz if self.structured else None

    @property
    def n_nodes(self) -> int | None:
        return (self.nelx + 1) * (self.nely + 1) * (self.nelz + 1) if self.structured else None

    def element_size(self, domain: Domain) -> float:
        """The target edge length of a body-fitted mesh: in-plane for extruded
        hexes (``nelz`` layers in z), in every direction for tetrahedra."""
        return self.size or min(domain.length / self.nelx, domain.height / self.nely)

    def as_dict(self) -> dict[str, Any]:
        return {
            "element": self.element.name,
            "mode": self.mode,
            "nelx": self.nelx,
            "nely": self.nely,
            "nelz": None if self.tetrahedral else self.nelz,
            "size": self.size,
            "expected_elements": self.n_elements,
            "expected_nodes": self.n_nodes,
        }


def _classify_curves(domain: Domain) -> dict[str, list[int]]:
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
    domain: Domain,
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

        gmsh.model.addPhysicalGroup(2, [surf_tag], name=PHYS_DOMAIN)

        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        _write_msh(msh_path)

    return msh_path, gmsh_log


def _classify_box(domain: Domain) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
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
    domain: Domain,
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

        gmsh.model.addPhysicalGroup(3, [vol_tag], name=PHYS_DOMAIN)

        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(1)
        _write_msh(msh_path)

    return msh_path, gmsh_log


def generate_fitted_mesh(
    domain: Domain,
    spec: MeshSpec | MeshSpec3D,
    brep_path: Path,
    out_dir: Path,
    log: Any = None,
    points: Any = (),
) -> tuple[Path, list[str]]:
    """Mesh the CAD profile in ``brep_path`` (any planar x-y face, cutouts and all)
    into unstructured quads — or, for a 3D domain, extrude those quads through
    the width into ``spec.nelz`` layers of hexahedra.  ``points`` are (x, y)
    positions the mesh must have nodes at (the corners of the named regions);
    they are fused into the profile before meshing.  Returns (msh path, gmsh log).
    """
    three_d = domain.dim == 3
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / "beam.msh"
    size = spec.element_size(domain)

    with _gmsh_session("design_domain_fitted", log) as gmsh_log:
        surf_tag = _import_single(brep_path, 2)
        before = len(gmsh.model.getEntities(0))
        tags = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in points]
        if tags:
            gmsh.model.occ.fragment([(2, surf_tag)], [(0, t) for t in tags])
            gmsh.model.occ.synchronize()
            surfaces = gmsh.model.getEntities(2)
            if len(surfaces) != 1:  # pragma: no cover - a point cannot split a face
                raise RuntimeError(f"embedding region points split the profile into {len(surfaces)}")
            surf_tag = surfaces[0][1]
        n_curves = len(gmsh.model.getEntities(1))
        if log is not None:
            added = len(gmsh.model.getEntities(0)) - before
            log.log(
                f"imported {brep_path.name}: 1 surface, {n_curves} curves; "
                f"{added} region point(s) embedded"
            )
            log.log(
                f"unstructured quads: target size {size:g} mm, frontal-Delaunay "
                "triangles recombined to all-quad"
                + (f", extruded into {spec.nelz} hex layer(s) through the width" if three_d else "")
            )

        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # frontal-Delaunay for quads
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # full-quad
        gmsh.model.mesh.setRecombine(2, surf_tag)

        if three_d:
            extruded = gmsh.model.occ.extrude(
                [(2, surf_tag)], 0.0, 0.0, domain.width, numElements=[spec.nelz], recombine=True
            )
            gmsh.model.occ.synchronize()
            vol_tag = next(tag for dim, tag in extruded if dim == 3)
            gmsh.model.addPhysicalGroup(3, [vol_tag], name=PHYS_DOMAIN)
            gmsh.model.mesh.generate(3)
        else:
            gmsh.model.addPhysicalGroup(2, [surf_tag], name=PHYS_DOMAIN)
            gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(1)
        _write_msh(msh_path)

    return msh_path, gmsh_log


def generate_tet_mesh(
    domain: Domain,
    spec: MeshSpec3D,
    brep_path: Path,
    out_dir: Path,
    log: Any = None,
    regions: dict[str, Any] | None = None,
) -> tuple[Path, list[str]]:
    """Mesh the solid in ``brep_path`` — any 3D part — into tetrahedra of
    ``spec.element`` (linear or quadratic) at edge length ``spec.size``.

    The named ``regions`` are fused into the solid first (OCC fragment), so a
    region that is not a whole CAD face or edge — a load line across a face,
    say — is imprinted on the boundary and gets mesh nodes along it.
    Quadratic elements put their mid-nodes on the CAD geometry, curved faces
    included.  Returns (msh path, gmsh log).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    msh_path = out_dir / "beam.msh"
    size = spec.element_size(domain)

    with _gmsh_session("design_domain_tet", log) as gmsh_log:
        gmsh.model.occ.importShapes(str(brep_path))
        gmsh.model.occ.synchronize()
        volumes = gmsh.model.getEntities(3)
        if len(volumes) != 1:  # pragma: no cover - the CAD check allows one solid
            raise RuntimeError(f"expected one solid in {brep_path}, got {len(volumes)}")
        if regions:
            # one file per region: importShapes keeps only the highest dimension
            # it finds, so a face and a line imported together would lose the line
            tools = []
            region_brep = out_dir / "region.brep"
            for shape in regions.values():
                shape.exportBrep(str(region_brep))
                tools += gmsh.model.occ.importShapes(str(region_brep))
            region_brep.unlink()
            gmsh.model.occ.fragment(volumes, tools)
            gmsh.model.occ.synchronize()
        vols = [tag for _, tag in gmsh.model.getEntities(3)]
        if log is not None:
            log.log(
                f"imported {brep_path.name}: {len(vols)} volume, "
                f"{len(gmsh.model.getEntities(2))} surfaces, {len(gmsh.model.getEntities(1))} curves"
                + (f"; {len(regions)} region(s) fused in" if regions else "")
            )
            kind = (
                "quadratic (10-node), mid-nodes on the CAD geometry"
                if spec.element.order == 2 else "linear (4-node)"
            )
            log.log(f"tetrahedra: target size {size:g} mm, HXT + Netgen optimisation, {kind}")
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # frontal-Delaunay surfaces
        # HXT volume meshing, then Gmsh's and Netgen's optimisers: plain Delaunay
        # leaves slivers in thin parts, which stiffen and ill-condition the solve
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
        gmsh.model.addPhysicalGroup(3, vols, name=PHYS_DOMAIN)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(spec.element.order)
        _write_msh(msh_path)

    return msh_path, gmsh_log
