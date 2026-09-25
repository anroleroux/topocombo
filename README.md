# topocombo

A topology optimization pipeline built from decoupled, open-source components — a parametric CAD tool, a mesher, an FEA solver, an optimizer, and separate visualization tools.

## Goal

Explore topology optimization (SIMP-based compliance minimization) on a cantilever beam, using an open-source toolchain end to end — as a 3D solid meshed with hexahedra (the default published run), or as the original 2D plane-stress problem meshed with quadrilaterals. The focus is a working, terminal-driven optimization loop; visualization is treated as a downstream, optional concern rather than something embedded in the loop itself. The FEA solver is a custom, minimal implementation, with CalculiX as a planned future swap-in once the core loop is validated.

## Components

- **CAD (geometry)** — [CadQuery](https://github.com/CadQuery/cadquery): parametric definition of the design domain (beam dimensions, aspect ratio, load/support regions), scripted in Python.
- **Mesher** — [Gmsh](https://gmsh.info/): structured (transfinite) or body-fitted quadrilateral / hexahedral meshing, and tetrahedral meshing of any solid, driven via its Python API.
- **FEA solver** — custom solver with Q4 plane-stress quads, H8 solid hexahedra and T4 / T10 tetrahedra (numpy/scipy: sparse stiffness assembly, a direct solve or multigrid-preconditioned CG via [PyAMG](https://github.com/pyamg/pyamg)), implemented in `src/topocombo/fea.py` and `src/topocombo/elements.py`. Chosen over an external solver initially so the optimizer has direct, in-memory access to element stiffness matrices and displacement fields for sensitivity analysis. [CalculiX](http://www.calculix.de/) is the planned later alternative for a verified, general-purpose solver.
- **Optimizer** — SIMP (Solid Isotropic Material with Penalization) loop, implemented in `src/topocombo/optimize.py`: density update via Optimality Criteria (with [NLopt](https://nlopt.readthedocs.io/)'s MMA as a planned alternative), with sensitivity or density filtering to avoid checkerboarding. The per-iteration coupling (FEA solve → compliance + sensitivity → filter → update) is custom code.
- **Visualization (decoupled)**:
  - [PyVista](https://pyvista.org/) — scripted plotting of density fields and results, run as a separate process/script against exported data, not called from within the optimization loop.
  - [Blender](https://www.blender.org/) (optional) — presentation-quality rendering of optimized geometry, consuming exported mesh data independently via its Python API (`bpy`) or manual import.

## Flows

### Main optimization loop (terminal, headless)
1. Define parametric geometry in CadQuery → export CAD file.
2. Mesh with Gmsh → quadrilateral (2D), hexahedral or tetrahedral (3D) mesh.
3. Run the SIMP loop:
   - Assemble stiffness matrix, solve FEA (custom Q4 plane-stress / H8 / T4 / T10 solid solver).
   - Compute compliance and sensitivities.
   - Apply density/sensitivity filter.
   - Update design variables (OC or NLopt-MMA).
   - Check convergence.
4. Print iteration metrics (compliance, volume fraction, design change) to terminal.
5. Periodically write results to disk: density field + mesh (e.g. `.vtu` via `meshio`, or `.npy`) and a scalar log (`log.csv`).

No plotting or rendering happens inside this loop — it only reads geometry/mesh input and writes result artifacts.

### Visualization (separate, on demand)
- **PyVista**: `python -m topocombo.viz --run <run dir>` reads the exported `.vtu` files and renders the thresholded topology, the density field and the deformed stress field — to PNGs off screen, or interactively with `--show`.
- **Blender**: import `optimization/topology.stl` — the density field thresholded at 0.5, as a closed, outward-facing surface written with `meshio` — for polished rendering.

Because both visualization paths consume the same on-disk result artifacts rather than talking to the solver directly, swapping the FEA backend (e.g. to CalculiX later) does not require changes to either visualization script.

## Running the cantilever example

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# CadQuery domain -> Gmsh quad mesh -> plane-stress solve -> SIMP loop -> artifacts
python -m topocombo.cli run --length 60 --height 20 --nelx 60 --nely 20 \
    --load -1000 --volfrac 0.5 --penal 3 --rmin 1.5 --out results/cantilever

# stop after the full-density solve, without optimizing
python -m topocombo.cli run --no-optimize

# render the run log as a static HTML page
python -m topocombo.cli report --run results/cantilever --site site

# or both at once
python -m topocombo.cli all

# the same cantilever as a 3D solid: hexahedra, one element through the width
python -m topocombo.cli all --dim 3 --width 1 --nelz 1 --out results/cantilever3d

# any design: a study script (mesh, material, constraints, loads, optimizer)
# pointing at a CadQuery part script (geometry + named regions) — what CI publishes
python -m topocombo.cli all --study examples/cantilever/study.py --out results/cantilever --site site

# just the CAD stage of a study: print the part script, write BREP, STEP and a PNG preview
python -m topocombo.cli cad --study examples/cantilever/study.py --out results/cad

# a 10 mm hole through z at x = 20, y = 10 (repeat --hole for more)
python -m topocombo.cli all --dim 3 --hole 20,10,10 --out results/holed

# the same part meshed body-fitted: the mesh follows the hole instead of holding
# grid cells void (--mesh-size sets the element edge, default L/nelx)
python -m topocombo.cli all --dim 3 --hole 20,10,10 --mesh body-fitted --out results/fitted

# a shaded PNG of any exported BREP or STL, no OpenGL needed
python -m topocombo.cadview results/cantilever3d/optimization/topology.stl

# PyVista views of a finished run, written to <run>/figures (optional extra)
pip install -e ".[viz]"
python -m topocombo.viz --run results/cantilever3d
```

### A design as two Python scripts

Any design is a directory with two scripts; `examples/cantilever/` is the
published one:

```
examples/cantilever/
  part.py     # CadQuery: `result` (the part) and `regions` (named places on it)
  study.py    # `study = Study(part="part.py", ...)`: mesh, material, constraints, loads, optimizer
```

**`part.py`** is a CadQuery script. It assigns the part to `result` and names
the places where it will be held and loaded in a `regions` dict — vertices,
edges or faces, picked from the part with selectors or built on their own:

```python
result = cq.Workplane("XY").box(60, 20, 1, centered=False)  # ... cut the notch
regions = {
    "wall": result.faces("<X"),                                    # the x = 0 face
    "tip": cq.Edge.makeLine(cq.Vector(60, 10, 0), cq.Vector(60, 10, 1)),  # a load line
}
```

**`study.py`** refers to those names:

```python
from topocombo.study import Fix, Force, Material, Mesh, SimpParams, Study

study = Study(
    part="part.py",
    mesh=Mesh(mode="body-fitted", size=1.0, layers=1),
    material=Material(youngs_modulus=210_000.0, poisson_ratio=0.3),
    constraints=[Fix("wall")],                  # clamped: every component zero
    loads=[Force("tip", (0.0, -1000.0, 0.0))],  # N, total, spread over the region
    optimize=SimpParams(volume_fraction=0.5, penal=3.0, filter_radius=1.5),
)
```

Both are ordinary Python, so values can be computed, looped over or imported.
Each building block checks its own fields, and `load_study` checks the two
scripts agree — every region the study names exists, the force has one
component per dimension, the element fits the part — before anything is
meshed. `--study` replaces the flags it covers; giving one of them as well is
an error (`--out`, `--site` and `--no-optimize` still apply).

Nothing in the pipeline knows where a part is held or loaded. Each region
becomes the node set of the mesh nodes lying on it (a distance test against
the CAD shape, so it works the same for a structured grid, a body-fitted mesh
and tetrahedra); the body-fitted meshers make sure a region that is not a
whole CAD edge or face — the load line above — still lands on real nodes (the
extruded-hex mesher embeds the regions' corners in the profile, the tet mesher
fuses the regions into the solid). A force is spread over its region by the
region's dimension — all on one node (a vertex), by tributary length (edges),
or by tributary area (faces) — split over each piece's nodes by the element's
consistent weights: evenly for linear elements, 1/6-2/3-1/6 along a quadratic
edge, all on the mid-nodes of a 6-node triangle. `Fix` clamps every displacement component for
now; one `Force` per study.

The part script is checked too, with a message naming the file: `result`
must be one connected face or solid; its bounding box must start at the
origin; and regions must be vertices, edges or faces that select something.
A face in the x-y plane is a 2D plane-stress part (`Study.thickness` sets its
out-of-plane size); a solid is 3D. Any solid can be meshed with tetrahedra;
hexahedra need a prism along z (its z = 0 face swept through the width — the
hex mesh is an extrusion of that profile), and a part that is not one is
refused with a message saying so and pointing at `Mesh(element="tet10")`. The pipeline reads everything else it needs from the built shape: the
bounding box, the x-y profile Gmsh meshes, the material area or volume the
mesh is checked against, and which grid cells lie outside the part.

Without a study, the flags describe the parametric cantilever: each run
generates a standalone CadQuery script from `dim`, `length`, `height`,
`thickness` (2D), `width` (3D) and `--hole` cutouts, with the beam's own
regions — `fixed` (the x = 0 edge or face) and `load` (the mid-height point,
or line across the width, at the free end) — and `--load` as the force.
Either way the part script that ran is written to `cad/design_domain.py` (it
opens as-is in CQ-editor) and a study to `study.py`; the report prints both.

Cutouts are given as `--hole X,Y,D` (repeatable) or cut in a part script. The
published part has one hole of diameter 10 mm at x = 20, moved from
mid-height (y = 10) to the bottom edge (y = 0) as a generalisation test: it
now bites a half-circle notch out of the beam, so the profile is no longer a
rectangle with an interior hole. Body-fitted, that meshes into 1214 hexes and
converges in 48 iterations to 1127 N·mm (761 N·mm at full density); on the
structured grid 40 cells are held void, giving 1111 N·mm.
CadQuery cuts holes from the model, so the script, STEP, BREP and picture all
carry them. The structured grid still covers the full L x H envelope
(`design_envelope.brep` is what Gmsh meshes), and the elements whose centres
fall outside the part are held at zero density by the optimizer: a passive,
non-design region, the usual SIMP treatment. The volume fraction stays
relative to the whole envelope, and the full-density solve already has the
cutout void.

### Mesh modes

`--mesh structured` (the CLI default) is the transfinite grid above: uniform
cells, one per design variable, over the L x H envelope. `--mesh body-fitted`
meshes the real CAD profile instead, cutouts included. Gmsh splits the free
edge at mid-height so the load lands on nodes, meshes the x-y face into
unstructured quads (frontal-Delaunay triangles recombined to all-quad) at
`--mesh-size` (default `min(L/nelx, H/nely)`), and in 3D extrudes them through
the width into `--nelz` layers of hexahedra. The 3D profile is written to
`cad/design_profile.brep`. Nothing is held void: the hole is simply not meshed.
It meshes any profile a script builds, not only holes. This is the mode CI publishes; the structured grid stays the CLI default and
the reference the exact 2D <-> 3D and beam-theory checks run on.

Validation changes with it. Instead of the grid count, the meshed area or
volume must match the CAD shape (cutouts removed) to 0.5%; straight element
edges on an arc give a chordal error of about 0.05% at 1 mm. No element may be
stretched past an in-plane edge ratio of 4. The density filter weights
neighbours by their size once the elements are unequal (on a uniform grid the
factor is constant and skipped, so structured results are unchanged). The
report draws the unstructured mesh and density field element by element; a 3D
run gets the side view only, because the top and end projections need grid
columns.

The volume fraction is always relative to the meshed region. Structured, that
is the envelope with the hole counted as void; body-fitted, it is the holed
part. So at the same `--volfrac` the body-fitted design has 50% of 1121 mm³
rather than 50% of 1200 mm³, and is a little more compliant (969 vs 909 N·mm
for the 3D case with the hole).

With `--dim 3` the beam is a CadQuery box meshed into hexahedra, solved with
the H8 solid element, and loaded along a line across the width at mid-height of
the free end. The default 60 x 20 x 1 mesh (1200 hexes, 7686 DOFs) converges in
60 iterations to 899.98 N·mm in about 15 s, the same truss as the 2D run; it is
0.4% stiffer than plane stress because the width is not free to contract.

`gmsh`'s shared library links against GLU, so on a bare Linux box install it first:
`sudo apt-get install libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1`.
Rendering PyVista off screen on a headless box also needs `libosmesa6` or `libegl1`.

Run the tests with `pytest` — they mesh a coarse beam and assert the grid is
uniform, every element is counter-clockwise, the meshed area matches the design
domain, and the boundary node sets land on the right edges; they check the
solver against rigid-body modes, load linearity, equilibrium of the reactions,
and Timoshenko beam theory on a slender beam; and they check the loop holds the
volume constraint every iteration, beats a uniform design of the same volume,
and produces a design symmetric about the beam's mid-height, as the symmetric
load case demands. The 3D tests check the hex mesh the same way, the H8 element
against its six rigid-body modes, a uniform strain state and beam theory, and
the strongest single check: with nu = 0 and one element through the width, the
3D solve and the whole 3D SIMP loop reproduce the 2D ones to round-off. They
also check that the exported topology surface is closed and encloses exactly
the solid elements.

### Artifacts of a run

| File | Contents |
| --- | --- |
| `cad/design_domain.py` | the CadQuery part script that built the design domain — yours with `--study` (runs in CQ-editor) |
| `study.py` | the study script, when the run came from one |
| `cad/design_domain.brep` | design domain; Gmsh's OCC importer meshes it directly when there are no cutouts |
| `cad/design_envelope.brep` | when the part does not fill its bounding box: the L x H envelope the structured grid meshes |
| `cad/design_domain.step` | same geometry for exchange with other CAD tools |
| `mesh/beam.msh` | hex (3D) or quad (2D) mesh with its `design_domain` physical group |
| `mesh/mesh.npz` | nodes, cell connectivity (`cells`, `cell_type`), region node sets (`set_<name>`), constrained nodes, load node(s) and force, `passive` void elements (with cutouts) — what the solver reads |
| `mesh/mesh.vtu` | the same mesh for PyVista / ParaView |
| `solution/solution.npz` | displacements, per-element compliance and von Mises stress |
| `solution/solution.vtu` | displacement and stress fields for PyVista / ParaView |
| `optimization/density.npz` | the optimized density field (one value per element) |
| `optimization/density.vtu` | the same field for PyVista / ParaView |
| `optimization/topology.stl` | the design thresholded at rho >= 0.5 as a closed surface, for Blender (2D runs extruded by the thickness) |
| `optimization/log.csv` | per-iteration compliance, volume fraction, change, Mnd |
| `optimization/snapshots/` | density field every 10 iterations, for animations |
| `run.json`, `pipeline.log` | structured and plain-text log of the run |
| `figures/*.png` | PyVista views, written by `python -m topocombo.viz` (not by the loop) |

In 2D, the 60 x 20 default gives 1200 quadrilaterals (one design variable each), 1281
nodes and 2562 displacement DOFs, with unit-square elements — the standard
cantilever benchmark discretisation.

At full density under a 1 kN tip load (steel, E = 210 GPa) the solve gives a
compliance of 561.2 N·mm and a tip deflection of 0.5612 mm, against 0.5589 mm
from Timoshenko beam theory — 0.42% apart. The clamped edge carries exactly the
applied load and `||KU - F||` on the free DOFs is ~1e-9, so the assembly and the
boundary conditions are doing what they claim.

The SIMP loop then converges in 60 iterations to a compliance of 903.5 N·mm
using half the material — 1.61x the solid beam's compliance for 0.5x its volume.
The design is the expected cantilever truss: top and bottom flanges with a
triangulated web. Measure of discreteness Mnd = 23%, with 37% of elements fully
solid and 35% void.

In 3D, the 60 x 20 x 1 default gives the same 1200 design variables as
hexahedra, 2562 nodes and 7686 DOFs; the tip load is spread over the two nodes
of the mid-height line across the width. At full density the compliance is
559.2 N·mm, 0.07% from Timoshenko theory for the 20 x 1 mm section (0.35%
stiffer than plane stress, since the width cannot contract freely). The loop
converges in 60 iterations to 899.98 N·mm — the same truss as the 2D run —
and the whole run takes about 15 s. `--nelz` refines the width; at
30 x 10 x 6 the design develops an I-section, which the report's top and end
projections show.

### Report

Each push runs the tests and the 3D pipeline in CI and publishes the procedure
log — the CadQuery input script with a picture of its output, parameters,
per-stage terminal output, mesh validation, the full-density solve, the
optimized density and a shaded view of the optimized topology — to GitHub Pages:
<https://anroleroux.github.io/topocombo/>. The page is a report only; the
scripts are where a run is set up. The Mesh section opens with what the run
was told — mesh size, loads and displacement constraints, by region and with
their node counts — and the study script as it ran; `run.json` records the
same under `params.mesh`, `params.boundary_conditions` and `params.regions`.
The page is generated by
`topocombo.report`, which reads only the artifacts of a run directory, so it is
a downstream consumer like the other visualization paths, not part of the loop.
A 3D density field is shown as projections: the mean through the width (side
view), plus top and end views once the mesh is more than one element deep.
The two CAD pictures are PNGs rendered by `topocombo.cadview` from
`design_domain.brep` and `topology.stl`: a tessellated, Lambert-shaded
axonometric view drawn with matplotlib's Agg backend, so it needs no OpenGL
on the CI runner.

## Layout

```
src/topocombo/
  study.py      a design's study script: Study, Mesh, Fix, Force; load_study / run_study
  geometry.py   design domain: a part script with named regions (CadDomain), or the parametric beam
  regions.py    named regions -> mesh node sets, and how a force spreads over them
  meshing.py    structured (transfinite), body-fitted (unstructured quad / extruded hex) or
                tetrahedral (any solid, regions fused in) meshing (Gmsh)
  mesh_io.py    .msh -> dimension-agnostic Mesh, quality checks, .npz/.vtu export
  elements.py   the element registry (quad4, hex8, tet4, tet10): reference cell, shape functions, quadrature,
                edges / facets; vectorised B matrices, stiffness and measures
  fea.py        linear-elastic solve on any registered element: cached assembly, loads,
                direct or multigrid-CG solve
  optimize.py   SIMP loop: neighbourhood filter, OC update, convergence, log.csv
  pipeline.py   the geometry -> mesh -> solve -> optimize run, terminal-driven
  runlog.py     structured, timed logging of a run
  topology.py   thresholded design -> closed STL surface (for Blender)
  report.py     static HTML report built from a run directory
  cadview.py    `python -m topocombo.cadview`: shaded PNG of a BREP or STL (no OpenGL)
  viz.py        `python -m topocombo.viz`: PyVista views of a run (optional)
  cli.py        `python -m topocombo.cli cad|run|report|all [--dim 3] [--study study.py]`
tests/          mesh invariants, solver verification, optimizer invariants, 2D <-> 3D checks
docs/           the 3D migration plan
examples/       designs as part.py + study.py (`--study`)
```

## Status

The main optimization loop is implemented end to end, in 2D and 3D:
parametric geometry, structured quad or hex meshing with validation, the
custom Q4 plane-stress / H8 / T4 / T10 solid FEA solve, and SIMP compliance minimisation
(sensitivities, sensitivity or density filtering, Optimality Criteria update,
convergence check) — all terminal-driven, writing result artifacts to a run
directory. The published report, the PyVista views and the STL for Blender are
built separately from those artifacts.

The move to 3D followed [`docs/3d-migration-plan.md`](docs/3d-migration-plan.md),
all five steps now done: step 5, solver scaling, arrived with the tetrahedra
(below).

Element code lives in one place: `elements.py` describes each element once
(reference cell, shape-function derivatives, quadrature with weights, edges,
outward facets, the node permutation that un-inverts a cell), and the solver,
mesh reading and checks, load spreading and the STL export all work from that
description, integrating many elements at once. Adding an element is a new
entry there plus a mesher for it; every entry is held to the same tests
(shape functions sum to one, quadrature integrates the reference cell, facets
face outward, the stiffness has exactly the rigid-body modes as its
nullspace). Moving the Q4 and H8 code there changed no result beyond
round-off (compliance within 3e-12 relative, densities within 1e-9, the same
iteration counts on the published study and four parametric runs).

### Tetrahedra

`Mesh(element="tet10")` in a study (or `--element tet10` for the parametric
beam) meshes any 3D part with quadratic tetrahedra; `tet4` gives linear ones,
kept for testing. Gmsh meshes the solid itself — HXT volume meshing, then its
own and Netgen's optimisers, which keep slivers out of thin parts — with
mid-nodes placed on the CAD geometry, curved bores included. Mesh validation
adds a shape-quality check (`6 sqrt(2) V / l_rms^3`, 1 for a regular tet) and
fails below 0.1 with a hint: it trips when elements are larger than the part
is thick (2 mm tets in the 1 mm plate give one sliver of quality 0.004;
1.5 mm gives a worst of 0.31). T10 is CalculiX's C3D10, which keeps the
solver swap open.

Checks, besides the registry invariants every element meets:

* a quadratic displacement field on a distorted T10 (linear on T4) gives its
  exact strain at every quadrature point — shape functions and mid-node order;
* on a 24 x 8 x 2 beam at 1 mm, T10 agrees with a 4x-refined hex mesh to 1.5%;
  T4 of the same size is ~5% too stiff (shear locking), which is why T10 is
  the one to use;
* on the published notched cantilever, T10 at 1.5 mm gives 768.9 N·mm at full
  density against H8's 761.3 (1% apart), and a tip-load line across the free
  end is fused into the solid so it carries nodes (3 corners + 2 mid-nodes on
  a 2 mm-wide beam at 1 mm, loaded 1/12-1/6-1/12 plus 1/3 per mid-node).

`examples/bracket/` is a part hexahedra cannot mesh: a thick flange bolted to
the wall and a narrower arm with a pin hole. The load (1 kN) hangs on the
bore's cylindrical face, the flange's back face is clamped, and 30% of the
material is kept. 2998 T10 at 2.5 mm (17k DOFs) converge in 64 iterations,
about 75 s; the design keeps the flange's bolted corners, a truss in the arm
and a ring round the pin. CI publishes it next to the cantilever, at
`bracket/`, drawn by its front surface — a tet mesh has no layers to show, so
each front-facing boundary facet takes its element's value — and in 3D as the
thresholded STL.

**Solvers.** `Study.solver` (or `--solver`) is `direct` (sparse LU), `cg`
(conjugate gradients preconditioned by smoothed-aggregation algebraic
multigrid, built on the rigid-body modes) or `auto`, the default: direct up
to 30 000 free DOFs, CG above. CG starts each SIMP iteration from the previous
displacements. On a T10 mesh of the published part at 1 mm (55k DOFs) it
gives the same compliances as the direct solve at every iteration and takes
~4 s per iteration against ~6.6 s. Reusing one multigrid hierarchy across
iterations was tried and was slower: the stale preconditioner needs several
times the CG iterations. Assembly is fast for any element: the sparsity
pattern and the map from element entries into it are built once per mesh, so
each iteration's assembly is one weighted `bincount` (0.08 s instead of 1.6 s
at 55k DOFs).

Next, in order:

1. More general boundary conditions: constraints on chosen components and
   prescribed displacements, several load cases, passive regions held solid,
   and a second example that is not a cantilever.

Then NLopt-MMA as an alternative to the OC update, and the CalculiX swap-in
for the solver once the loop is trusted.
