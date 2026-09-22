# topocombo

A topology optimization pipeline built from decoupled, open-source components — a parametric CAD tool, a mesher, an FEA solver, an optimizer, and separate visualization tools.

## Goal

Explore topology optimization (SIMP-based compliance minimization) on a 2D cantilever beam meshed with quadrilateral elements, using an open-source toolchain end to end. The initial focus is a working, terminal-driven optimization loop; visualization is treated as a downstream, optional concern rather than something embedded in the loop itself. The FEA solver starts as a custom, minimal plane-stress implementation, with CalculiX as a planned future swap-in once the core loop is validated.

## Components

- **CAD (geometry)** — [CadQuery](https://github.com/CadQuery/cadquery): parametric definition of the design domain (beam dimensions, aspect ratio, load/support regions), scripted in Python.
- **Mesher** — [Gmsh](https://gmsh.info/): quadrilateral meshing of the CAD geometry (transfinite or recombined mesh), driven via its Python API.
- **FEA solver** — custom 2D plane-stress solver (numpy/scipy: sparse stiffness assembly, direct solve), implemented in `src/topocombo/fea.py`. Chosen over an external solver initially so the optimizer has direct, in-memory access to element stiffness matrices and displacement fields for sensitivity analysis. [CalculiX](http://www.calculix.de/) is the planned later alternative for a verified, general-purpose solver.
- **Optimizer** — SIMP (Solid Isotropic Material with Penalization) loop: density update via Optimality Criteria or [NLopt](https://nlopt.readthedocs.io/)'s MMA, with density/sensitivity filtering to avoid checkerboarding. The per-iteration coupling (FEA solve → compliance + sensitivity → filter → update) is custom code.
- **Visualization (decoupled)**:
  - [PyVista](https://pyvista.org/) — scripted plotting of density fields and results, run as a separate process/script against exported data, not called from within the optimization loop.
  - [Blender](https://www.blender.org/) (optional) — presentation-quality rendering of optimized geometry, consuming exported mesh data independently via its Python API (`bpy`) or manual import.

## Flows

### Main optimization loop (terminal, headless)
1. Define parametric geometry in CadQuery → export CAD file.
2. Mesh with Gmsh → quadrilateral mesh.
3. Run the SIMP loop:
   - Assemble stiffness matrix, solve FEA (custom plane-stress solver).
   - Compute compliance and sensitivities.
   - Apply density/sensitivity filter.
   - Update design variables (OC or NLopt-MMA).
   - Check convergence.
4. Print iteration metrics (compliance, volume fraction, design change) to terminal.
5. Periodically write results to disk: density field + mesh (e.g. `.vtu` via `meshio`, or `.npy`) and a scalar log (`log.csv`).

No plotting or rendering happens inside this loop — it only reads geometry/mesh input and writes result artifacts.

### Visualization (separate, on demand)
- **PyVista**: a standalone script reads exported result files from a run directory and produces static or interactive plots of density evolution / final topology.
- **Blender**: a standalone script or manual workflow imports an exported mesh (thresholded density field converted to OBJ/STL/PLY via `meshio`) for polished rendering.

Because both visualization paths consume the same on-disk result artifacts rather than talking to the solver directly, swapping the FEA backend (e.g. to CalculiX later) does not require changes to either visualization script.

## Running the cantilever example

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# CadQuery domain -> Gmsh quad mesh -> plane-stress solve -> artifacts
python -m topocombo.cli run --length 60 --height 20 --nelx 60 --nely 20 \
    --load -1000 --out results/cantilever

# render the run log as a static HTML page
python -m topocombo.cli report --run results/cantilever --site site

# or both at once
python -m topocombo.cli all
```

`gmsh`'s shared library links against GLU, so on a bare Linux box install it first:
`sudo apt-get install libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1`.

Run the tests with `pytest` — they mesh a coarse beam and assert the grid is
uniform, every element is counter-clockwise, the meshed area matches the design
domain, and the boundary node sets land on the right edges; then they check the
solver against rigid-body modes, load linearity, equilibrium of the reactions,
and Timoshenko beam theory on a slender beam.

### Artifacts of a run

| File | Contents |
| --- | --- |
| `cad/design_domain.brep` | design domain, consumed by Gmsh's OCC importer |
| `cad/design_domain.step` | same geometry for exchange with other CAD tools |
| `mesh/beam.msh` | quad mesh with `design_domain`, `fixed` and `load_edge` physical groups |
| `mesh/mesh.npz` | nodes, quad connectivity, boundary node sets, tip-load node — what the solver reads |
| `mesh/mesh.vtu` | the same mesh for PyVista / ParaView |
| `solution/solution.npz` | displacements, per-element compliance and von Mises stress |
| `solution/solution.vtu` | displacement and stress fields for PyVista / ParaView |
| `run.json`, `pipeline.log` | structured and plain-text log of the run |

The 60 x 20 default gives 1200 quadrilaterals (one design variable each), 1281
nodes and 2562 displacement DOFs, with unit-square elements — the standard
cantilever benchmark discretisation.

At full density under a 1 kN tip load (steel, E = 210 GPa) the solve gives a
compliance of 561.2 N·mm and a tip deflection of 0.5612 mm, against 0.5589 mm
from Timoshenko beam theory — 0.42% apart. The clamped edge carries exactly the
applied load and `||KU - F||` on the free DOFs is ~1e-9, so the assembly and the
boundary conditions are doing what they claim.

### Report

Each push runs the pipeline in CI and publishes the procedure log — parameters,
per-stage terminal output, mesh validation and an SVG of the mesh — to GitHub
Pages: <https://anroleroux.github.io/topocombo/>. The page is generated by
`topocombo.report`, which reads only `run.json` and `mesh.npz` from a run
directory, so it is a downstream consumer of artifacts like the other
visualization paths, not part of the loop.

## Layout

```
src/topocombo/
  geometry.py   parametric design domain (CadQuery)
  meshing.py    transfinite quad meshing and physical groups (Gmsh)
  mesh_io.py    .msh -> numpy arrays, mesh quality checks, .npz/.vtu export
  fea.py        Q4 plane-stress solver: element stiffness, assembly, direct solve
  pipeline.py   the geometry -> mesh -> solve run, terminal-driven
  runlog.py     structured, timed logging of a run
  report.py     static HTML report built from a run directory
  cli.py        `python -m topocombo.cli run|report|all`
tests/          mesh invariants and solver verification
```

## Status

Implemented: stages 1-3 of the main flow — parametric geometry, quadrilateral
meshing, mesh validation, and the custom plane-stress FEA solve (sparse
assembly, direct solve, compliance, per-element compliance and von Mises
stress), plus the published run report. The solver already takes an element
density vector with SIMP penalisation, so the optimizer plugs into it directly.

Next: the SIMP loop itself (sensitivities, density/sensitivity filtering, OC or
NLopt-MMA update, convergence) and the PyVista visualization script.
