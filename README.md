# topocombo

A topology optimization pipeline built from decoupled, open-source components — a parametric CAD tool, a mesher, an FEA solver, an optimizer, and separate visualization tools.

## Goal

Explore topology optimization (SIMP-based compliance minimization) on a 2D cantilever beam meshed with quadrilateral elements, using an open-source toolchain end to end. The initial focus is a working, terminal-driven optimization loop; visualization is treated as a downstream, optional concern rather than something embedded in the loop itself. The FEA solver starts as a custom, minimal plane-stress implementation, with CalculiX as a planned future swap-in once the core loop is validated.

## Components

- **CAD (geometry)** — [CadQuery](https://github.com/CadQuery/cadquery): parametric definition of the design domain (beam dimensions, aspect ratio, load/support regions), scripted in Python.
- **Mesher** — [Gmsh](https://gmsh.info/): quadrilateral meshing of the CAD geometry (transfinite or recombined mesh), driven via its Python API.
- **FEA solver** — custom 2D plane-stress solver (numpy/scipy: sparse stiffness assembly, direct solve). Chosen over an external solver initially so the optimizer has direct, in-memory access to element stiffness matrices and displacement fields for sensitivity analysis. [CalculiX](http://www.calculix.de/) is the planned later alternative for a verified, general-purpose solver.
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

## Status

Early exploration — no code yet. This README captures the intended architecture before implementation begins.
