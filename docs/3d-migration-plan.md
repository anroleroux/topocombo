# Moving the pipeline to 3D

The CadQuery → Gmsh → solver → SIMP → artifacts chain stays the same; each
stage changes its representation. 3D is added **next to** 2D (`--dim 3`), not
in place of it: the 2D path is verified against beam theory, fast in CI, and
gives the 3D solver an exact cross-check.

## Default discretisation: one element through the width

The pipeline runs on GitHub Actions and has to stay light, so accuracy through
the width is not a goal yet. The default 3D mesh is **60 x 20 x 1** — the same
1200 design variables as the 2D benchmark, as hexahedra one element thick in
the current out-of-plane direction:

| | 2D (60 x 20) | 3D (60 x 20 x 1) |
| --- | --- | --- |
| elements | 1200 quads | 1200 hexes |
| nodes | 1281 | 2562 |
| DOFs | 2562 | 7686 |
| element matrix | 8 x 8 | 24 x 24 |

A direct sparse solve at ~7.7k DOFs is cheap, so the SIMP loop keeps its
current run time within a small factor. `nelz` stays a parameter, so finer
through-width meshes can be switched on later without code changes.

With one element through the width, Poisson's ratio set to 0 and the load
spread evenly across the width, the 3D solve reproduces the 2D plane-stress
solve (times the width) to round-off, which makes the most useful single test
of the new element.

## What changes, module by module

| Module | 2D today | 3D |
| --- | --- | --- |
| `geometry.py` | planar face, `thickness` is a scalar | a box: length x height x width |
| `meshing.py` | transfinite quads, edges as physical groups | transfinite hexes, faces as physical groups |
| `mesh_io.py` | quads, shoelace area, `line` boundary blocks | hexes, element volume, `quad` boundary blocks |
| `fea.py` | Q4, 3x3 D, 2 DOFs/node, 8x8 ke | H8, 6x6 D, 3 DOFs/node, 24x24 ke |
| `optimize.py` | almost dimension-agnostic (centroid KD-tree) | area -> volume, larger filter neighbourhood |
| `report.py` | per-element SVG polygons | projected views (side / top / end) |

## Steps

Each step is one PR with its own tests.

### 0. Dimension-agnostic interfaces (no behaviour change)
- `QuadMesh` becomes `Mesh`: `nodes (n, dim)`, `cells (m, k)`, `cell_type`,
  `node_sets`, `cell_measures()` (area or volume).
- DOF indexing in `fea.py` uses `dofs_per_node = dim` instead of a hard-coded 2;
  element routines dispatch on `cell_type`.
- `optimize.py` weighs the volume constraint with `cell_measures()`.
- `mesh.npz` stores `cells` and `cell_type` instead of `quads`.
- All existing tests pass unchanged in substance.

### 1. Geometry: `BeamDomain3D`
- `length`, `height`, `width`; `cq.Workplane("XY").box(...)` with the corner at
  the origin; BREP + STEP export.
- The load becomes a line load along the free end at mid-height, spread over the
  width (a point load in 3D gives a stress singularity; with `nelz = 1` it is
  simply split over the two tip nodes).

### 2. Meshing: structured hexes
- `setTransfiniteCurve` on the 12 edges (classified x / y / z),
  `setTransfiniteSurface` + `setRecombine(2, ...)` on the 6 faces,
  `setTransfiniteVolume`, `generate(3)`.
- Physical groups: `design_domain` (dim 3), `fixed` (face x = 0),
  `load_edge` (face x = L).
- `MeshSpec` gains `nelz`, **default 1**.
- Tests: counts, uniform grid, every face classified.

### 3. `mesh_io`: read and check hexes
- Read `hexahedron` blocks; boundary node sets from `quad` surface blocks.
- Orientation via the Jacobian at the element centre (flip inverted hexes).
- `check_mesh`: volumes sum to L*H*W, bounding box, no orphans, non-empty sets.
- `.vtu` with `hexahedron` cells and 3-component displacement.

### 4. FEA: the H8 element
- Isotropic 6x6 D, 6x24 B, 2x2x2 Gauss, 3D von Mises.
- On a uniform structured grid all `ke` are identical: compute once and scale
  by density instead of storing one matrix per element.
- Tests: `ke` symmetric with exactly 6 zero eigenvalues; equilibrium; load
  linearity; tip deflection vs Timoshenko for a b x h section; the
  2D <-> 3D cross-check (nu = 0, `nelz = 1`) to round-off.

### 5. Solver scaling (only when `nelz > 1` is needed)
- Direct `spsolve` is fine at the default size. Beyond ~50-100k DOFs switch to
  CG + AMG (`pyamg`) or CHOLMOD (`scikit-sparse`), warm-started from the
  previous iteration's displacement; log per-iteration solve time.
- Optional: exploit the z mid-plane symmetry.

### 6. Optimizer in 3D
- Mostly parameters after step 0. With `nelz = 1` the filter neighbourhood stays
  in-plane, so behaviour should track the 2D run closely.
- Tests: volume constraint every iteration, beats a uniform design, symmetric
  about mid-height (and mid-width once `nelz > 1`).

### 7. Pipeline + CLI
- `--dim {2,3}`, `--width`, `--nelz` (default 1); record dim / element type in
  `run.json`.

### 8. Result artifacts
- `density.vtu` with hex cells.
- Iso-surface at rho = 0.5 -> `optimization/topology.stl` (for Blender).
- The standalone PyVista script from the README roadmap.

### 9. Report
- Replace per-element polygons with projected views (side x-y, top x-z, end
  y-z) of density, rendered with the existing `_quad_field_svg`. With
  `nelz = 1` the side view is exactly the current figure.
- Optional later: an isometric PNG from off-screen PyVista (needs xvfb/OSMesa).

### 10. CI + docs
- Tests use a coarse 3D mesh (e.g. 12 x 4 x 1); Pages runs the 3D default.
- README: artifacts table, benchmark numbers, layout, status.

## Risks
- **Run time** grows with `nelz`; the default of 1 keeps CI cost near today's.
- **Load modelling** changes from point to line load; the 2D <-> 3D cross-check
  keeps the benchmarks comparable.
- **CalculiX** becomes more attractive in 3D (C3D8 maps onto H8) as an
  independent check of step 4.
