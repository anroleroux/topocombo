"""End-to-end checks for the geometry → mesh stage."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from topocombo.geometry import BeamDomain
from topocombo.mesh_io import Mesh, load_mesh
from topocombo.geometry import BEAM_FIXED, BEAM_LOAD
from topocombo.meshing import MeshSpec
from topocombo.regions import apply_regions
from topocombo.pipeline import run
from topocombo.report import build_site


@pytest.fixture(scope="module")
def coarse_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    domain = BeamDomain(length=12.0, height=4.0)
    spec = MeshSpec(nelx=6, nely=2)
    log, summary = run(domain=domain, spec=spec, out_dir=out, echo=False)
    return {"dir": out, "domain": domain, "spec": spec, "log": log, "summary": summary}


def test_domain_geometry():
    domain = BeamDomain(length=60.0, height=20.0)
    assert domain.area == pytest.approx(1200.0)
    assert domain.aspect_ratio == pytest.approx(3.0)
    assert domain.load_point == (60.0, 10.0)
    assert domain.face().Area() == pytest.approx(domain.area)


@pytest.mark.parametrize("bad", [{"length": 0.0}, {"height": -1.0}, {"thickness": 0.0}])
def test_domain_rejects_non_positive_dimensions(bad):
    with pytest.raises(ValueError):
        BeamDomain(**bad)


def test_mesh_spec_rejects_empty_grid():
    with pytest.raises(ValueError):
        MeshSpec(nelx=0, nely=4)


def test_cad_artifacts_written(coarse_run):
    cad = coarse_run["dir"] / "cad"
    assert (cad / "design_domain.brep").exists()
    assert (cad / "design_domain.step").exists()
    assert (cad / "design_domain.py").read_text() == coarse_run["domain"].cadquery_script()


def test_mesh_counts_match_spec(coarse_run):
    spec, summary = coarse_run["spec"], coarse_run["summary"]
    assert summary["n_elements"] == spec.n_elements
    assert summary["n_nodes"] == spec.n_nodes
    assert summary["n_dofs"] == 2 * spec.n_nodes
    assert summary["all_checks_passed"]


def test_mesh_is_a_uniform_structured_grid(coarse_run):
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    domain, spec = coarse_run["domain"], coarse_run["spec"]

    areas = mesh.cell_measures()
    assert np.all(areas > 0)  # counter-clockwise everywhere
    assert areas.sum() == pytest.approx(domain.area)
    expected_area = domain.area / spec.n_elements
    assert np.allclose(areas, expected_area)

    edges = mesh.edge_lengths()
    assert np.allclose(edges.max(axis=1) / edges.min(axis=1), 1.0)


def test_boundary_node_sets(coarse_run):
    """The beam's regions pick the clamped edge and the tip-load node; the mesh
    file itself carries no boundary groups any more."""
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    domain, spec = coarse_run["domain"], coarse_run["spec"]
    assert mesh.node_sets == {}
    apply_regions(mesh, domain.regions())

    fixed = mesh.node_sets[BEAM_FIXED]
    load = mesh.node_sets[BEAM_LOAD]
    assert fixed.size == spec.nely + 1
    assert np.allclose(mesh.nodes[fixed, 0], 0.0)
    assert load.size == 1
    assert mesh.nodes[load[0]] == pytest.approx(domain.load_point)


def test_load_node_sits_at_the_free_edge_mid_height(coarse_run):
    summary = coarse_run["summary"]
    assert summary["load_node_coords"] == pytest.approx(list(coarse_run["domain"].load_point))


def test_solver_facing_npz(coarse_run):
    data = np.load(coarse_run["dir"] / "mesh" / "mesh.npz")
    spec = coarse_run["spec"]
    assert data["nodes"].shape == (spec.n_nodes, 2)
    assert data["cells"].shape == (spec.n_elements, 4)
    assert str(data["cell_type"]) == "quad"
    assert f"set_{BEAM_FIXED}" in data
    assert int(data["load_node"][0]) in range(spec.n_nodes)
    assert (coarse_run["dir"] / "mesh" / "mesh.vtu").exists()


def test_mesh_rejects_cells_of_the_wrong_arity():
    nodes = np.zeros((4, 2))
    with pytest.raises(ValueError):
        Mesh(nodes=nodes, cells=np.array([[0, 1, 2]]), node_sets={}, cell_type="quad")
    with pytest.raises(ValueError):
        Mesh(nodes=nodes, cells=np.array([[0, 1, 2, 3]]), node_sets={}, cell_type="pentagon")


def test_mesh_reports_its_dimension(coarse_run):
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    assert mesh.dim == mesh.dofs_per_node == 2
    assert mesh.n_dofs == 2 * mesh.n_nodes
    lower, upper = mesh.bounding_box()
    assert lower.shape == upper.shape == (2,)


def test_run_log_written(coarse_run):
    run_json = json.loads((coarse_run["dir"] / "run.json").read_text())
    names = [s["name"] for s in run_json["steps"]]
    assert names[:4] == ["geometry", "meshing", "validation", "export"]
    assert all(s["status"] == "ok" for s in run_json["steps"])
    assert (coarse_run["dir"] / "pipeline.log").read_text().strip()


def test_report_is_self_contained(coarse_run, tmp_path):
    index = build_site(run_dir=coarse_run["dir"], site_dir=tmp_path / "site")
    html = index.read_text()
    assert "<svg" in html and "</svg>" in html
    assert "1. Define parametric geometry (CadQuery)" in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "")
    assert (tmp_path / "site" / "artifacts" / "mesh.npz").exists()
    assert (tmp_path / "site" / "artifacts" / "run.json").exists()


# --------------------------------------------------------------------------
# plane-stress solver
# --------------------------------------------------------------------------
from topocombo.fea import (  # noqa: E402
    LoadCase,
    Material,
    assemble_stiffness,
    element_dofs,
    element_stiffness,
    element_stiffnesses,
    node_dofs,
    simp_scaling,
    solve,
    timoshenko_tip_deflection,
)


@pytest.fixture(scope="module")
def slender_beam(tmp_path_factory):
    """A slender beam, where Timoshenko theory is a meaningful reference."""
    out = tmp_path_factory.mktemp("slender")
    domain = BeamDomain(length=80.0, height=8.0, thickness=1.0)
    spec = MeshSpec(nelx=80, nely=8)
    _, summary = run(domain=domain, spec=spec, out_dir=out, echo=False)
    mesh = load_mesh(out / "mesh" / "beam.msh")
    apply_regions(mesh, domain.regions())
    material = Material()
    load = LoadCase(node=summary["load_node"], fy=-500.0)
    result = solve(mesh, material, domain.thickness, load, BEAM_FIXED)
    return {"domain": domain, "mesh": mesh, "material": material, "load": load, "result": result}


def test_material_rejects_impossible_constants():
    with pytest.raises(ValueError):
        Material(youngs_modulus=-1.0)
    with pytest.raises(ValueError):
        Material(poisson_ratio=0.5)


def test_element_stiffness_is_symmetric_with_rigid_body_modes():
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    ke = element_stiffness(coords, Material().constitutive_matrix(), thickness=1.0)

    assert np.allclose(ke, ke.T)
    # three rigid-body modes (2 translations + 1 rotation) carry no energy
    tx = np.tile([1.0, 0.0], 4)
    ty = np.tile([0.0, 1.0], 4)
    rot = np.column_stack([-coords[:, 1], coords[:, 0]]).reshape(-1)
    for mode in (tx, ty, rot):
        assert ke @ mode == pytest.approx(np.zeros(8), abs=1e-6 * np.abs(ke).max())
    assert np.linalg.matrix_rank(ke) == 5


def test_stiffness_scales_with_youngs_modulus():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    soft = element_stiffness(coords, Material(youngs_modulus=1.0).constitutive_matrix(), 1.0)
    stiff = element_stiffness(coords, Material(youngs_modulus=3.0).constitutive_matrix(), 1.0)
    assert np.allclose(stiff, 3.0 * soft)


def test_element_dofs_are_node_major_for_any_dofs_per_node():
    cells = np.array([[0, 1, 3, 2], [4, 5, 7, 6]])
    assert element_dofs(cells, 2).tolist() == [
        [0, 1, 2, 3, 6, 7, 4, 5],
        [8, 9, 10, 11, 14, 15, 12, 13],
    ]
    assert element_dofs(cells[:1], 3).tolist() == [[0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]]
    assert node_dofs(np.array([2, 0]), 3).tolist() == [0, 1, 2, 6, 7, 8]


def test_load_case_components_follow_the_mesh_dimension():
    load = LoadCase(node=0, fx=1.0, fy=-2.0, fz=3.0)
    assert load.components(3).tolist() == [1.0, -2.0, 3.0]
    assert load.magnitude == pytest.approx(np.sqrt(14.0))
    assert LoadCase(node=0, fx=1.0, fy=-2.0).components(2).tolist() == [1.0, -2.0]
    with pytest.raises(ValueError):
        load.components(2)  # no out-of-plane load on a 2D mesh


def test_global_stiffness_is_singular_before_boundary_conditions(coarse_run):
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    ke_all = element_stiffnesses(mesh, Material(), thickness=1.0)
    k = assemble_stiffness(mesh, ke_all)
    assert k.shape == (mesh.n_dofs, mesh.n_dofs) == (2 * mesh.n_nodes, 2 * mesh.n_nodes)
    assert abs((k - k.T)).max() < 1e-6 * abs(k).max()
    rigid = np.tile([1.0, 0.0], mesh.n_nodes)  # unconstrained translation
    assert np.abs(k @ rigid).max() < 1e-6 * abs(k).max()


def test_simp_scaling_spans_void_to_solid():
    scale = simp_scaling(np.array([0.0, 0.5, 1.0]), penal=3.0)
    assert scale[0] == pytest.approx(1e-9)
    assert scale[1] == pytest.approx(0.125, rel=1e-6)
    assert scale[2] == pytest.approx(1.0)


def test_solution_satisfies_equilibrium(slender_beam):
    result, load = slender_beam["result"], slender_beam["load"]
    assert result.equilibrium_residual < 1e-6 * abs(load.fy)
    # the clamped edge carries exactly the applied load
    assert result.component(1, "reactions").sum() == pytest.approx(-load.fy, rel=1e-9)
    assert result.component(0, "reactions").sum() == pytest.approx(0.0, abs=1e-6 * abs(load.fy))


def test_compliance_equals_summed_element_compliance(slender_beam):
    result = slender_beam["result"]
    assert result.element_compliance.sum() == pytest.approx(result.compliance, rel=1e-9)
    assert result.compliance > 0.0


def test_tip_deflection_matches_beam_theory(slender_beam):
    domain, material = slender_beam["domain"], slender_beam["material"]
    result, load = slender_beam["result"], slender_beam["load"]

    theory = timoshenko_tip_deflection(
        domain.length, domain.height, domain.thickness, material, load.fy
    )
    tip = abs(result.component(1)[load.node])
    assert tip == pytest.approx(theory["total"], rel=0.05)


def test_solution_scales_linearly_with_load(slender_beam):
    mesh, material, domain = slender_beam["mesh"], slender_beam["material"], slender_beam["domain"]
    base = slender_beam["result"]
    doubled = solve(
        mesh, material, domain.thickness,
        LoadCase(node=slender_beam["load"].node, fy=2.0 * slender_beam["load"].fy),
        BEAM_FIXED,
    )
    assert np.allclose(doubled.u, 2.0 * base.u)
    assert doubled.compliance == pytest.approx(4.0 * base.compliance, rel=1e-9)


def test_density_scaling_softens_the_structure(slender_beam):
    mesh, material, domain = slender_beam["mesh"], slender_beam["material"], slender_beam["domain"]
    half = solve(
        mesh, material, domain.thickness, slender_beam["load"], BEAM_FIXED,
        densities=np.full(mesh.n_elements, 0.5), penal=3.0,
    )
    # uniform density x with penalty p scales stiffness by x^p, so compliance by x^-p
    assert half.compliance == pytest.approx(8.0 * slender_beam["result"].compliance, rel=1e-6)


def test_solution_artifacts_written(coarse_run):
    data = np.load(coarse_run["dir"] / "solution" / "solution.npz")
    spec = coarse_run["spec"]
    assert data["displacements"].shape == (spec.n_nodes, 2)
    assert data["element_compliance"].shape == (spec.n_elements,)
    assert data["von_mises"].shape == (spec.n_elements,)
    assert (coarse_run["dir"] / "solution" / "solution.vtu").exists()


def test_run_records_the_solve(coarse_run):
    run_json = json.loads((coarse_run["dir"] / "run.json").read_text())
    names = [s["name"] for s in run_json["steps"]]
    assert names[:6] == ["geometry", "meshing", "validation", "export", "solve", "solution_export"]
    solve_step = next(s for s in run_json["steps"] if s["name"] == "solve")
    assert solve_step["data"]["compliance"] > 0.0
    assert solve_step["data"]["beam_theory_rel_diff"] < 0.25  # stubby beam, loose bound


# --------------------------------------------------------------------------
# SIMP loop
# --------------------------------------------------------------------------
from topocombo.optimize import (  # noqa: E402
    SimpParams,
    build_filter,
    element_centroids,
    oc_update,
    optimize,
    save_history,
)
from topocombo.report import read_history  # noqa: E402


@pytest.fixture(scope="module")
def optimized(slender_beam):
    """A short, coarse optimization run — enough to exercise every code path."""
    params = SimpParams(
        volume_fraction=0.4, filter_radius=2.0, max_iterations=25, tolerance=0.02
    )
    seen: list[dict[str, float]] = []
    result = optimize(
        mesh=slender_beam["mesh"],
        material=slender_beam["material"],
        thickness=slender_beam["domain"].thickness,
        load=slender_beam["load"],
        fixed_node_set=BEAM_FIXED,
        params=params,
        on_iteration=lambda record, densities: seen.append(record),
    )
    return {**slender_beam, "result": result, "params": params, "seen": seen}


@pytest.mark.parametrize(
    "bad",
    [
        {"volume_fraction": 0.0},
        {"volume_fraction": 1.5},
        {"penal": 0.5},
        {"filter_radius": 0.0},
        {"filter_type": "gaussian"},
    ],
)
def test_simp_params_are_validated(bad):
    with pytest.raises(ValueError):
        SimpParams(**bad)


def test_filter_is_symmetric_and_preserves_a_uniform_field(coarse_run):
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    h, hs = build_filter(mesh, radius=2.0)

    assert abs(h - h.T).max() == 0.0  # weights depend only on distance
    uniform = np.full(mesh.n_elements, 0.37)
    assert np.allclose(np.asarray(h @ uniform).ravel() / hs, uniform)


def test_filter_radius_controls_the_neighbourhood(coarse_run):
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    centroids = element_centroids(mesh)
    small, _ = build_filter(mesh, radius=1.01)
    large, _ = build_filter(mesh, radius=3.0)

    assert small.nnz < large.nnz
    # no weight reaches past the radius
    rows, cols = large.nonzero()
    assert np.linalg.norm(centroids[rows] - centroids[cols], axis=1).max() <= 3.0


def test_oc_update_respects_move_limits_and_bounds():
    rng = np.random.default_rng(0)
    n = 50
    x = np.full(n, 0.5)
    dc = -rng.random(n)
    dv = np.full(n, 1.0 / n)
    x_new = oc_update(
        x=x, dc=dc, dv=dv, volume_fraction=0.4, move=0.1, volume_of=lambda d: float(d.mean())
    )
    assert np.all(x_new >= 0.0) and np.all(x_new <= 1.0)
    assert np.abs(x_new - x).max() <= 0.1 + 1e-12
    assert x_new.mean() == pytest.approx(0.4, abs=1e-6)


def test_optimization_holds_the_volume_constraint(optimized):
    target = optimized["params"].volume_fraction
    assert optimized["result"].volume_fraction == pytest.approx(target, abs=1e-6)
    for record in optimized["seen"]:
        assert record["volume_fraction"] == pytest.approx(target, abs=1e-6)


def test_optimization_reduces_compliance(optimized):
    history = optimized["result"].history
    assert history[-1]["compliance"] < history[0]["compliance"]
    # and beats a uniform design of the same volume, which is the naive alternative
    uniform = solve(
        optimized["mesh"], optimized["material"], optimized["domain"].thickness,
        optimized["load"], BEAM_FIXED,
        densities=np.full(optimized["mesh"].n_elements, optimized["params"].volume_fraction),
        penal=optimized["params"].penal,
    )
    assert optimized["result"].compliance < uniform.compliance


def test_densities_stay_in_bounds_and_report_discreteness(optimized):
    x = optimized["result"].densities
    assert x.min() >= 0.0 and x.max() <= 1.0 + 1e-12
    assert 0.0 <= optimized["result"].measure_of_discreteness() <= 100.0


def test_design_is_symmetric_about_mid_height(optimized):
    """The load sits at mid-height, so the optimum must be top-bottom symmetric."""
    mesh = optimized["mesh"]
    height = optimized["domain"].height
    centroids = element_centroids(mesh)
    mirrored = np.column_stack([centroids[:, 0], height - centroids[:, 1]])

    tree = cKDTree(centroids)
    dist, partner = tree.query(mirrored)
    assert dist.max() < 1e-6  # every element has a mirror image in this mesh

    x = optimized["result"].densities
    assert np.abs(x - x[partner]).max() < 1e-6


def test_history_csv_roundtrips(optimized, tmp_path):
    path = save_history(optimized["result"].history, tmp_path / "log.csv")
    rows = read_history(path)
    assert len(rows) == len(optimized["result"].history)
    assert rows[-1]["compliance"] == pytest.approx(
        optimized["result"].history[-1]["compliance"], rel=1e-9
    )
    assert path.read_text().splitlines()[0].startswith("iteration,compliance")


def test_pipeline_writes_the_design(coarse_run):
    data = np.load(coarse_run["dir"] / "optimization" / "density.npz")
    assert data["densities"].shape == (coarse_run["spec"].n_elements,)
    assert (coarse_run["dir"] / "optimization" / "density.vtu").exists()
    assert (coarse_run["dir"] / "optimization" / "log.csv").exists()

    run_json = json.loads((coarse_run["dir"] / "run.json").read_text())
    names = [s["name"] for s in run_json["steps"]]
    assert names[-2:] == ["optimize", "design_export"]


# --------------------------------------------------------------------------
# 3D design domain and hexahedral mesh
# --------------------------------------------------------------------------
import meshio  # noqa: E402

from topocombo.geometry import BeamDomain3D, export_domain  # noqa: E402
from topocombo.meshing import PHYS_DOMAIN, MeshSpec3D, generate_hex_mesh  # noqa: E402


def test_domain_3d_geometry():
    domain = BeamDomain3D(length=60.0, height=20.0, width=2.0)
    assert domain.volume == pytest.approx(2400.0)
    assert domain.load_point == (60.0, 10.0, 1.0)
    assert domain.load_line == ((60.0, 10.0, 0.0), (60.0, 10.0, 2.0))
    solid = domain.solid()
    assert solid.Volume() == pytest.approx(domain.volume)
    bb = solid.BoundingBox()
    assert (bb.xmin, bb.ymin, bb.zmin) == pytest.approx((0.0, 0.0, 0.0))
    assert (bb.xmax, bb.ymax, bb.zmax) == pytest.approx((60.0, 20.0, 2.0))


@pytest.mark.parametrize("bad", [{"length": 0.0}, {"height": -1.0}, {"width": 0.0}])
def test_domain_3d_rejects_non_positive_dimensions(bad):
    with pytest.raises(ValueError):
        BeamDomain3D(**bad)


def test_mesh_spec_3d_defaults_to_one_element_through_the_width():
    spec = MeshSpec3D()
    assert (spec.nelx, spec.nely, spec.nelz) == (60, 20, 1)
    assert spec.n_elements == 1200
    assert spec.n_nodes == 61 * 21 * 2
    with pytest.raises(ValueError):
        MeshSpec3D(nelz=0)


@pytest.fixture(scope="module", params=[1, 2], ids=["nelz1", "nelz2"])
def hex_run(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"hex{request.param}")
    domain = BeamDomain3D(length=12.0, height=4.0, width=2.0)
    spec = MeshSpec3D(nelx=6, nely=2, nelz=request.param)
    cad = export_domain(domain, out / "cad")
    msh_path, gmsh_log = generate_hex_mesh(domain, spec, cad["brep"], out / "mesh")
    return {
        "domain": domain,
        "spec": spec,
        "cad": cad,
        "msh_path": msh_path,
        "msh": meshio.read(str(msh_path)),
        "gmsh_log": gmsh_log,
    }


def _group_cells(msh: meshio.Mesh, name: str, cell_type: str) -> np.ndarray:
    tag = int(msh.field_data[name][0])
    rows = [
        block.data[np.asarray(phys) == tag]
        for block, phys in zip(msh.cells, msh.cell_data["gmsh:physical"])
        if block.type == cell_type
    ]
    return np.vstack(rows) if rows else np.empty((0, 0), dtype=int)


def test_cad_3d_artifacts_written(hex_run):
    for path in hex_run["cad"].values():
        assert path.exists() and path.stat().st_size > 0
    assert hex_run["gmsh_log"]


def test_hex_mesh_counts_match_spec(hex_run):
    msh, spec = hex_run["msh"], hex_run["spec"]
    hexes = _group_cells(msh, PHYS_DOMAIN, "hexahedron")
    assert hexes.shape == (spec.n_elements, 8)
    assert np.unique(hexes).size == spec.n_nodes  # every node belongs to a hex
    assert {b.type for b in msh.cells} <= {"hexahedron", "quad"}  # no tets or prisms


def test_hex_mesh_is_a_uniform_structured_grid(hex_run):
    msh, domain, spec = hex_run["msh"], hex_run["domain"], hex_run["spec"]
    xyz = msh.points[_group_cells(msh, PHYS_DOMAIN, "hexahedron")]  # (m, 8, 3)
    size = xyz.max(axis=1) - xyz.min(axis=1)
    cell = np.array([domain.length / spec.nelx, domain.height / spec.nely, domain.width / spec.nelz])
    assert np.allclose(size, cell)  # axis-aligned bricks of one size
    assert np.prod(size, axis=1).sum() == pytest.approx(domain.volume)
    # each brick has exactly its 8 corners: 2 distinct values per axis
    for axis in range(3):
        assert all(np.unique(np.round(c, 9)).size == 2 for c in xyz[:, :, axis])


def test_hex_boundary_faces(hex_run):
    """The clamped face region picks the whole x = 0 face; the load region, a
    line across the free end, picks its mid-height nodes."""
    domain, spec = hex_run["domain"], hex_run["spec"]
    mesh = load_mesh(hex_run["msh_path"])
    apply_regions(mesh, domain.regions())
    fixed, loaded = mesh.node_sets[BEAM_FIXED], mesh.node_sets[BEAM_LOAD]
    assert fixed.size == (spec.nely + 1) * (spec.nelz + 1)
    assert np.allclose(mesh.nodes[fixed, 0], 0.0)
    assert loaded.size == spec.nelz + 1
    assert np.allclose(mesh.nodes[loaded, :2], [domain.length, domain.height / 2.0])


# --------------------------------------------------------------------------
# reading and checking the hex mesh (step 3)
# --------------------------------------------------------------------------
from topocombo.fea import (  # noqa: E402
    centroid_von_mises,
    hex_element_stiffness,
    load_vector,
)
from topocombo.mesh_io import (  # noqa: E402
    check_mesh,
    find_node,
    nodes_on_segment,
    orient_cells,
    save_mesh,
    save_solution,
)
from topocombo.meshing import generate_quad_mesh  # noqa: E402

UNIT_CUBE = np.array(
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    dtype=float,
)


def test_load_mesh_reads_hexes_and_face_node_sets(hex_run):
    mesh = load_mesh(hex_run["msh_path"])
    domain, spec = hex_run["domain"], hex_run["spec"]
    assert mesh.cell_type == "hexahedron"
    assert mesh.dim == mesh.dofs_per_node == 3
    assert mesh.n_elements == spec.n_elements
    assert mesh.n_nodes == spec.n_nodes
    assert mesh.n_dofs == 3 * spec.n_nodes

    volumes = mesh.cell_measures()
    assert np.allclose(volumes, domain.volume / spec.n_elements)  # all positive, all equal

    edges = mesh.edge_lengths()
    assert edges.shape == (spec.n_elements, 12)
    cell = [domain.length / spec.nelx, domain.height / spec.nely, domain.width / spec.nelz]
    assert np.allclose(np.sort(np.unique(np.round(edges, 9))), np.unique(np.round(cell, 9)))


def test_check_mesh_validates_a_hex_mesh(hex_run):
    mesh = load_mesh(hex_run["msh_path"])
    summary = check_mesh(mesh, hex_run["domain"], hex_run["spec"].n_elements)
    assert summary["all_checks_passed"], summary["checks"]
    assert summary["dim"] == 3 and summary["cell_type"] == "hexahedron"
    assert summary["volume_sum"] == pytest.approx(hex_run["domain"].volume)
    assert "all_elements_positive_volume" in summary["checks"]
    assert len(summary["bounding_box"]) == 6
    with pytest.raises(ValueError):
        check_mesh(mesh, BeamDomain(length=12.0, height=4.0), hex_run["spec"].n_elements)


def test_orient_cells_turns_inverted_hexes_right_way_round():
    inverted = np.array([[4, 5, 6, 7, 0, 1, 2, 3]])  # top face first: a mirrored hex
    assert Mesh(UNIT_CUBE, inverted, {}, "hexahedron").cell_measures()[0] == pytest.approx(-1.0)
    fixed = orient_cells(UNIT_CUBE, inverted, "hexahedron")
    assert Mesh(UNIT_CUBE, fixed, {}, "hexahedron").cell_measures()[0] == pytest.approx(1.0)
    assert sorted(fixed[0]) == list(range(8))
    good = np.arange(8)[None, :]
    assert np.array_equal(orient_cells(UNIT_CUBE, good, "hexahedron"), good)


def test_mesh_rejects_nodes_of_the_wrong_dimension():
    with pytest.raises(ValueError):
        Mesh(UNIT_CUBE[:, :2], np.arange(8)[None, :], {}, "hexahedron")


def test_load_line_is_picked_from_the_loaded_face(hex_run):
    mesh = load_mesh(hex_run["msh_path"])
    domain, spec = hex_run["domain"], hex_run["spec"]
    apply_regions(mesh, domain.regions())
    line = nodes_on_segment(mesh, *domain.load_line, candidates=mesh.node_sets[BEAM_LOAD])
    assert line.size == spec.nelz + 1
    assert np.allclose(mesh.nodes[line, :2], [domain.length, domain.height / 2.0])
    assert np.all(np.diff(mesh.nodes[line, 2]) > 0)  # ordered from the start point


def test_hex_mesh_artifacts(hex_run, tmp_path):
    mesh = load_mesh(hex_run["msh_path"])
    paths = save_mesh(mesh, tmp_path, load_node=find_node(mesh, hex_run["domain"].load_point))
    data = np.load(paths["npz"])
    assert data["nodes"].shape == (mesh.n_nodes, 3)
    assert data["cells"].shape == (mesh.n_elements, 8)
    assert str(data["cell_type"]) == "hexahedron"
    vtu = meshio.read(str(paths["vtu"]))
    assert [b.type for b in vtu.cells] == ["hexahedron"]


# --------------------------------------------------------------------------
# the H8 solid element (step 4)
# --------------------------------------------------------------------------
def _hex_mesh(out, domain: BeamDomain3D, spec: MeshSpec3D) -> Mesh:
    cad = export_domain(domain, out / "cad")
    msh, _ = generate_hex_mesh(domain, spec, cad["brep"], out / "mesh")
    mesh = load_mesh(msh)
    apply_regions(mesh, domain.regions())
    return mesh


def _quad_mesh(out, domain: BeamDomain, spec: MeshSpec) -> Mesh:
    cad = export_domain(domain, out / "cad")
    msh, _ = generate_quad_mesh(domain, spec, cad["brep"], out / "mesh")
    mesh = load_mesh(msh)
    apply_regions(mesh, domain.regions())
    return mesh


def _tip_line_load(mesh: Mesh, domain: BeamDomain3D, fy: float) -> LoadCase:
    line = nodes_on_segment(mesh, *domain.load_line, candidates=mesh.node_sets[BEAM_LOAD])
    return LoadCase.along_line(mesh, line, fy=fy)


def test_hex_stiffness_is_symmetric_with_six_rigid_body_modes():
    coords = UNIT_CUBE * [2.0, 1.0, 0.5] + 0.1 * UNIT_CUBE[:, [1, 2, 0]]  # a sheared brick
    ke = hex_element_stiffness(coords, Material().constitutive_matrix_3d())
    assert np.allclose(ke, ke.T)

    x, y, z = coords.T
    zero = np.zeros(8)
    modes = [
        np.tile([1.0, 0.0, 0.0], 8),
        np.tile([0.0, 1.0, 0.0], 8),
        np.tile([0.0, 0.0, 1.0], 8),
        np.column_stack([-y, x, zero]).reshape(-1),  # about z
        np.column_stack([zero, -z, y]).reshape(-1),  # about x
        np.column_stack([z, zero, -x]).reshape(-1),  # about y
    ]
    for mode in modes:
        assert ke @ mode == pytest.approx(np.zeros(24), abs=1e-6 * np.abs(ke).max())
    assert np.linalg.matrix_rank(ke) == 18


def test_hex_stiffness_scales_with_youngs_modulus():
    soft = hex_element_stiffness(UNIT_CUBE, Material(youngs_modulus=1.0).constitutive_matrix_3d())
    stiff = hex_element_stiffness(UNIT_CUBE, Material(youngs_modulus=3.0).constitutive_matrix_3d())
    assert np.allclose(stiff, 3.0 * soft)


def test_hex_reproduces_a_uniform_strain_state():
    """u = eps * x (uniaxial strain): exact energy and von Mises on one element."""
    material = Material()
    lam = material.constitutive_matrix_3d()[0, 1]
    mu = material.shear_modulus
    eps = 1e-3
    size = np.array([2.0, 1.0, 3.0])
    nodes = UNIT_CUBE * size
    mesh = Mesh(nodes, np.arange(8)[None, :], {}, "hexahedron")
    u = np.column_stack([eps * nodes[:, 0], np.zeros(8), np.zeros(8)]).reshape(-1)

    ke = hex_element_stiffness(nodes, material.constitutive_matrix_3d())
    assert u @ ke @ u == pytest.approx(np.prod(size) * (lam + 2.0 * mu) * eps**2, rel=1e-12)
    # sxx = (lam + 2mu) eps, syy = szz = lam eps  ->  von Mises = 2 mu eps
    vm = centroid_von_mises(mesh, u, material)
    assert vm[0] == pytest.approx(2.0 * mu * eps, rel=1e-12)


def test_line_load_uses_tributary_shares():
    # a column of three unit cubes along z; the load runs up its x = y = 0 edge
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    nodes = np.vstack([np.column_stack([corners, np.full(4, z)]) for z in range(4)])
    cells = np.array([[*range(4 * k, 4 * k + 8)] for k in range(3)])
    mesh = Mesh(nodes, cells, {}, "hexahedron")
    line = np.array([0, 4, 8, 12])
    load = LoadCase.along_line(mesh, line, fy=-600.0)
    assert load.node_shares() == pytest.approx([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    f = load_vector(mesh, load).reshape(-1, 3)
    assert f[line, 1] == pytest.approx([-100.0, -200.0, -200.0, -100.0])
    assert f[:, 1].sum() == pytest.approx(-600.0)
    assert f[:, [0, 2]] == pytest.approx(0.0)
    assert load.as_dict()["node"] == [0, 4, 8, 12]
    with pytest.raises(ValueError):
        LoadCase(node=(0, 1), shares=(0.3, 0.3))


def test_hex_solve_matches_plane_stress_with_one_element_through_the_width(tmp_path):
    """With nu = 0 and one element through the width, H8 == Q4 x width, exactly."""
    material = Material(poisson_ratio=0.0)
    width = 1.5
    domain2 = BeamDomain(length=20.0, height=4.0, thickness=width)
    domain3 = BeamDomain3D(length=20.0, height=4.0, width=width)
    mesh2 = _quad_mesh(tmp_path / "q4", domain2, MeshSpec(nelx=20, nely=4))
    mesh3 = _hex_mesh(tmp_path / "h8", domain3, MeshSpec3D(nelx=20, nely=4, nelz=1))

    r2 = solve(mesh2, material, width, LoadCase(node=find_node(mesh2, domain2.load_point), fy=-1000.0), BEAM_FIXED)
    load3 = _tip_line_load(mesh3, domain3, fy=-1000.0)
    assert load3.node_shares() == pytest.approx([0.5, 0.5])
    r3 = solve(mesh3, material, width, load3, BEAM_FIXED)

    assert r3.compliance == pytest.approx(r2.compliance, rel=1e-10)

    # every 3D node moves exactly like the 2D node at the same (x, y); no z motion
    tree = cKDTree(mesh2.nodes)
    _, match = tree.query(mesh3.nodes[:, :2])
    u2 = r2.u.reshape(-1, 2)[match]
    u3 = r3.u.reshape(-1, 3)
    scale = np.abs(u2).max()
    assert np.abs(u3[:, :2] - u2).max() < 1e-9 * scale
    assert np.abs(u3[:, 2]).max() < 1e-9 * scale

    # element by element, too
    _, cell_match = cKDTree(mesh2.nodes[mesh2.cells].mean(axis=1)).query(
        mesh3.nodes[mesh3.cells].mean(axis=1)[:, :2]
    )
    assert np.allclose(r3.element_compliance, r2.element_compliance[cell_match], rtol=1e-8)
    assert np.allclose(r3.von_mises, r2.von_mises[cell_match], rtol=1e-8)


@pytest.fixture(scope="module")
def slender_hex_beam(tmp_path_factory):
    out = tmp_path_factory.mktemp("slender_hex")
    domain = BeamDomain3D(length=80.0, height=8.0, width=1.0)
    mesh = _hex_mesh(out, domain, MeshSpec3D(nelx=80, nely=8, nelz=1))
    material = Material()
    load = _tip_line_load(mesh, domain, fy=-500.0)
    result = solve(mesh, material, domain.width, load, BEAM_FIXED)
    return {"domain": domain, "mesh": mesh, "material": material, "load": load, "result": result, "dir": out}


def test_hex_solution_satisfies_equilibrium(slender_hex_beam):
    result, load = slender_hex_beam["result"], slender_hex_beam["load"]
    assert result.dofs_per_node == 3
    assert result.equilibrium_residual < 1e-6 * abs(load.fy)
    assert result.component(1, "reactions").sum() == pytest.approx(-load.fy, rel=1e-9)
    for axis in (0, 2):
        assert result.component(axis, "reactions").sum() == pytest.approx(0.0, abs=1e-6 * abs(load.fy))
    assert result.element_compliance.sum() == pytest.approx(result.compliance, rel=1e-9)


def test_hex_tip_deflection_matches_beam_theory(slender_hex_beam):
    domain, material = slender_hex_beam["domain"], slender_hex_beam["material"]
    result, load = slender_hex_beam["result"], slender_hex_beam["load"]
    theory = timoshenko_tip_deflection(domain.length, domain.height, domain.width, material, load.fy)
    tip = abs(result.component(1)[load.nodes].mean())
    assert tip == pytest.approx(theory["total"], rel=0.05)


def test_hex_solution_is_linear_and_softens_with_density(slender_hex_beam):
    mesh, material, domain = slender_hex_beam["mesh"], slender_hex_beam["material"], slender_hex_beam["domain"]
    base, load = slender_hex_beam["result"], slender_hex_beam["load"]
    doubled = solve(
        mesh, material, domain.width,
        LoadCase(node=load.node, fy=2.0 * load.fy, shares=load.shares), BEAM_FIXED,
    )
    assert np.allclose(doubled.u, 2.0 * base.u)
    half = solve(
        mesh, material, domain.width, load, BEAM_FIXED,
        densities=np.full(mesh.n_elements, 0.5), penal=3.0,
    )
    assert half.compliance == pytest.approx(8.0 * base.compliance, rel=1e-6)


def test_hex_solution_artifacts(slender_hex_beam, tmp_path):
    mesh, result = slender_hex_beam["mesh"], slender_hex_beam["result"]
    paths = save_solution(mesh, result, tmp_path)
    data = np.load(paths["npz"])
    assert data["displacements"].shape == (mesh.n_nodes, 3)
    assert data["reactions"].shape == (mesh.n_nodes, 3)
    assert data["von_mises"].shape == (mesh.n_elements,)
    vtu = meshio.read(str(paths["vtu"]))
    assert vtu.point_data["displacement"].shape == (mesh.n_nodes, 3)


# --------------------------------------------------------------------------
# the SIMP loop on hexahedra (step 6)
# --------------------------------------------------------------------------
def test_hex_optimization_matches_the_2d_design_with_one_element_through_the_width(tmp_path):
    """nu = 0, nelz = 1: every iteration of the 3D loop reproduces the 2D loop."""
    material = Material(poisson_ratio=0.0)
    width = 1.0
    params = SimpParams(volume_fraction=0.4, filter_radius=1.5, max_iterations=15, tolerance=1e-4)
    domain2 = BeamDomain(length=30.0, height=10.0, thickness=width)
    domain3 = BeamDomain3D(length=30.0, height=10.0, width=width)
    mesh2 = _quad_mesh(tmp_path / "q4", domain2, MeshSpec(nelx=30, nely=10))
    mesh3 = _hex_mesh(tmp_path / "h8", domain3, MeshSpec3D(nelx=30, nely=10, nelz=1))

    load2 = LoadCase(node=find_node(mesh2, domain2.load_point), fy=-1000.0)
    design2 = optimize(mesh2, material, width, load2, BEAM_FIXED, params)
    design3 = optimize(mesh3, material, width, _tip_line_load(mesh3, domain3, -1000.0), BEAM_FIXED, params)

    assert design3.iterations == design2.iterations
    c2 = [h["compliance"] for h in design2.history]
    c3 = [h["compliance"] for h in design3.history]
    assert c3 == pytest.approx(c2, rel=1e-8)
    _, match = cKDTree(element_centroids(mesh2)).query(element_centroids(mesh3)[:, :2])
    assert np.abs(design3.densities - design2.densities[match]).max() < 1e-6


@pytest.fixture(scope="module")
def optimized_hex(tmp_path_factory):
    """Two elements through the width, so the width direction is exercised too."""
    out = tmp_path_factory.mktemp("opt_hex")
    domain = BeamDomain3D(length=24.0, height=8.0, width=2.0)
    mesh = _hex_mesh(out, domain, MeshSpec3D(nelx=24, nely=8, nelz=2))
    material = Material()
    load = _tip_line_load(mesh, domain, fy=-1000.0)
    params = SimpParams(volume_fraction=0.4, filter_radius=1.5, max_iterations=15, tolerance=0.02)
    seen: list[dict[str, float]] = []
    result = optimize(
        mesh, material, domain.width, load, BEAM_FIXED, params,
        on_iteration=lambda record, densities: seen.append(record),
    )
    return {"domain": domain, "mesh": mesh, "material": material, "load": load,
            "params": params, "result": result, "seen": seen}


def test_hex_optimization_holds_the_volume_constraint(optimized_hex):
    target = optimized_hex["params"].volume_fraction
    assert all(r["volume_fraction"] == pytest.approx(target, abs=1e-6) for r in optimized_hex["seen"])
    d = optimized_hex["result"].densities
    assert d.min() >= 0.0 and d.max() <= 1.0


def test_hex_optimization_beats_a_uniform_design(optimized_hex):
    o = optimized_hex
    uniform = solve(
        o["mesh"], o["material"], o["domain"].width, o["load"], BEAM_FIXED,
        densities=np.full(o["mesh"].n_elements, o["params"].volume_fraction), penal=o["params"].penal,
    )
    assert o["result"].compliance < 0.5 * uniform.compliance


def test_hex_design_is_symmetric_about_mid_height_and_mid_width(optimized_hex):
    mesh, domain = optimized_hex["mesh"], optimized_hex["domain"]
    d = optimized_hex["result"].densities
    c = element_centroids(mesh)
    tree = cKDTree(c)
    for mirror in ([1.0, -1.0, 1.0], [1.0, 1.0, -1.0]):
        reflected = c * mirror + [0.0, domain.height, domain.width] * (np.array(mirror) < 0)
        dist, partner = tree.query(reflected)
        assert dist.max() < 1e-9
        assert np.abs(d - d[partner]).max() < 1e-6


# --------------------------------------------------------------------------
# the 3D pipeline and CLI (step 7)
# --------------------------------------------------------------------------
from topocombo.cli import main as cli_main  # noqa: E402


@pytest.fixture(scope="module")
def coarse_run_3d(tmp_path_factory):
    out = tmp_path_factory.mktemp("run3d")
    domain = BeamDomain3D(length=12.0, height=4.0, width=1.0)
    spec = MeshSpec3D(nelx=12, nely=4, nelz=1)
    simp = SimpParams(max_iterations=5)
    log, summary = run(domain=domain, spec=spec, out_dir=out, simp=simp, echo=False)
    return {"dir": out, "domain": domain, "spec": spec, "summary": summary}


def test_pipeline_runs_in_3d(coarse_run_3d):
    run_json = json.loads((coarse_run_3d["dir"] / "run.json").read_text())
    assert run_json["params"]["dim"] == 3
    assert run_json["params"]["element"] == "H8 hexahedron"
    assert all(s["status"] == "ok" for s in run_json["steps"])
    names = [s["name"] for s in run_json["steps"]]
    assert names == ["geometry", "meshing", "validation", "export", "solve",
                     "solution_export", "optimize", "design_export"]

    summary, spec = coarse_run_3d["summary"], coarse_run_3d["spec"]
    assert summary["all_checks_passed"]
    assert summary["n_dofs"] == 3 * spec.n_nodes
    assert summary["volume_sum"] == pytest.approx(coarse_run_3d["domain"].volume)
    assert len(summary["load_nodes"]) == spec.nelz + 1
    solve_data = summary["solve"]
    assert solve_data["load"]["shares"] == pytest.approx([0.5, 0.5])
    assert solve_data["reaction_y"] == pytest.approx(1000.0, rel=1e-9)
    assert summary["optimization"]["iterations"] == 5


def test_pipeline_3d_artifacts(coarse_run_3d):
    out, spec = coarse_run_3d["dir"], coarse_run_3d["spec"]
    mesh_data = np.load(out / "mesh" / "mesh.npz")
    assert mesh_data["nodes"].shape == (spec.n_nodes, 3)
    assert mesh_data["cells"].shape == (spec.n_elements, 8)
    assert mesh_data["load_nodes"].size == spec.nelz + 1
    assert np.load(out / "solution" / "solution.npz")["displacements"].shape == (spec.n_nodes, 3)
    assert np.load(out / "optimization" / "density.npz")["densities"].shape == (spec.n_elements,)
    vtu = meshio.read(str(out / "optimization" / "density.vtu"))
    assert [b.type for b in vtu.cells] == ["hexahedron"]


def test_report_renders_a_3d_run(coarse_run_3d, tmp_path):
    index = build_site(run_dir=coarse_run_3d["dir"], site_dir=tmp_path / "site")
    html = index.read_text()
    assert html.count("<svg") >= 3  # mesh, solution and density figures
    assert "hexahedra" in html
    assert "3D solid solve" in html and "Plane-stress solve" not in html


def test_report_side_view_of_a_hex_mesh_tiles_the_domain(coarse_run_3d):
    """Each hex is drawn by its z-normal face, counter-clockwise, covering L x H."""
    from topocombo.report import _side_view

    data = np.load(coarse_run_3d["dir"] / "mesh" / "mesh.npz")
    nodes, quads, drawn = _side_view(data["nodes"], data["cells"])
    assert nodes.shape[1] == 2 and quads.shape == (data["cells"].shape[0], 4)
    xy = nodes[quads]
    x, y = xy[:, :, 0], xy[:, :, 1]
    areas = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    domain = coarse_run_3d["domain"]
    assert np.all(areas > 0)
    assert areas.sum() == pytest.approx(domain.length * domain.height)


def test_pipeline_rejects_a_mismatched_domain_and_spec(tmp_path):
    with pytest.raises(TypeError):
        run(domain=BeamDomain3D(), spec=MeshSpec(), out_dir=tmp_path, echo=False)


def test_cli_runs_the_3d_pipeline(tmp_path, capsys):
    out = tmp_path / "cli3d"
    code = cli_main([
        "run", "--dim", "3", "--length", "8", "--height", "4", "--width", "2",
        "--nelx", "4", "--nely", "2", "--nelz", "2", "--no-optimize", "--out", str(out),
    ])
    assert code == 0
    run_json = json.loads((out / "run.json").read_text())
    assert run_json["params"]["dim"] == 3
    assert run_json["params"]["mesh"]["nelz"] == 2
    assert run_json["params"]["domain"]["width"] == 2.0
    assert np.load(out / "mesh" / "mesh.npz")["cells"].shape == (4 * 2 * 2, 8)


# --------------------------------------------------------------------------
# result artifacts: the topology surface and the PyVista views (step 8)
# --------------------------------------------------------------------------
from topocombo.topology import (  # noqa: E402
    boundary_faces,
    enclosed_volume,
    extrude,
    save_topology_stl,
    solid_surface,
)
from topocombo.viz import available_views  # noqa: E402


def _brick_grid(nx: int, ny: int, nz: int) -> Mesh:
    """A unit-cube hex grid built by hand, in Gmsh/VTK corner order."""
    idx = np.arange((nx + 1) * (ny + 1) * (nz + 1)).reshape(nx + 1, ny + 1, nz + 1)
    gx, gy, gz = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1), indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]).astype(float)
    cells = [
        [idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
         idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1]]
        for i in range(nx) for j in range(ny) for k in range(nz)
    ]
    return Mesh(nodes, np.array(cells), {}, "hexahedron")


def _edge_counts(triangles: np.ndarray) -> np.ndarray:
    edges = np.sort(np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    return np.unique(edges, axis=0, return_counts=True)[1]


def test_shared_faces_are_dropped_from_the_surface():
    mesh = _brick_grid(2, 1, 1)
    assert boundary_faces(mesh.cells).shape == (10, 4)  # 12 faces, the shared one twice
    points, triangles = solid_surface(mesh, np.ones(2))
    assert triangles.shape == (20, 3)
    assert enclosed_volume(points, triangles) == pytest.approx(2.0)  # closed and outward
    assert np.all(_edge_counts(triangles) == 2)  # watertight: every edge has two sides


def test_surface_of_a_thresholded_design_encloses_the_solid_elements():
    mesh = _brick_grid(4, 3, 2)
    rng = np.random.default_rng(3)
    densities = rng.random(mesh.n_elements)
    points, triangles = solid_surface(mesh, densities, threshold=0.5)
    assert enclosed_volume(points, triangles) == pytest.approx(float((densities >= 0.5).sum()))


def test_extruding_a_quad_mesh_gives_positive_hexes():
    nodes = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    quad = Mesh(nodes, np.array([[0, 1, 2, 3]]), {}, "quad")
    hexes = extrude(quad, thickness=0.5)
    assert hexes.cell_type == "hexahedron" and hexes.n_nodes == 8
    assert hexes.cell_measures() == pytest.approx([1.0])
    with pytest.raises(ValueError):
        extrude(hexes, 1.0)


@pytest.mark.parametrize("which", ["coarse_run", "coarse_run_3d"])
def test_pipeline_writes_a_closed_topology_stl(which, request):
    run_fixture = request.getfixturevalue(which)
    out = run_fixture["dir"]
    stl = meshio.read(str(out / "optimization" / "topology.stl"))
    triangles = stl.cells[0].data
    assert triangles.size and np.all(_edge_counts(triangles) >= 2)
    data = json.loads((out / "run.json").read_text())
    step = next(s for s in data["steps"] if s["name"] == "design_export")
    topo = step["data"]["topology"]
    assert topo["volume"] == pytest.approx(topo["solid_element_volume"], rel=1e-9)
    assert enclosed_volume(stl.points, triangles) == pytest.approx(topo["volume"], rel=1e-6)


def test_viz_finds_the_views_of_a_run(coarse_run_3d, tmp_path):
    views = available_views(coarse_run_3d["dir"])
    assert set(views) == {"topology", "density", "solution"}
    assert available_views(tmp_path) == {}


def test_viz_renders_pngs_off_screen(coarse_run_3d, tmp_path):
    pytest.importorskip("pyvista", reason="PyVista is the optional [viz] extra")
    from topocombo.viz import render

    try:
        written = render(coarse_run_3d["dir"], tmp_path, window_size=(400, 200))
    except RuntimeError as err:  # pragma: no cover - no off-screen OpenGL on this machine
        pytest.skip(f"PyVista cannot render off screen here: {err}")
    assert set(written) == {"topology", "density", "solution"}
    assert all(p.stat().st_size > 0 for p in written.values())


# --------------------------------------------------------------------------
# projected density views in the report (step 9)
# --------------------------------------------------------------------------
from topocombo.report import PROJECTIONS, density_svg, project_field  # noqa: E402


def test_projection_averages_through_the_hidden_axis():
    mesh = _brick_grid(3, 2, 2)
    field = np.arange(mesh.n_elements, dtype=float)
    points, rects, means = project_field(mesh.nodes, mesh.cells, field, *PROJECTIONS["side"][:3])
    assert rects.shape == (6, 4) and means.shape == (6,)
    # elements are ordered k fastest, so each side column averages a consecutive pair
    assert sorted(means) == pytest.approx(sorted(field.reshape(-1, 2).mean(axis=1)))
    assert points.shape == (24, 2)
    _, _, top = project_field(mesh.nodes, mesh.cells, field, *PROJECTIONS["top"][:3])
    assert top.size == 3 * 2 and top.mean() == pytest.approx(field.mean())


def test_density_figure_adds_top_and_end_views_only_for_deep_meshes(tmp_path):
    for nz, expected in ((1, 1), (2, 3)):
        mesh = _brick_grid(4, 2, nz)
        np.savez(tmp_path / f"mesh{nz}.npz", nodes=mesh.nodes, cells=mesh.cells)
        np.savez(tmp_path / f"rho{nz}.npz", densities=np.linspace(0, 1, mesh.n_elements))
        svg = density_svg(tmp_path / f"mesh{nz}.npz", tmp_path / f"rho{nz}.npz")
        assert svg.count("<svg") == expected


# --------------------------------------------------------------------------
# CadQuery input control and CAD pictures in the report
# --------------------------------------------------------------------------
import html as html_lib  # noqa: E402

from topocombo.cadview import _weld, feature_edges, render_triangles, shape_triangles  # noqa: E402
from topocombo.geometry import CadDomain, run_cadquery_script  # noqa: E402


@pytest.mark.parametrize(
    "domain", [BeamDomain(length=7.0, height=3.0), BeamDomain3D(length=7.0, height=3.0, width=2.0)]
)
def test_the_exported_script_rebuilds_the_domain(domain, tmp_path):
    """The script is the CadQuery input: running it on its own gives the same shape."""
    script = domain.cadquery_script()
    assert "length = 7.0" in script and "height = 3.0" in script
    shape = run_cadquery_script(script).val()
    bb = shape.BoundingBox()
    assert (bb.xmin, bb.ymin, bb.zmin) == pytest.approx((0.0, 0.0, 0.0))
    assert (bb.xmax, bb.ymax) == pytest.approx((7.0, 3.0))
    if isinstance(domain, BeamDomain3D):
        assert bb.zmax == pytest.approx(2.0)
        assert shape.Volume() == pytest.approx(domain.volume)
    else:
        assert shape.Area() == pytest.approx(domain.area)


def test_a_script_without_a_result_is_rejected():
    with pytest.raises(ValueError, match="result"):
        run_cadquery_script("import cadquery as cq\nbox = cq.Workplane().box(1, 1, 1)\n")


def test_a_box_has_twelve_feature_edges():
    points, triangles = shape_triangles(BeamDomain3D(length=3.0, height=2.0, width=1.0).solid())
    points, triangles = _weld(points, triangles)
    edges, sides = feature_edges(points, triangles)
    assert edges.shape == (12, 2)
    assert np.all(sides >= 0)  # a closed solid: every edge has two faces


def test_render_writes_a_png(tmp_path):
    points, triangles = shape_triangles(BeamDomain(length=3.0, height=2.0).face())
    png = render_triangles(points, triangles, tmp_path / "face.png", two_sided=True)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("which", ["coarse_run", "coarse_run_3d"])
def test_report_prints_the_cadquery_input_and_shows_the_cad_output(which, request, tmp_path):
    run_info = request.getfixturevalue(which)
    site = tmp_path / "site"
    html = build_site(run_dir=run_info["dir"], site_dir=site).read_text()
    assert "Geometry (CadQuery)" in html
    script = html_lib.escape(run_info["domain"].cadquery_script())
    assert script in html
    assert "figures/cad_domain.png" in html and "figures/topology.png" in html
    assert (site / "figures" / "cad_domain.png").stat().st_size > 0
    assert (site / "figures" / "topology.png").stat().st_size > 0
    assert (site / "artifacts" / "design_domain.py").exists()



# --------------------------------------------------------------------------
# circular cutouts through z
# --------------------------------------------------------------------------
import cadquery as cq  # noqa: E402

HOLE = (20.0, 10.0, 10.0)


@pytest.mark.parametrize("cls", [BeamDomain, BeamDomain3D])
def test_cutout_is_cut_from_the_cad_model(cls):
    domain = cls(holes=[HOLE])
    shape = domain.workplane().val()
    hole_area = np.pi * 5.0**2
    if cls is BeamDomain3D:
        assert isinstance(shape, cq.Solid)
        assert shape.Volume() == pytest.approx(domain.volume - hole_area * domain.width)
        assert domain.material_volume == pytest.approx(shape.Volume())
    else:
        assert isinstance(shape, cq.Face)
        assert shape.Area() == pytest.approx(domain.area - hole_area)
        assert domain.material_area == pytest.approx(shape.Area())
    assert "holes = [(20.0, 10.0, 10.0)]" in domain.cadquery_script()
    assert domain.envelope().holes == ()


@pytest.mark.parametrize(
    "hole, message",
    [((58.0, 10.0, 10.0), "fit"), ((20.0, 10.0, 0.0), "positive"), ((20.0, 10.0), "diameter")],
)
def test_cutouts_must_fit_inside_the_domain(hole, message):
    with pytest.raises(ValueError, match=message):
        BeamDomain3D(holes=[hole])


def test_overlapping_cutouts_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        BeamDomain(holes=[(20.0, 10.0, 10.0), (26.0, 10.0, 4.0)])


def test_void_mask_picks_the_centroids_inside_the_cutout():
    domain = BeamDomain(holes=[HOLE])
    centroids = np.array([[20.0, 10.0], [24.9, 10.0], [25.1, 10.0], [50.0, 10.0]])
    assert domain.void_mask(centroids).tolist() == [True, True, False, False]


def test_passive_elements_stay_void_and_the_volume_holds():
    mesh = _brick_grid(8, 4, 1)
    mesh.node_sets[BEAM_FIXED] = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    tip = int(np.flatnonzero((mesh.nodes[:, 0] == 8.0) & (mesh.nodes[:, 1] == 2.0))[0])
    passive = np.zeros(mesh.n_elements, dtype=bool)
    passive[:4] = True  # the column of elements next to the clamp
    for filter_type in ("sensitivity", "density"):
        design = optimize(
            mesh, Material(), 1.0, LoadCase(node=tip, fy=-10.0), BEAM_FIXED,
            SimpParams(max_iterations=8, filter_type=filter_type, filter_radius=1.5),
            passive=passive,
        )
        assert np.all(design.densities[passive] == 0.0)
        assert design.volume_fraction == pytest.approx(0.5, abs=1e-6)


@pytest.fixture(scope="module")
def holed_run_3d(tmp_path_factory):
    out = tmp_path_factory.mktemp("holed3d")
    domain = BeamDomain3D(length=24.0, height=8.0, width=1.0, holes=[(8.0, 4.0, 4.0)])
    spec = MeshSpec3D(nelx=24, nely=8, nelz=1)
    _, summary = run(domain=domain, spec=spec, out_dir=out,
                     simp=SimpParams(max_iterations=6), echo=False)
    return {"dir": out, "domain": domain, "summary": summary}


def test_pipeline_holds_the_cutout_void(holed_run_3d):
    out, summary = holed_run_3d["dir"], holed_run_3d["summary"]
    assert summary["all_checks_passed"]
    assert (out / "cad" / "design_envelope.brep").exists()
    passive = np.load(out / "mesh" / "mesh.npz")["passive"]
    assert passive.sum() == summary["passive_elements"] > 0
    assert summary["passive_measure"] == pytest.approx(summary["cutout_measure"], rel=0.25)
    densities = np.load(out / "optimization" / "density.npz")["densities"]
    assert np.all(densities[passive] == 0.0)


def test_report_shows_the_cutout(holed_run_3d, tmp_path):
    html = build_site(run_dir=holed_run_3d["dir"], site_dir=tmp_path / "site").read_text()
    assert 'class="hole"' in html and "held void" in html
    assert "(8, 4, ⌀4)" in html


# --------------------------------------------------------------------------
# body-fitted meshing: the CAD profile, cutouts included
# --------------------------------------------------------------------------
from topocombo.mesh_io import check_mesh  # noqa: E402


@pytest.fixture(scope="module", params=[2, 3], ids=["2d", "3d"])
def fitted_run(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"fitted{request.param}d")
    holes = [(8.0, 4.0, 4.0)]
    if request.param == 3:
        domain = BeamDomain3D(length=24.0, height=8.0, width=1.0, holes=holes)
        spec = MeshSpec3D(nelx=24, nely=8, nelz=2, mode="body-fitted")
    else:
        domain = BeamDomain(length=24.0, height=8.0, holes=holes)
        spec = MeshSpec(nelx=24, nely=8, mode="body-fitted")
    _, summary = run(domain=domain, spec=spec, out_dir=out,
                     simp=SimpParams(max_iterations=6), echo=False)
    return {"dir": out, "domain": domain, "spec": spec, "summary": summary}


def test_mesh_spec_rejects_an_unknown_mode_and_a_bad_size():
    with pytest.raises(ValueError, match="mode"):
        MeshSpec(mode="tetra")
    with pytest.raises(ValueError, match="size"):
        MeshSpec3D(mode="body-fitted", size=0.0)
    assert MeshSpec(mode="body-fitted").n_elements is None
    assert MeshSpec(nelx=30, nely=10, mode="body-fitted").element_size(BeamDomain()) == 2.0


def test_fitted_mesh_follows_the_cad_boundary(fitted_run):
    domain, summary = fitted_run["domain"], fitted_run["summary"]
    assert summary["all_checks_passed"] and summary["mesh_mode"] == "body-fitted"
    mesh = load_mesh(fitted_run["dir"] / "mesh" / "beam.msh")
    cad = domain.material_volume if mesh.dim == 3 else domain.material_area
    # straight chords over the arc: slightly more than the CAD, never less
    assert cad < mesh.cell_measures().sum() < cad * (1 + 5e-3)
    centroids = mesh.nodes[mesh.cells].mean(axis=1)
    assert not domain.void_mask(centroids).any()  # nothing is meshed inside the hole
    assert "passive" not in np.load(fitted_run["dir"] / "mesh" / "mesh.npz")
    r = np.hypot(mesh.nodes[:, 0] - 8.0, mesh.nodes[:, 1] - 4.0)
    assert r.min() == pytest.approx(2.0, abs=1e-6)  # nodes sit on the hole's edge


def test_fitted_mesh_puts_the_load_on_real_nodes(fitted_run):
    domain, summary = fitted_run["domain"], fitted_run["summary"]
    mesh = load_mesh(fitted_run["dir"] / "mesh" / "beam.msh")
    loaded = mesh.nodes[summary["load_nodes"]]
    assert np.allclose(loaded[:, :2], domain.load_point[:2])
    if mesh.dim == 3:
        assert len(summary["load_nodes"]) == fitted_run["spec"].nelz + 1
        assert mesh.cell_type == "hexahedron"
    assert summary["solve"]["reaction_y"] == pytest.approx(1000.0, rel=1e-9)


def test_fitted_run_optimizes_and_reports(fitted_run, tmp_path):
    opt = fitted_run["summary"]["optimization"]
    assert opt["iterations"] == 6
    assert opt["volume_fraction"] == pytest.approx(0.5, abs=1e-6)
    html = build_site(run_dir=fitted_run["dir"], site_dir=tmp_path / "site").read_text()
    assert "body-fitted" in html and "real hole" in html


def test_check_mesh_rejects_a_fitted_mesh_that_misses_the_cutout(coarse_run):
    """The structured coarse mesh covers the hole, so as a body-fitted mesh of a
    holed domain its area is too large."""
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    holed = BeamDomain(length=12.0, height=4.0, holes=[(6.0, 2.0, 2.0)])
    summary = check_mesh(mesh, holed, expected_elements=None)
    assert not summary["checks"]["area_sum_matches_cad"]


def test_filter_weighs_neighbours_by_size_only_on_non_uniform_meshes():
    uniform = _brick_grid(3, 1, 1)
    h, _ = build_filter(uniform, 1.5)
    assert np.allclose(h.toarray(), h.toarray().T)
    stretched = _brick_grid(3, 1, 1)
    stretched.nodes[stretched.nodes[:, 0] == 3.0, 0] = 5.0  # last element twice as long
    h, hs = build_filter(stretched, 1.5)
    dense = h.toarray()
    assert dense[1, 2] == pytest.approx(2 * dense[2, 1] * dense[1, 1] / dense[2, 2])


def test_cli_runs_a_body_fitted_mesh(tmp_path):
    out = tmp_path / "cli_fitted"
    code = cli_main([
        "run", "--dim", "3", "--length", "12", "--height", "4", "--hole", "4,2,2",
        "--mesh", "body-fitted", "--mesh-size", "0.5", "--no-optimize", "--out", str(out),
    ])
    assert code == 0
    run_json = json.loads((out / "run.json").read_text())
    assert run_json["params"]["mesh"]["mode"] == "body-fitted"
    assert run_json["params"]["mesh"]["size"] == 0.5
    assert (out / "cad" / "design_profile.brep").exists()


# --------------------------------------------------------------------------
# a design as two Python scripts: part.py (geometry + regions) and study.py
# --------------------------------------------------------------------------
from pathlib import Path  # noqa: E402

from topocombo.regions import node_shares, region_dim  # noqa: E402
from topocombo.study import Fix, Force, Mesh as StudyMesh, Study, load_study  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cantilever"


def _part(length=12.0, height=4.0, notch_y=0.0, d=2.0, regions=None) -> str:
    regions = regions or (
        '{"wall": result.faces("<X"), "tip": cq.Edge.makeLine('
        f'cq.Vector({length}, {height / 2}, 0), cq.Vector({length}, {height / 2}, 1))}}'
    )
    return (
        "import cadquery as cq\n"
        f"result = cq.Workplane('XY').box({length}, {height}, 1.0, centered=False)\n"
        f"result = result.cut(cq.Workplane('XY').center({length / 3}, {notch_y})"
        f".circle({d / 2}).extrude(3.0, both=True))\n"
        f"regions = {regions}\n"
    )


def _study(tmp_path, part_source: str, body: str) -> Path:
    (tmp_path / "part.py").write_text(part_source)
    study = tmp_path / "study.py"
    study.write_text(
        "from topocombo.study import Fix, Force, Material, Mesh, SimpParams, Study\n"
        f"study = Study(part='part.py', {body})\n"
    )
    return study


def test_the_example_study_loads_its_part_and_regions():
    loaded = load_study(EXAMPLE / "study.py")
    domain = loaded.domain
    assert domain.dim == 3 and domain.path == str(EXAMPLE / "part.py")
    assert (domain.length, domain.height, domain.width) == pytest.approx((60.0, 20.0, 1.0))
    # the hole on the bottom edge (y = 0) bites a half-circle notch out of the beam
    assert domain.material_volume == pytest.approx(1200.0 - np.pi * 25.0 / 2)
    assert set(domain.regions()) == {"wall", "tip"}
    assert region_dim(domain.regions()["wall"]) == 2 and region_dim(domain.regions()["tip"]) == 1
    assert loaded.study.constraints == (Fix("wall"),)
    assert loaded.study.loads == (Force("tip", (0.0, -1000.0, 0.0)),)
    assert loaded.spec.mode == "body-fitted" and loaded.spec.nelz == 1


def test_a_part_notch_is_read_from_the_shape():
    domain = CadDomain(_part(60.0, 20.0, 0.0, 10.0))
    assert domain.has_cutouts
    assert domain.material_area == pytest.approx(1200.0 - np.pi * 25.0 / 2)
    assert domain.envelope().volume == pytest.approx(1200.0)
    centroids = np.array([[20.0, 0.5, 0.5], [20.0, 10.0, 0.5], [40.0, 0.5, 0.5]])
    assert domain.void_mask(centroids).tolist() == [True, False, False]


def test_a_part_face_is_a_2d_domain():
    source = (
        "import cadquery as cq\n"
        "result = cq.Workplane('XY').rect(12.0, 4.0, centered=False).extrude(1.0)"
        ".faces('<Z').val()\n"
        "regions = {'left': cq.Workplane().add(result).edges('<X')}\n"
    )
    domain = CadDomain(source, thickness=2.0)
    assert domain.dim == 2 and domain.thickness == 2.0
    assert domain.area == pytest.approx(48.0) and not domain.has_cutouts
    assert region_dim(domain.regions()["left"]) == 1


@pytest.mark.parametrize(
    "body, message",
    [
        ("result = cq.Workplane('XY').box(6, 2, 1)", "origin"),
        ("result = cq.Workplane('XY').box(2, 2, 1, centered=False)"
         ".add(cq.Workplane('XY').box(2, 2, 1, centered=False).translate((4, 0, 0)).val())",
         "one connected"),
        ("box = cq.Workplane('XY').box(6, 2, 1)", "result"),
        ("result = cq.Workplane('XY').box(6, 2, 1, centered=False)\n"
         "regions = {'none': result.faces('%SPHERE')}", "selects nothing"),
    ],
    ids=["off-origin", "two-solids", "no-result", "empty-region"],
)
def test_a_part_the_pipeline_cannot_use_is_rejected(body, message):
    with pytest.raises(ValueError, match=message):
        CadDomain("import cadquery as cq\n" + body + "\n")


@pytest.mark.parametrize(
    "body, message",
    [
        ("constraints=[], loads=[Force('tip', (0, -1, 0))]", "at least one Fix"),
        ("constraints=[Fix('wall')], loads=[]", "either loads"),
        ("constraints=[Fix('base')], loads=[Force('tip', (0, -1, 0))]", "base not defined"),
        ("constraints=[Fix('wall')], loads=[Force('tip', (0, -1))]", "2 components"),
        ("constraints=[Fix('wall')], loads=[Force('tip', (0, -1, 0))], mesh=Mesh(mode='tet')",
         "Mesh.mode"),
        ("constraints=[Fix('wall')], loads=[Force('tip', (0, -1, 0))], mesh=Mesh(element='quad4')",
         "hex8"),
    ],
    ids=["no-fix", "no-force", "unknown-region", "wrong-dimension", "bad-mode", "wrong-element"],
)
def test_a_study_that_does_not_fit_its_part_is_rejected(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        load_study(_study(tmp_path, _part(), body))


def test_loads_spread_by_tributary_length_and_area():
    mesh = _brick_grid(2, 2, 1)  # 2 x 2 x 1 unit bricks
    face = np.flatnonzero(np.isclose(mesh.nodes[:, 0], 2.0))  # the x = 2 face, 3 x 2 nodes
    shares = node_shares(mesh.nodes, mesh.cells, mesh.cell_type, face, 2, "end")
    corner, edge_mid = 1.0 / 8.0, 2.0 / 8.0  # a quarter of one / two unit facets, of 2
    y = mesh.nodes[face, 1]
    assert shares.sum() == pytest.approx(1.0)
    assert np.allclose(shares[np.isclose(y, 1.0)], edge_mid)
    assert np.allclose(shares[~np.isclose(y, 1.0)], corner)
    line = face[np.isclose(mesh.nodes[face, 2], 0.0)]  # along y on that face: 3 nodes
    shares = node_shares(mesh.nodes, mesh.cells, mesh.cell_type, line, 1, "edge")
    assert sorted(shares) == pytest.approx([0.25, 0.25, 0.5])
    with pytest.raises(ValueError, match="no mesh nodes"):
        node_shares(mesh.nodes, mesh.cells, mesh.cell_type, np.empty(0, int), 1, "gone")


@pytest.mark.parametrize("mode", ["structured", "body-fitted"])
def test_cli_runs_a_study(mode, tmp_path):
    study = _study(
        tmp_path, _part(),
        f"mesh=Mesh(mode='{mode}', nelx=24, nely=8), constraints=[Fix('wall')], "
        "loads=[Force('tip', (0.0, -1000.0, 0.0))], optimize=SimpParams(max_iterations=5)",
    )
    out = tmp_path / "run"
    code = cli_main(["all", "--study", str(study), "--out", str(out),
                     "--site", str(tmp_path / "site")])
    assert code == 0
    run_json = json.loads((out / "run.json").read_text())
    params = run_json["params"]
    assert params["dim"] == 3 and params["study"]["path"] == str(study)
    assert params["boundary_conditions"]["constraints"][0]["region"] == "wall"
    steps = {step["name"]: step.get("data", {}) for step in run_json["steps"]}
    assert steps["validation"]["all_checks_passed"]
    assert steps["validation"]["node_sets"]["tip"] == 2
    assert steps["solve"]["reactions"] == pytest.approx([0.0, 1000.0, 0.0], abs=1e-6)
    assert steps["solve"]["beam_theory"] is None  # a study is not assumed to be a beam
    assert (out / "study.py").read_text() == study.read_text()
    assert (out / "cad" / "design_domain.py").read_text() == (tmp_path / "part.py").read_text()
    if mode == "structured":  # the grid cells in the notch are held void
        passive = np.load(out / "mesh" / "mesh.npz")["passive"]
        assert passive.sum() == steps["validation"]["passive_elements"] > 0
        densities = np.load(out / "optimization" / "density.npz")["densities"]
        assert np.all(densities[passive] == 0.0)
    html = (tmp_path / "site" / "index.html").read_text()
    assert "study.py" in html and "&#x27;wall&#x27;: face region" in html


def test_a_study_face_load_matches_the_line_load_it_spreads(tmp_path):
    """The same total force over the whole free face instead of the mid-height
    line: equilibrium still holds and every face node takes a share."""
    study = _study(
        tmp_path, _part(),
        "mesh=Mesh(mode='structured', nelx=24, nely=8), constraints=[Fix('wall')], "
        "loads=[Force('tip', (0.0, -1000.0, 0.0))], optimize=None",
    )
    part = (tmp_path / "part.py").read_text().replace(
        '"tip": cq.Edge', '"tip": result.faces(">X"), "line": cq.Edge'
    )
    (tmp_path / "part.py").write_text(part)
    from topocombo.study import run_study

    _, summary = run_study(study, tmp_path / "run", echo=False)
    assert len(summary["load_nodes"]) == 9 * 2  # the whole x = L face
    assert summary["solve"]["reactions"][1] == pytest.approx(1000.0, rel=1e-9)


def test_cli_rejects_flags_a_study_sets(tmp_path):
    with pytest.raises(SystemExit):
        cli_main(["run", "--study", str(EXAMPLE / "study.py"), "--volfrac", "0.3",
                  "--out", str(tmp_path / "run")])


# --------------------------------------------------------------------------
# mesh and boundary-condition inputs in the report
# --------------------------------------------------------------------------
def test_run_records_the_boundary_conditions(coarse_run):
    params = json.loads((coarse_run["dir"] / "run.json").read_text())["params"]
    bcs = params["boundary_conditions"]
    assert bcs["constraints"][0]["region"] == BEAM_FIXED
    assert bcs["constraints"][0]["kind"] == "edge"
    assert bcs["constraints"][0]["displacements"] == {"ux": 0.0, "uy": 0.0}
    assert bcs["loads"][0]["region"] == BEAM_LOAD and bcs["loads"][0]["kind"] == "vertex"
    assert bcs["loads"][0]["force"] == {"fx": 0.0, "fy": -1000.0}
    domain, spec = coarse_run["domain"], coarse_run["spec"]
    assert params["mesh"]["element_size"] == pytest.approx(
        [domain.length / spec.nelx, domain.height / spec.nely]
    )


def test_report_shows_the_mesh_force_and_constraint_inputs(coarse_run, fitted_run, tmp_path):
    html = build_site(run_dir=coarse_run["dir"], site_dir=tmp_path / "a").read_text()
    for title in ("Mesh size", "Loads", "Displacement constraints"):
        assert f"<h3>{title}</h3>" in html
    assert "(0, -1000)" in html and "clamped" in html
    assert "<input" not in html  # a report, not a form
    fitted = build_site(run_dir=fitted_run["dir"], site_dir=tmp_path / "b").read_text()
    assert "target element edge (mm)" in fitted
    if fitted_run["summary"]["dim"] == 3:
        assert "(0, -1000, 0)" in fitted


# --------------------------------------------------------------------------
# the element registry: invariants every element must satisfy (tets included,
# once they are added)
# --------------------------------------------------------------------------
from topocombo import elements as E  # noqa: E402

ALL_ELEMENTS = list(E.ELEMENTS.values())


def _distorted(el, seed=0):
    rng = np.random.default_rng(seed)
    return el.reference_nodes * np.arange(1.0, el.dim + 1) * 0.7 + 0.1 * rng.random(
        el.reference_nodes.shape
    )


@pytest.mark.parametrize("el", ALL_ELEMENTS, ids=lambda e: e.name)
def test_element_shape_functions_and_quadrature_are_consistent(el):
    points, weights = el.quadrature
    for p in points:  # the shape functions sum to one: their derivatives to zero
        assert np.allclose(el.shape_derivatives(p).sum(axis=0), 0.0)
    ref = el.reference_nodes
    # the reference cell mapped onto itself has det J = 1: its measure is sum(weights)
    assert E.measures(el, ref, np.arange(el.n_nodes)[None])[0] == pytest.approx(weights.sum())
    # a linear map x = A xi scales every measure by det A
    a = np.diag(np.arange(2.0, el.dim + 2)) + 0.1
    assert E.measures(el, ref @ a.T, np.arange(el.n_nodes)[None])[0] == pytest.approx(
        weights.sum() * np.linalg.det(a)
    )


@pytest.mark.parametrize("el", ALL_ELEMENTS, ids=lambda e: e.name)
def test_element_facets_face_outward_and_flip_inverts(el):
    ref = el.reference_nodes
    centre = ref.mean(axis=0)
    for facet in el.facets:
        p = ref[list(facet)]
        if el.dim == 2:
            d = p[1] - p[0]
            normal = np.array([d[1], -d[0]])
        else:
            normal = np.cross(p[1] - p[0], p[2] - p[0])
        assert normal @ (p.mean(axis=0) - centre) > 0
    cells = np.arange(el.n_nodes)[None]
    flipped = cells[:, list(el.flip)]
    assert E.measures(el, ref, flipped)[0] == pytest.approx(-E.measures(el, ref, cells)[0])
    assert sorted(el.flip) == list(range(el.n_nodes))
    assert all(len(f) == len(el.facets[0]) for f in el.facets)


@pytest.mark.parametrize("el", ALL_ELEMENTS, ids=lambda e: e.name)
def test_element_stiffness_has_only_rigid_body_modes_as_nullspace(el):
    d = Material().stiffness_matrix(el.dim)
    coords = _distorted(el)
    ke = E.stiffness(el, coords[None], d)[0]
    assert np.allclose(ke, ke.T, atol=1e-9 * abs(ke).max())
    eig = np.linalg.eigvalsh(ke)
    rigid = 3 if el.dim == 2 else 6
    assert np.sum(np.abs(eig) < 1e-8 * eig.max()) == rigid
    # vectorised over elements = one element at a time
    batch = np.stack([_distorted(el, s) for s in range(3)])
    together = E.stiffness(el, batch, d, chunk=2)
    for i in range(3):
        assert np.allclose(together[i], E.stiffness(el, batch[i][None], d)[0])


def test_element_lookup_names_the_supported_types():
    assert E.element("hexahedron") is E.HEX8 and E.BY_NAME["quad4"] is E.QUAD4
    with pytest.raises(ValueError, match="supported"):
        E.element("pyramid")


def test_an_inverted_element_is_refused():
    el = E.QUAD4
    with pytest.raises(ValueError, match="non-positive Jacobian"):
        E.stiffness(el, el.reference_nodes[list(el.flip)][None], Material().stiffness_matrix(2))


@pytest.mark.parametrize("el, order", [(E.TET4, 1), (E.TET10, 2), (E.HEX8, 1), (E.QUAD4, 1)],
                         ids=["tet4", "tet10", "hex8", "quad4"])
def test_element_reproduces_complete_polynomial_fields(el, order):
    """Nodal values of a polynomial field of the element's order give exactly its
    strain at any point: this checks the shape functions and the mid-node order."""
    rng = np.random.default_rng(3)
    a = np.eye(el.dim) + 0.2 * rng.random((el.dim, el.dim))  # a distorted, straight element
    coords = el.reference_nodes @ a.T
    c1 = rng.random((el.dim, el.dim))  # u_i = c1_ij x_j + c2_ijk x_j x_k
    c2 = rng.random((el.dim, el.dim, el.dim)) * (order == 2)
    u = coords @ c1.T + np.einsum("ijk,nj,nk->ni", c2, coords, coords)
    for point in el.quadrature[0]:
        b, _ = E.strain_displacement(el, coords[None], point)
        # the physical point: coords = reference @ A^T, so the map is x = A xi
        x = point @ a.T
        grad = c1 + np.einsum("ijk,k->ij", c2, x) + np.einsum("ijk,j->ik", c2, x)  # du_i/dx_j
        strain = b[0] @ u.reshape(-1)
        if el.dim == 3:
            expect = [grad[0, 0], grad[1, 1], grad[2, 2], grad[0, 1] + grad[1, 0],
                      grad[1, 2] + grad[2, 1], grad[2, 0] + grad[0, 2]]
        else:
            expect = [grad[0, 0], grad[1, 1], grad[0, 1] + grad[1, 0]]
        assert strain == pytest.approx(expect, abs=1e-10)


def test_tet10_face_and_edge_loads_are_consistent():
    """A uniform load on one 6-node face goes wholly to its mid-nodes, a third
    each; along a quadratic edge it splits 1/6, 1/6, 2/3."""
    el = E.TET10
    nodes, cells = el.reference_nodes, np.arange(10)[None]
    face = np.array(el.facets[0])  # the z = 0 face
    shares = node_shares(nodes, cells, "tetra10", face, 2, "base")
    assert shares == pytest.approx([0, 0, 0, 1 / 3, 1 / 3, 1 / 3])
    edge = np.array([0, 1, 4])  # corners 0, 1 and their mid-node
    shares = node_shares(nodes, cells, "tetra10", edge, 1, "edge")
    assert shares == pytest.approx([1 / 6, 1 / 6, 2 / 3])


# --------------------------------------------------------------------------
# tetrahedra: meshing any 3D part, T10 accuracy, solvers
# --------------------------------------------------------------------------
from topocombo.fea import SOLVERS, rigid_body_modes  # noqa: E402
from topocombo.meshing import generate_tet_mesh  # noqa: E402

BRACKET = Path(__file__).resolve().parents[1] / "examples" / "bracket"


def _tet_beam_run(tmp, element, size=1.0, **kw):
    domain = BeamDomain3D(length=24.0, height=8.0, width=2.0)
    spec = MeshSpec3D(mode="body-fitted", size=size, element=E.BY_NAME[element])
    return run(domain=domain, spec=spec, out_dir=tmp, optimize_design=False, echo=False, **kw)


@pytest.fixture(scope="module")
def tet_beams(tmp_path_factory):
    out = {}
    for element in ("tet10", "tet4"):
        _, out[element] = _tet_beam_run(tmp_path_factory.mktemp(element), element)
    hex_domain = BeamDomain3D(length=24.0, height=8.0, width=2.0)
    _, out["hex8"] = run(domain=hex_domain, spec=MeshSpec3D(nelx=96, nely=32, nelz=4),
                         out_dir=tmp_path_factory.mktemp("hexref"), optimize_design=False,
                         echo=False)
    return out


def test_tet_meshes_pass_validation_and_balance_the_load(tet_beams):
    for element in ("tet10", "tet4"):
        summary = tet_beams[element]
        assert summary["all_checks_passed"], summary["checks"]
        assert summary["tet_quality_min"] > 0.1
        assert summary["solve"]["reactions"] == pytest.approx([0.0, 1000.0, 0.0], abs=1e-6)


def test_tet10_matches_a_fine_hex_mesh_and_tet4_is_stiffer(tet_beams):
    """Quadratic tets at 1 mm agree with a 4x-refined hex mesh to ~1%; linear
    tets of the same size are too stiff in bending (shear locking; ~5% here)."""
    c10 = tet_beams["tet10"]["solve"]["compliance"]
    c8 = tet_beams["hex8"]["solve"]["compliance"]
    c4 = tet_beams["tet4"]["solve"]["compliance"]
    assert c10 == pytest.approx(c8, rel=0.015)
    assert c4 < 0.97 * c10


def test_tet_mesher_imprints_a_load_line_across_a_face(tet_beams):
    """The beam's load region is a line across the free end, not a CAD edge;
    fused into the solid it carries corner and mid-edge nodes (width 2 at 1 mm:
    3 corners, 2 mid-nodes), shared 1/6-2/3-1/6 per segment."""
    summary = tet_beams["tet10"]
    assert len(summary["load_nodes"]) == 5
    shares = sorted(summary["solve"]["load"]["shares"])
    assert shares == pytest.approx([1 / 12, 1 / 12, 1 / 6, 1 / 3, 1 / 3])


def test_hex8_refuses_a_part_that_is_not_a_prism_and_tet10_meshes_it(tmp_path):
    loaded = load_study(BRACKET / "study.py")
    assert loaded.domain.dim == 3 and not loaded.domain.is_prism
    with pytest.raises(ValueError, match="tet10"):
        run(domain=loaded.domain, spec=StudyMesh(mode="body-fitted").spec(3),
            out_dir=tmp_path / "hex", constraints=loaded.study.constraints,
            load_cases=loaded.study.cases[:1], echo=False)
    _, summary = run(
        domain=loaded.domain, spec=StudyMesh(element="tet10", size=4.0).spec(3),
        out_dir=tmp_path / "tet", constraints=loaded.study.constraints,
        load_cases=loaded.study.cases[:1], simp=SimpParams(volume_fraction=0.3, max_iterations=4,
                                                  filter_radius=6.0),
        echo=False,
    )
    assert summary["all_checks_passed"]
    assert summary["solve"]["reactions"][1] == pytest.approx(1000.0, rel=1e-9)
    # the load hangs on the pin's bore: every loaded node is on the hole's radius
    nodes = np.load(tmp_path / "tet" / "mesh" / "mesh.npz")["nodes"]
    r = np.hypot(nodes[summary["load_nodes"], 0] - 52.0, nodes[summary["load_nodes"], 1] - 15.0)
    assert np.allclose(r, 3.0, atol=1e-6)
    assert summary["optimization"]["volume_fraction"] == pytest.approx(0.3, abs=1e-6)
    # the STL is closed: every triangle edge is shared by an even number of
    # triangles — two, or four where two solid elements touch along an edge only
    # (its volume differs slightly from the elements': mid-nodes on the curved
    # bore make those elements curved, the STL flattens them into sub-triangles)
    tri = meshio.read(str(tmp_path / "tet" / "optimization" / "topology.stl"))
    points, triangles = _weld(tri.points, tri.cells_dict["triangle"])
    edges = np.sort(triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts % 2 == 0) and np.mean(counts == 2) > 0.9


def test_mesh_specs_reject_tets_on_a_grid_or_in_2d():
    with pytest.raises(ValueError, match="body-fitted"):
        MeshSpec3D(element=E.TET10)  # structured by default
    with pytest.raises(ValueError, match="3D elements"):
        MeshSpec3D(mode="body-fitted", element=E.QUAD4)
    with pytest.raises(SystemExit):
        cli_main(["run", "--dim", "2", "--element", "tet10", "--out", "unused"])


def test_cg_and_direct_solves_agree(tmp_path):
    _, direct = _tet_beam_run(tmp_path / "d", "tet10", size=2.0, solver="direct")
    _, cg = _tet_beam_run(tmp_path / "c", "tet10", size=2.0, solver="cg")
    assert direct["solve"]["solver"] == "direct" and cg["solve"]["solver"] == "amg-cg"
    assert cg["solve"]["solver_iterations"] > 0
    assert cg["solve"]["compliance"] == pytest.approx(direct["solve"]["compliance"], rel=1e-8)
    assert "auto" in SOLVERS


def test_rigid_body_modes_are_the_stiffness_nullspace():
    for el in ALL_ELEMENTS:
        mesh = Mesh(_distorted(el), np.arange(el.n_nodes)[None], {}, el.cell_type)
        k = element_stiffnesses(mesh, Material(), 1.0)[0]
        assert np.abs(k @ rigid_body_modes(mesh)).max() < 1e-9 * np.abs(k).max()


def test_cached_assembly_equals_a_fresh_coo_assembly(coarse_run):
    import scipy.sparse as sp

    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    ke = element_stiffnesses(mesh, Material(), 1.0)
    scale = np.linspace(0.1, 1.0, mesh.n_elements)
    k1 = assemble_stiffness(mesh, ke, scale)
    k2 = assemble_stiffness(mesh, ke, scale)  # from the cached plan
    dofs = element_dofs(mesh.cells, 2)
    n = dofs.shape[1]
    ref = sp.coo_matrix(
        ((ke * scale[:, None, None]).reshape(-1),
         (np.repeat(dofs, n, axis=1).reshape(-1), np.tile(dofs, (1, n)).reshape(-1))),
        shape=k1.shape,
    ).tocsc()
    assert abs(k1 - ref).max() < 1e-12 * abs(ref).max()
    assert abs(k2 - ref).max() < 1e-12 * abs(ref).max()


def test_report_draws_a_tet_run_by_its_front_surface(tmp_path):
    _tet_beam_run(tmp_path / "run", "tet10", size=2.0)
    html = build_site(run_dir=tmp_path / "run", site_dir=tmp_path / "site").read_text()
    assert "front facets drawn" in html and "T10 tetrahedron" in html
    assert "body-fitted quadratic tetrahedra" in html


def test_reports_link_to_each_other(coarse_run, tmp_path):
    html = build_site(run_dir=coarse_run["dir"], site_dir=tmp_path / "site",
                      nav=[("bracket (tet10)", "bracket/")]).read_text()
    assert "<a class='chip' href='bracket/'>bracket (tet10)</a>" in html


# --------------------------------------------------------------------------
# general boundary conditions: components, prescribed displacements, load
# cases, passive regions
# --------------------------------------------------------------------------
from topocombo.fea import rigid_body_free, solve_cases  # noqa: E402
from topocombo.study import Displace, LoadCase as Case, Passive, run_loaded  # noqa: E402

MBB = Path(__file__).resolve().parents[1] / "examples" / "mbb"


def _plate(length, height, regions: str) -> str:
    return (
        "import cadquery as cq\n"
        f"result = cq.Workplane('XY').rect({length}, {height}, centered=False).extrude(1.0)"
        ".faces('<Z').val()\n"
        "edges = lambda sel: cq.Workplane('XY').add(result).edges(sel)\n"
        f"regions = {regions}\n"
    )


def _run_part(tmp_path, source, spec, **kw):
    domain = CadDomain(source)
    kw.setdefault("optimize_design", False)
    return run(domain=domain, spec=spec, out_dir=tmp_path, echo=False, **kw)


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: Fix("a", dofs="xw"), "dofs"),
        (lambda: Fix("a", dofs=""), "dofs"),
        (lambda: Displace("a"), "at least one"),
        (lambda: Case("c", []), "Force"),
        (lambda: Case("c", [Force("a", (0, 1))], weight=0), "weight"),
        (lambda: Passive("a", state="grey"), "solid"),
        (lambda: Passive("a", within=-1), "within"),
    ],
)
def test_boundary_condition_blocks_check_their_fields(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_fix_and_displace_pick_components():
    assert Fix("a").components(2) == {0: 0.0, 1: 0.0}
    assert Fix("a", dofs="y").components(3) == {1: 0.0}
    assert Displace("a", uy=-0.1).components(2) == {1: -0.1}
    with pytest.raises(ValueError, match="no z"):
        Fix("a", dofs="z").components(2)


def test_rigid_body_check_counts_the_free_motions(coarse_run):
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    left = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    assert rigid_body_free(mesh, np.sort(np.concatenate([2 * left, 2 * left + 1]))) == 0
    assert rigid_body_free(mesh, 2 * left) == 1  # ux on a line: it can still slide in y
    corner = left[np.argmin(mesh.nodes[left, 1])]
    assert rigid_body_free(mesh, np.array([2 * corner, 2 * corner + 1])) == 1  # can spin


def test_half_mbb_by_symmetry_is_half_the_full_beam(tmp_path):
    """Symmetry (ux only) and a roller (uy only) on the half beam give exactly
    half the full beam's compliance, pinned and rolling at its corners."""
    half = _plate(24.0, 8.0, "{'sym': edges('<X'), 'roller': cq.Vertex.makeVertex(24, 0, 0), "
                  "'load': cq.Vertex.makeVertex(0, 8, 0)}")
    full = _plate(48.0, 8.0, "{'pin': cq.Vertex.makeVertex(0, 0, 0), "
                  "'roller': cq.Vertex.makeVertex(48, 0, 0), 'load': cq.Vertex.makeVertex(24, 8, 0)}")
    _, h = _run_part(tmp_path / "h", half, MeshSpec(nelx=24, nely=8),
                     constraints=[Fix("sym", dofs="x"), Fix("roller", dofs="y")],
                     loads=[Force("load", (0.0, -500.0))])
    _, f = _run_part(tmp_path / "f", full, MeshSpec(nelx=48, nely=8),
                     constraints=[Fix("pin"), Fix("roller", dofs="y")],
                     loads=[Force("load", (0.0, -1000.0))])
    assert f["solve"]["compliance"] == pytest.approx(2 * h["solve"]["compliance"], rel=1e-9)
    assert h["solve"]["reactions"] == pytest.approx([0.0, 500.0], abs=1e-6)


def test_the_mbb_example_reaches_the_textbook_compliance(tmp_path):
    """Sigmund's 99-line MBB (60 x 20, volfrac 0.5, p = 3, r = 1.5) ends near
    c = 203 for E = 1 and a unit load; ours, rescaled by F^2 / E, lands there."""
    _, summary = run_loaded(load_study(MBB / "study.py"), tmp_path / "mbb", echo=False)
    opt = summary["optimization"]
    assert opt["converged"]
    normalised = opt["compliance"] * 210_000.0 / 1000.0**2
    assert normalised == pytest.approx(203.0, rel=0.01)


def test_constraints_that_leave_a_motion_free_are_refused(tmp_path):
    half = _plate(24.0, 8.0, "{'sym': edges('<X'), 'load': cq.Vertex.makeVertex(0, 8, 0)}")
    with pytest.raises(ValueError, match="free to move"):
        _run_part(tmp_path, half, MeshSpec(nelx=12, nely=4),
                  constraints=[Fix("sym", dofs="x")], loads=[Force("load", (0.0, -1.0))])


def test_conflicting_constraints_are_refused(tmp_path):
    part = _plate(12.0, 4.0, "{'left': edges('<X'), 'tip': cq.Vertex.makeVertex(12, 2, 0)}")
    with pytest.raises(ValueError, match="hold node"):
        _run_part(tmp_path, part, MeshSpec(nelx=12, nely=4),
                  constraints=[Fix("left"), Displace("left", ux=0.1)],
                  loads=[Force("tip", (0.0, -1.0))])


def test_a_prescribed_stretch_gives_the_bar_reaction(tmp_path):
    """A bar pulled by a prescribed end displacement, free to contract: the
    reaction is E A delta / L exactly, and compliance f.u - u_p.r_p is minus
    the work the support does, -delta * reaction."""
    part = _plate(10.0, 2.0, "{'left': edges('<X'), 'corner': cq.Vertex.makeVertex(0, 0, 0), "
                  "'right': edges('>X')}")
    _, s = _run_part(tmp_path, part, MeshSpec(nelx=10, nely=2),
                     constraints=[Fix("left", dofs="x"), Fix("corner", dofs="y"),
                                  Displace("right", ux=0.01)],
                     loads=[Force("corner", (0.0, 0.0))])
    reaction = 210_000.0 * 2.0 * 0.01 / 10.0  # E * (H * t) * delta / L
    assert s["solve"]["reactions"][0] == pytest.approx(0.0, abs=1e-6)  # the two ends balance
    assert s["solve"]["compliance"] == pytest.approx(-0.01 * reaction, rel=1e-9)


def test_compliance_sensitivity_holds_with_prescribed_displacements():
    """d/dx of f.u - u_p.r_p is -p x^(p-1) u_e^T k0 u_e, as for loads alone:
    checked against central differences with a force and a pull together."""
    mesh = _brick_grid(4, 2, 1)
    left = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    right = np.flatnonzero(mesh.nodes[:, 0] == 4.0)
    mesh.node_sets["left"] = left
    pull = {3 * int(n): 0.01 for n in right}
    tip = int(right[np.argmax(mesh.nodes[right, 1])])
    load = LoadCase(node=tip, fy=-50.0)
    rho = np.random.default_rng(1).uniform(0.3, 1.0, mesh.n_elements)

    def compliance(r):
        return solve(mesh, Material(), 1.0, load, "left", densities=r, prescribed=pull).compliance

    res = solve(mesh, Material(), 1.0, load, "left", densities=rho, prescribed=pull)
    dc = -3.0 * rho**2 * (1 - 1e-9) * res.element_compliance_unscaled
    eye = np.eye(mesh.n_elements) * 1e-6
    fd = np.array([(compliance(rho + e) - compliance(rho - e)) / 2e-6 for e in eye])
    assert np.abs(fd - dc).max() < 1e-6 * np.abs(dc).max()


def test_load_cases_share_one_factorisation_and_weights_add_up(tmp_path):
    mesh = _brick_grid(4, 2, 1)
    mesh.node_sets["left"] = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    tip = int(np.flatnonzero((mesh.nodes[:, 0] == 4.0))[0])
    down, side = LoadCase(node=tip, fy=-10.0), LoadCase(node=tip, fz=5.0)
    both = solve_cases(mesh, Material(), 1.0, [down, side, [down, side]], "left")
    # linear: the case with both forces is the sum of the two
    assert np.allclose(both[2].u, both[0].u + both[1].u, rtol=1e-9, atol=1e-15)
    # two identical cases at half weight each design exactly like one case
    params = SimpParams(max_iterations=5, filter_radius=1.5)
    one = optimize(mesh, Material(), 1.0, down, "left", params)
    two = optimize(mesh, Material(), 1.0, None, "left", params,
                   cases=[(0.5, down), (0.5, down)])
    assert np.allclose(one.densities, two.densities)
    assert two.case_compliances == pytest.approx([one.compliance] * 2)


def test_passive_solid_elements_stay_solid_and_the_volume_holds():
    mesh = _brick_grid(8, 4, 1)
    mesh.node_sets["left"] = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    tip = int(np.flatnonzero((mesh.nodes[:, 0] == 8.0) & (mesh.nodes[:, 1] == 2.0))[0])
    solid = np.zeros(mesh.n_elements, dtype=bool)
    solid[-4:] = True  # the column at the tip
    design = optimize(mesh, Material(), 1.0, LoadCase(node=tip, fy=-10.0), "left",
                      SimpParams(max_iterations=8, filter_radius=1.5), solid=solid)
    assert np.all(design.densities[solid] == 1.0)
    assert design.volume_fraction == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(ValueError, match="nothing is left"):
        optimize(mesh, Material(), 1.0, LoadCase(node=tip, fy=-10.0), "left",
                 SimpParams(volume_fraction=0.1), solid=np.ones(mesh.n_elements, bool))


def test_the_bracket_study_has_two_cases_and_two_passive_rings(tmp_path):
    loaded = load_study(BRACKET / "study.py")
    assert [c.name for c in loaded.study.cases] == ["hang", "sway"]
    assert {p.region for p in loaded.study.passive} == {"pin", "wall"}
    small = dataclasses.replace(
        loaded, study=dataclasses.replace(
            loaded.study, mesh=StudyMesh(element="tet10", size=4.0),
            optimize=SimpParams(volume_fraction=0.3, max_iterations=3, filter_radius=6.0),
        ),
    )
    _, summary = run_loaded(small, tmp_path / "bracket", echo=False)
    cases = summary["solve"]["cases"]
    assert [c["name"] for c in cases] == ["hang", "sway"]
    assert cases[0]["reactions"] == pytest.approx([0, 1000, 0], abs=1e-6)
    assert cases[1]["reactions"] == pytest.approx([0, 0, 300], abs=1e-6)
    assert summary["passive_solid_elements"] > 0
    densities = np.load(tmp_path / "bracket" / "optimization" / "density.npz")["densities"]
    solid = np.load(tmp_path / "bracket" / "mesh" / "mesh.npz")["solid"]
    assert np.all(densities[solid] == 1.0)
    html = build_site(run_dir=tmp_path / "bracket", site_dir=tmp_path / "site").read_text()
    assert "Passive regions" in html and "case &#x27;sway&#x27;" in html


def test_report_draws_rollers_for_partial_supports(tmp_path):
    _, _ = run_loaded(dataclasses.replace(
        load_study(MBB / "study.py"),
    ), tmp_path / "mbb", echo=False, optimize=False)
    html = build_site(run_dir=tmp_path / "mbb", site_dir=tmp_path / "site").read_text()
    assert 'class="roll"' in html and "rollers / symmetry: 21 ux, 1 uy" in html
    assert "Symmetry holds ux = 0" in html


# --------------------------------------------------------------------------
# MMA: exact gradients, the MMA step, objectives and limits
# --------------------------------------------------------------------------
from topocombo import responses as R  # noqa: E402
from topocombo.mma import Limit, MMAState, mma_step, optimize_mma  # noqa: E402
from topocombo.study import ComplianceLimit, DisplacementLimit, StressLimit  # noqa: E402

LBRACKET = Path(__file__).resolve().parents[1] / "examples" / "lbracket"


def _responses_at(mesh, load, fixed, x, presc, ke):
    res, (system,) = solve_cases(mesh, Material(), 1.0, [load], fixed, densities=x, ke_all=ke,
                              prescribed=presc, return_system=True)
    top = np.flatnonzero(mesh.nodes[:, 0] == mesh.nodes[:, 0].max())
    sel = R.displacement_selector(mesh, top, 1)
    c, gc, _ = R.compliance(res, [1.0], x, 3.0)
    d, gd = R.displacement(mesh, system, res[0], ke, sel, x, 3.0)
    s, gs, _ = R.stress_pnorm(mesh, Material(), system, res[0], ke, x, 3.0)
    return np.array([c, d, s]), np.array([gc, gd, gs])


@pytest.mark.parametrize("three_d", [False, True], ids=["quad4", "hex8"])
def test_response_gradients_match_central_differences(three_d):
    """Compliance, a displacement and the stress p-norm, with a force and a
    prescribed pull together: every adjoint gradient against finite differences."""
    if three_d:
        mesh = _brick_grid(4, 2, 1)
    else:
        xs, ys = np.meshgrid(np.arange(5.0), np.arange(3.0), indexing="ij")
        nodes = np.column_stack([xs.ravel(), ys.ravel()])
        cells = np.array([[i * 3 + j, (i + 1) * 3 + j, (i + 1) * 3 + j + 1, i * 3 + j + 1]
                          for i in range(4) for j in range(2)])
        mesh = Mesh(nodes, cells, {}, "quad")
    d = mesh.dofs_per_node
    mesh.node_sets["left"] = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    right = np.flatnonzero(mesh.nodes[:, 0] == 4.0)
    tip = int(right[np.argmax(mesh.nodes[right, 1])])
    load = LoadCase(node=tip, fy=-50.0)
    presc = {d * int(right[0]): 0.002}
    ke = element_stiffnesses(mesh, Material(), 1.0)
    x = np.random.default_rng(4).uniform(0.2, 1.0, mesh.n_elements)
    _, grads = _responses_at(mesh, load, "left", x, presc, ke)
    fd = np.zeros_like(grads)
    for e in range(mesh.n_elements):
        step = np.zeros(mesh.n_elements)
        step[e] = 1e-6
        plus, _ = _responses_at(mesh, load, "left", x + step, presc, ke)
        minus, _ = _responses_at(mesh, load, "left", x - step, presc, ke)
        fd[:, e] = (plus - minus) / 2e-6
    for g, f in zip(grads, fd):
        assert np.abs(g - f).max() < 1e-6 * np.abs(f).max()


def test_mma_step_solves_a_small_constrained_problem():
    """min x1 + x2 subject to 1/x1 + 1/x2 <= 4: the answer is (0.5, 0.5)."""
    state, x = MMAState(2), np.array([1.0, 1.0])
    lo, hi = np.full(2, 0.1), np.ones(2)
    for _ in range(30):
        g = np.array([(1 / x).sum() / 4 - 1])
        x = mma_step(state, x, x.sum(), np.ones(2), g, np.array([-1 / x**2 / 4]), lo, hi, 0.2)
    assert x == pytest.approx([0.5, 0.5], abs=1e-4)


@pytest.fixture(scope="module")
def small_mbb():
    """A 24 x 8 half MBB beam as a mesh, load and constraint map."""
    mesh = _brick_grid(24, 8, 1)
    left = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    corner = int(np.flatnonzero((mesh.nodes[:, 0] == 24.0) & (mesh.nodes[:, 1] == 0.0))[0])
    presc = {3 * int(n): 0.0 for n in left}
    presc |= {3 * corner + 1: 0.0}
    presc |= {3 * int(n) + 2: 0.0 for n in range(mesh.n_nodes)}  # plane: uz = 0 everywhere
    top = int(np.flatnonzero((mesh.nodes[:, 0] == 0.0) & (mesh.nodes[:, 1] == 8.0)
                             & (mesh.nodes[:, 2] == 0.0))[0])
    return {"mesh": mesh, "presc": presc, "load": LoadCase(node=top, fy=-100.0)}


def test_mma_matches_oc_on_minimum_compliance(small_mbb):
    """Same problem, same density filter: MMA lands no worse than OC."""
    m, presc, load = small_mbb["mesh"], small_mbb["presc"], small_mbb["load"]
    params = SimpParams(filter_radius=1.5, filter_type="density", max_iterations=200)
    oc = optimize(m, Material(), 1.0, None, None, params, cases=[(1.0, load)], prescribed=presc)
    mma = optimize_mma(m, Material(), 1.0, [(1.0, load)], params, prescribed=presc)
    assert mma.converged and mma.optimizer == "mma"
    assert mma.volume_fraction == pytest.approx(0.5, abs=1e-3)
    assert mma.compliance < oc.compliance * 1.01


def test_minimum_volume_under_a_compliance_limit(small_mbb):
    m, presc, load = small_mbb["mesh"], small_mbb["presc"], small_mbb["load"]
    full = solve(m, Material(), 1.0, load, None, prescribed=presc).compliance
    params = SimpParams(filter_radius=1.5, max_iterations=200)
    r = optimize_mma(m, Material(), 1.0, [(1.0, load)], params, objective="volume",
                     limits=[Limit("compliance", 3.0 * full, "compliance")], prescribed=presc)
    assert r.converged
    assert r.compliance <= 3.0 * full * 1.001  # the limit holds ...
    assert r.compliance >= 3.0 * full * 0.99  # ... and is active: no volume left on the table
    assert r.volume_fraction < 0.6


def test_a_displacement_limit_is_met_and_active(small_mbb):
    m, presc, load = small_mbb["mesh"], small_mbb["presc"], small_mbb["load"]
    params = SimpParams(filter_radius=1.5, max_iterations=200, volume_fraction=0.5)
    free = optimize_mma(m, Material(), 1.0, [(1.0, load)], params, prescribed=presc)
    # the mid-height of the free end, pulled down by the design
    probe = np.flatnonzero((m.nodes[:, 0] == 24.0) & (m.nodes[:, 1] == 8.0))
    sel = R.displacement_selector(m, probe, 1)
    free_u = abs(sel @ solve(m, Material(), 1.0, load, None, densities=free.densities,
                             prescribed=presc).u)
    bound = 0.9 * free_u
    r = optimize_mma(m, Material(), 1.0, [(1.0, load)], params, prescribed=presc,
                     limits=[Limit("displacement", bound, "tip", case=0, selector=sel)])
    u = abs(sel @ solve(m, Material(), 1.0, load, None, densities=r.densities, prescribed=presc).u)
    assert u <= bound * 1.001
    assert r.limits[1]["satisfied"] and r.compliance >= free.compliance * 0.999


def test_the_stress_limited_lbracket_lowers_the_peak_stress(tmp_path):
    """A coarse L-bracket, lightest design within a stress limit, against the
    minimum-compliance design of the same volume: the limit holds and the
    peak stress is lower."""
    loaded = load_study(LBRACKET / "study.py")
    coarse = dataclasses.replace(loaded, study=dataclasses.replace(
        loaded.study, mesh=StudyMesh(mode="body-fitted", size=2.5),
        optimize=SimpParams(optimizer="mma", filter_radius=5.0, max_iterations=200),
    ))
    _, s = run_loaded(coarse, tmp_path / "stress", echo=False)
    opt = s["optimization"]
    stress = [lim for lim in opt["limits"] if lim["kind"] == "stress"][0]
    assert stress["satisfied"] and opt["volume_fraction"] < 0.6
    same = dataclasses.replace(coarse, study=dataclasses.replace(
        coarse.study, objective="compliance", limits=(),
        optimize=SimpParams(optimizer="mma", filter_radius=5.0, max_iterations=200,
                            volume_fraction=opt["volume_fraction"]),
    ))
    _, c = run_loaded(same, tmp_path / "compliance", echo=False)
    mesh = load_mesh(tmp_path / "stress" / "mesh" / "beam.msh")
    data = np.load(tmp_path / "stress" / "mesh" / "mesh.npz")
    presc = {2 * int(n) + k: 0.0 for n in data["fixed_nodes"] for k in range(2)}
    load = LoadCase(node=tuple(int(n) for n in data["load_nodes"]), fy=-1500.0)
    ke = element_stiffnesses(mesh, Material(), 5.0)

    def pnorm(run_dir):
        x = np.load(run_dir / "optimization" / "density.npz")["densities"]
        res, (system,) = solve_cases(mesh, Material(), 5.0, [load], None, densities=x,
                                  ke_all=ke, prescribed=presc, return_system=True)
        return R.stress_pnorm(mesh, Material(), system, res[0], ke, x, 3.0)[0]

    assert pnorm(tmp_path / "stress") < pnorm(tmp_path / "compliance")
    html = build_site(run_dir=tmp_path / "stress", site_dir=tmp_path / "site").read_text()
    assert "Method of Moving" in html and "stress p-norm (load)" in html


def test_the_nlopt_driver_runs_the_same_problem(small_mbb):
    m, presc, load = small_mbb["mesh"], small_mbb["presc"], small_mbb["load"]
    params = SimpParams(filter_radius=1.5, max_iterations=150)
    r = optimize_mma(m, Material(), 1.0, [(1.0, load)], params, prescribed=presc,
                     driver="nlopt")
    assert r.optimizer == "nlopt" and r.volume_fraction == pytest.approx(0.5, abs=1e-3)


@pytest.mark.parametrize(
    "extra, message",
    [
        ("objective='volume'", "needs limits"),
        ("limits=[StressLimit(100.0, case='gust')]", "unknown load case"),
        ("limits=[StressLimit(100.0)], optimize=SimpParams(optimizer='oc')", "OC takes"),
    ],
)
def test_a_study_with_impossible_limits_is_refused(tmp_path, extra, message):
    body = ("constraints=[Fix('wall')], loads=[Force('tip', (0.0, -1.0, 0.0))], " + extra)
    (tmp_path / "part.py").write_text(_part())
    study = tmp_path / "study.py"
    study.write_text(
        "from topocombo.study import *\n"
        f"study = Study(part='part.py', {body})\n"
    )
    with pytest.raises(ValueError, match=message):
        load_study(study)


def test_limit_blocks_check_their_fields():
    with pytest.raises(ValueError):
        StressLimit(0.0)
    with pytest.raises(ValueError):
        DisplacementLimit("a", "w", 1.0)
    with pytest.raises(ValueError):
        ComplianceLimit(-1.0)
    with pytest.raises(ValueError):
        Limit("stress", 1.0, "s")  # needs a case


# --------------------------------------------------------------------------
# load cases with their own constraints
# --------------------------------------------------------------------------
def test_each_case_can_have_its_own_supports():
    """Per-case prescribed maps solve like separate runs; cases held on the
    same DOFs share one system; the weighted compliance gradient still
    matches central differences."""
    mesh = _brick_grid(4, 2, 1)
    left = np.flatnonzero(mesh.nodes[:, 0] == 0.0)
    right = np.flatnonzero(mesh.nodes[:, 0] == 4.0)
    clamp = {int(d): 0.0 for d in node_dofs(left, 3)}
    tip = int(right[np.argmax(mesh.nodes[right, 1])])
    load = LoadCase(node=tip, fy=-50.0)
    propped = {**clamp, **{3 * int(n) + 1: 0.0 for n in right}}
    settled = {**clamp, **{3 * int(n) + 1: -0.01 for n in right}}
    rho = np.random.default_rng(2).uniform(0.3, 1.0, mesh.n_elements)
    maps = [clamp, propped, settled, clamp]
    res, systems = solve_cases(mesh, Material(), 1.0, [load] * 4, None, densities=rho,
                               prescribed=maps, return_system=True)
    for r, m in zip(res, maps):
        alone = solve(mesh, Material(), 1.0, load, None, densities=rho, prescribed=m)
        assert np.allclose(r.u, alone.u, rtol=1e-10, atol=1e-14)
        assert r.compliance == pytest.approx(alone.compliance, rel=1e-10)
    assert systems[0] is systems[3] and systems[1] is systems[2]
    assert systems[0] is not systems[1]
    # the prop takes load: the propped tip moves least
    assert abs(res[1].u[3 * tip + 1]) < 1e-12 < abs(res[0].u[3 * tip + 1])

    def weighted(r):
        out = solve_cases(mesh, Material(), 1.0, [load] * 3, None, densities=r,
                          prescribed=maps[:3])
        return R.compliance(out, [1.0, 0.5, 2.0], r, 3.0)

    _, grad, _ = weighted(rho)
    eye = np.eye(mesh.n_elements) * 1e-6
    fd = np.array([(weighted(rho + e)[0] - weighted(rho - e)[0]) / 2e-6 for e in eye])
    assert np.abs(fd - grad).max() < 1e-6 * np.abs(grad).max()
    with pytest.raises(ValueError, match="3 maps for 4"):
        solve_cases(mesh, Material(), 1.0, [load] * 4, None, prescribed=maps[:3])


def _span() -> str:
    return _plate(60.0, 10.0, "{'left': cq.Vertex.makeVertex(0, 0, 0), "
                  "'mid': cq.Vertex.makeVertex(30, 0, 0), "
                  "'right': cq.Vertex.makeVertex(60, 0, 0), 'deck': edges('>Y')}")


def test_a_case_settles_a_support_only_in_that_case(tmp_path):
    """A two-span beam: in 'deck' the middle support holds, in 'settled' it
    drops.  Each case's result equals a run with those supports alone."""
    spec = MeshSpec(nelx=30, nely=5)
    deck = Force("deck", (0.0, -600.0))
    shared = [Fix("left"), Fix("right", dofs="y")]
    _, both = _run_part(
        tmp_path / "both", _span(), spec, constraints=shared,
        load_cases=[Case("deck", [deck], constraints=[Fix("mid", dofs="y")]),
                    Case("settled", [deck], weight=0.5,
                         constraints=[Displace("mid", uy=-0.02)])],
    )
    _, held = _run_part(tmp_path / "held", _span(), spec,
                        constraints=shared + [Fix("mid", dofs="y")], loads=[deck])
    _, sunk = _run_part(tmp_path / "sunk", _span(), spec,
                        constraints=shared + [Displace("mid", uy=-0.02)], loads=[deck])
    cases = {c["name"]: c for c in both["solve"]["cases"]}
    assert cases["deck"]["compliance"] == pytest.approx(held["solve"]["compliance"], rel=1e-10)
    assert cases["settled"]["compliance"] == pytest.approx(sunk["solve"]["compliance"], rel=1e-10)
    assert both["solve"]["compliance"] == pytest.approx(
        held["solve"]["compliance"] + 0.5 * sunk["solve"]["compliance"], rel=1e-10)
    # the case's value wins over the study's on the same component
    _, over = _run_part(tmp_path / "over", _span(), spec,
                        constraints=shared + [Fix("mid", dofs="y")],
                        load_cases=[Case("settled", [deck],
                                         constraints=[Displace("mid", uy=-0.02)])])
    assert over["solve"]["compliance"] == pytest.approx(sunk["solve"]["compliance"], rel=1e-10)


def test_cases_may_bring_all_the_supports_and_a_case_may_have_no_forces(tmp_path):
    """No study-wide constraints: each case holds the part its own way; a
    case driven by a prescribed displacement alone has compliance -u_p.r_p
    = -(twice its strain energy), which the optimizer makes stiffer."""
    spec = MeshSpec(nelx=30, nely=5)
    cases = [
        Case("service", [Force("deck", (0.0, -600.0))],
             constraints=[Fix("left"), Fix("right", dofs="y")]),
        Case("jacked", constraints=[Fix("left"), Fix("right"), Displace("mid", uy=0.01)]),
    ]
    _, s = _run_part(tmp_path, _span(), spec, constraints=[], load_cases=cases,
                     optimize_design=True, simp=SimpParams(max_iterations=10, filter_radius=2.0))
    jacked = s["solve"]["cases"][1]
    assert jacked["compliance"] < 0
    opt = s["optimization"]
    assert opt["case_compliances"][1] < 0
    assert opt["case_compliances"][1] > jacked["compliance"]  # 40% of the material, less stiff
    html = build_site(run_dir=tmp_path, site_dir=tmp_path / "site").read_text()
    assert "case &#x27;jacked&#x27; only: &#x27;mid&#x27;" in html


def test_a_case_whose_supports_leave_a_motion_free_is_named(tmp_path):
    with pytest.raises(ValueError, match="in case 'loose'"):
        _run_part(tmp_path, _span(), MeshSpec(nelx=12, nely=2), constraints=[],
                  load_cases=[
                      Case("held", [Force("deck", (0.0, -1.0))],
                           constraints=[Fix("left"), Fix("right", dofs="y")]),
                      Case("loose", [Force("deck", (0.0, -1.0))],
                           constraints=[Fix("left", dofs="y"), Fix("right", dofs="y")]),
                  ])


@pytest.mark.parametrize(
    "body, message",
    [
        ("load_cases=[LoadCase('a', [Force('tip', (0.0, -1.0, 0.0))])]", "every load case"),
        ("constraints=[Fix('wall')], load_cases=[LoadCase('a', constraints=[Fix('wall')])]",
         "Displace"),
        ("constraints=[Fix('wall')], load_cases=[LoadCase('a', [Force('tip', (0.0, -1.0, 0.0))],"
         " constraints=[Force('tip', (0.0, 1.0, 0.0))])]", "constraints takes"),
    ],
)
def test_case_constraints_are_checked(tmp_path, body, message):
    (tmp_path / "part.py").write_text(_part())
    study = tmp_path / "study.py"
    study.write_text(f"from topocombo.study import *\nstudy = Study(part='part.py', {body})\n")
    with pytest.raises(ValueError, match=message):
        load_study(study)


# --------------------------------------------------------------------------
# Heaviside projection
# --------------------------------------------------------------------------
from topocombo.mma import design_map  # noqa: E402
from topocombo.optimize import Continuation, project  # noqa: E402

BRIDGE = Path(__file__).resolve().parents[1] / "examples" / "bridge"


def test_projection_keeps_the_ends_and_its_slope_is_exact():
    x = np.linspace(0.0, 1.0, 41)
    for beta in (1.0, 8.0, 64.0):
        y, dy = project(x, beta, 0.3)
        assert y[0] == pytest.approx(0.0, abs=1e-15) and y[-1] == pytest.approx(1.0)
        assert np.all(np.diff(y) >= 0)  # it saturates to 0 and 1 for a large beta
        fd = (project(x + 1e-7, beta, 0.3)[0] - project(x - 1e-7, beta, 0.3)[0]) / 2e-7
        assert np.allclose(dy, fd, rtol=1e-5, atol=1e-8)
    sharp, _ = project(np.array([0.2, 0.4]), 256.0, 0.3)
    assert sharp == pytest.approx([0.0, 1.0], abs=1e-9)
    assert project(x, 0.0)[0] is x  # beta 0: no projection


def test_beta_doubles_on_schedule_or_when_the_design_settles():
    beta = Continuation(SimpParams(projection=8.0, projection_every=3, optimizer="mma"))
    seen = [beta.beta]
    for change in (0.5, 0.5, 0.5, 0.001, 0.5, 0.5, 0.5, 0.001):
        beta.step(change, 0.01)
        seen.append(beta.beta)
    assert seen == [1, 1, 1, 2, 4, 4, 4, 8, 8]
    assert beta.at_final
    off = Continuation(SimpParams())
    assert off.beta == 0.0 and off.at_final and not off.step(0.0, 0.01)


def test_the_gradient_chain_through_filter_and_projection_is_exact(small_mbb):
    """d(compliance)/d(design) through the density filter and a sharp
    projection, with held elements, against central differences."""
    m, presc, load = small_mbb["mesh"], small_mbb["presc"], small_mbb["load"]
    params = SimpParams(filter_radius=1.5, projection=8.0, optimizer="mma")
    beta = Continuation(params)
    beta.beta = 8.0
    passive = np.zeros(m.n_elements, bool)
    solid = np.zeros(m.n_elements, bool)
    passive[:3], solid[-3:] = True, True
    physical, to_design = design_map(m, params, passive, solid, beta)
    ke = element_stiffnesses(m, Material(), 1.0)

    def objective(x):
        xp = physical(x)
        res = solve_cases(m, Material(), 1.0, [load], None, densities=xp, ke_all=ke,
                          prescribed=presc)
        c, g, _ = R.compliance(res, [1.0], xp, 3.0)
        return c, to_design(g)

    x = np.random.default_rng(3).uniform(0.3, 0.7, m.n_elements)
    _, grad = objective(x)
    picks = np.random.default_rng(4).choice(m.n_elements, 12, replace=False)
    for e in picks:
        d = np.zeros(m.n_elements)
        d[e] = 1e-4  # smaller steps drown in roundoff (the compliance is ~100 N.mm)
        fd = (objective(x + d)[0] - objective(x - d)[0]) / 2e-4
        assert fd == pytest.approx(grad[e], rel=1e-6, abs=1e-9 * np.abs(grad).max())


def test_projection_makes_mma_designs_black_and_white(small_mbb):
    m, presc, load = small_mbb["mesh"], small_mbb["presc"], small_mbb["load"]
    base = dict(filter_radius=1.5, filter_type="density", max_iterations=400, optimizer="mma")
    grey = optimize_mma(m, Material(), 1.0, [(1.0, load)], SimpParams(**base), prescribed=presc)
    crisp = optimize_mma(m, Material(), 1.0, [(1.0, load)],
                         SimpParams(**base, projection=16.0, projection_every=30),
                         prescribed=presc)
    assert crisp.converged and crisp.history[-1]["beta"] == 16.0
    assert crisp.volume_fraction == pytest.approx(0.5, abs=1e-3)
    assert crisp.measure_of_discreteness() < 5.0 < grey.measure_of_discreteness()
    assert crisp.compliance < grey.compliance  # grey is penalised stiffness spent


def test_projection_is_refused_where_it_cannot_run():
    with pytest.raises(ValueError, match="needs optimizer='mma'"):
        SimpParams(projection=8.0, optimizer="oc")
    with pytest.raises(ValueError, match="needs optimizer='mma'"):
        SimpParams(projection=8.0, optimizer="nlopt")
    with pytest.raises(ValueError, match=">= 1"):
        SimpParams(projection=0.5)
    with pytest.raises(ValueError, match="eta"):
        SimpParams(projection_eta=1.0)


def test_the_bridge_example_settles_its_pier_in_one_case(tmp_path):
    """The published bridge: per-case supports, MMA with projection."""
    _, s = run_loaded(load_study(BRIDGE / "study.py"), tmp_path, echo=False)
    cases = {c["name"]: c for c in s["solve"]["cases"]}
    # the settled pier carries less of the deck, so the deck bends more
    assert cases["settled"]["compliance"] > cases["traffic"]["compliance"]
    opt = s["optimization"]
    assert opt["converged"] and opt["optimizer"] == "mma"
    assert opt["measure_of_discreteness"] < 5.0
    assert opt["volume_fraction"] == pytest.approx(0.4, abs=1e-3)
    log = (tmp_path / "pipeline.log").read_text()
    assert "case 'settled': constraint on 'middle' (1 nodes): uy = -0.05" in log
    assert "Heaviside projection" in log
    html = build_site(run_dir=tmp_path, site_dir=tmp_path / "site").read_text()
    assert "Heaviside projection" in html
    assert "case &#x27;settled&#x27; only: &#x27;middle&#x27;" in html
