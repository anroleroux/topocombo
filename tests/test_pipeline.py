"""End-to-end checks for the geometry → mesh stage."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from topocombo.geometry import BeamDomain
from topocombo.mesh_io import Mesh, load_mesh
from topocombo.meshing import PHYS_FIXED, PHYS_LOAD, MeshSpec
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
    mesh = load_mesh(coarse_run["dir"] / "mesh" / "beam.msh")
    domain, spec = coarse_run["domain"], coarse_run["spec"]

    fixed = mesh.node_sets[PHYS_FIXED]
    load_edge = mesh.node_sets[PHYS_LOAD]
    assert fixed.size == spec.nely + 1
    assert load_edge.size == spec.nely + 1
    assert np.allclose(mesh.nodes[fixed, 0], 0.0)
    assert np.allclose(mesh.nodes[load_edge, 0], domain.length)


def test_load_node_sits_at_the_free_edge_mid_height(coarse_run):
    summary = coarse_run["summary"]
    assert summary["load_node_coords"] == pytest.approx(list(coarse_run["domain"].load_point))


def test_solver_facing_npz(coarse_run):
    data = np.load(coarse_run["dir"] / "mesh" / "mesh.npz")
    spec = coarse_run["spec"]
    assert data["nodes"].shape == (spec.n_nodes, 2)
    assert data["cells"].shape == (spec.n_elements, 4)
    assert str(data["cell_type"]) == "quad"
    assert f"set_{PHYS_FIXED}" in data
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
    material = Material()
    load = LoadCase(node=summary["load_node"], fy=-500.0)
    result = solve(mesh, material, domain.thickness, load, PHYS_FIXED)
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
        PHYS_FIXED,
    )
    assert np.allclose(doubled.u, 2.0 * base.u)
    assert doubled.compliance == pytest.approx(4.0 * base.compliance, rel=1e-9)


def test_density_scaling_softens_the_structure(slender_beam):
    mesh, material, domain = slender_beam["mesh"], slender_beam["material"], slender_beam["domain"]
    half = solve(
        mesh, material, domain.thickness, slender_beam["load"], PHYS_FIXED,
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
        fixed_node_set=PHYS_FIXED,
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
        optimized["load"], PHYS_FIXED,
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
    msh, domain, spec = hex_run["msh"], hex_run["domain"], hex_run["spec"]
    face_nodes = (spec.nely + 1) * (spec.nelz + 1)
    fixed = np.unique(_group_cells(msh, PHYS_FIXED, "quad"))
    loaded = np.unique(_group_cells(msh, PHYS_LOAD, "quad"))
    assert fixed.size == loaded.size == face_nodes
    assert np.allclose(msh.points[fixed, 0], 0.0)
    assert np.allclose(msh.points[loaded, 0], domain.length)
    # the load line (mid-height, across the width) lies on the loaded face
    on_line = np.isclose(msh.points[loaded, 1], domain.height / 2.0)
    assert on_line.sum() == spec.nelz + 1


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
    face_nodes = (spec.nely + 1) * (spec.nelz + 1)
    assert mesh.node_sets[PHYS_FIXED].size == mesh.node_sets[PHYS_LOAD].size == face_nodes

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
    line = nodes_on_segment(mesh, *domain.load_line, candidates=mesh.node_sets[PHYS_LOAD])
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
    return load_mesh(msh)


def _quad_mesh(out, domain: BeamDomain, spec: MeshSpec) -> Mesh:
    cad = export_domain(domain, out / "cad")
    msh, _ = generate_quad_mesh(domain, spec, cad["brep"], out / "mesh")
    return load_mesh(msh)


def _tip_line_load(mesh: Mesh, domain: BeamDomain3D, fy: float) -> LoadCase:
    line = nodes_on_segment(mesh, *domain.load_line, candidates=mesh.node_sets[PHYS_LOAD])
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

    r2 = solve(mesh2, material, width, LoadCase(node=find_node(mesh2, domain2.load_point), fy=-1000.0), PHYS_FIXED)
    load3 = _tip_line_load(mesh3, domain3, fy=-1000.0)
    assert load3.node_shares() == pytest.approx([0.5, 0.5])
    r3 = solve(mesh3, material, width, load3, PHYS_FIXED)

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
    result = solve(mesh, material, domain.width, load, PHYS_FIXED)
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
        LoadCase(node=load.node, fy=2.0 * load.fy, shares=load.shares), PHYS_FIXED,
    )
    assert np.allclose(doubled.u, 2.0 * base.u)
    half = solve(
        mesh, material, domain.width, load, PHYS_FIXED,
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
