"""End-to-end checks for the geometry → mesh stage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from topocombo.geometry import BeamDomain
from topocombo.mesh_io import load_mesh
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

    areas = mesh.element_areas()
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
    assert data["quads"].shape == (spec.n_elements, 4)
    assert f"set_{PHYS_FIXED}" in data
    assert int(data["load_node"][0]) in range(spec.n_nodes)
    assert (coarse_run["dir"] / "mesh" / "mesh.vtu").exists()


def test_run_log_written(coarse_run):
    run_json = json.loads((coarse_run["dir"] / "run.json").read_text())
    names = [s["name"] for s in run_json["steps"]]
    assert names == ["geometry", "meshing", "validation", "export"]
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
