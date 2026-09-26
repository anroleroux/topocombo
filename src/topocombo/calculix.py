"""CalculiX as the linear solver: write the model, run ``ccx``, read it back.

The same mesh, material, SIMP densities, supports and nodal forces the
built-in solver takes are written as a CalculiX input deck — one static step
per load case, each with its own boundary conditions — and the displacements
and reactions are read from the ``.dat`` file:

* quad4 -> ``CPS4`` (plane stress, the thickness on the section; CalculiX
  expands it into a layer of 3D elements internally), hex8 -> ``C3D8``,
  tet4 -> ``C3D4``, tet10 -> ``C3D10``.  The node orders of Gmsh/VTK and
  CalculiX agree for all four.
* SIMP: every distinct stiffness factor ``E_min + x^p (1 - E_min)`` becomes
  a material and an element set; at full density there is one.
* Supports: ``*BOUNDARY, OP=NEW`` per step, so cases may hold different DOFs.

CalculiX reads at most 20 characters per number, so the deck is written in
``%.12e`` form, and prints results with seven significant digits, which
bounds how closely the two solvers can agree.  ``ccx`` is an external program (the
``calculix-ccx`` package on Debian/Ubuntu); :func:`available` says whether
it is on the path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .mesh_io import Mesh

#: topocombo element name -> CalculiX element type
ELEMENT_TYPES = {"quad4": "CPS4", "hex8": "C3D8", "tet4": "C3D4", "tet10": "C3D10"}
#: the executable; override with the TOPOCOMBO_CCX environment variable
CCX = os.environ.get("TOPOCOMBO_CCX", "ccx")


def available() -> bool:
    """Whether the ``ccx`` executable can be found."""
    return shutil.which(CCX) is not None


def _num(value: float) -> str:
    """A number as CalculiX reads it: at most 20 characters per field (a
    longer one is cut, so -7.85e-14 in full precision reads as -0.785)."""
    return f"{float(value):.12e}"


@dataclass
class CcxCase:
    """One load case as CalculiX sees it: the global force vector and the
    constrained DOFs with their values."""

    force: np.ndarray  # (n_dofs,)
    fixed: np.ndarray  # constrained DOF indices
    values: np.ndarray  # their prescribed displacements


def write_input(mesh: Mesh, youngs_modulus: float, poisson_ratio: float, thickness: float,
                cases: Sequence[CcxCase], scale: np.ndarray | None = None) -> str:
    """The CalculiX input deck (text) for ``cases`` on ``mesh``.

    ``scale`` is each element's stiffness factor (SIMP); None is full
    density.  Nodes and elements are numbered from 1 in mesh order.
    """
    el = mesh.element
    if el.name not in ELEMENT_TYPES:
        raise ValueError(f"CalculiX has no element for {el.name}")
    d = mesh.dofs_per_node
    lines = ["*HEADING", "topocombo model", "*NODE, NSET=NALL"]
    xyz = np.zeros((mesh.n_nodes, 3))
    xyz[:, : mesh.nodes.shape[1]] = mesh.nodes
    lines += [f"{i + 1}, {_num(x)}, {_num(y)}, {_num(z)}" for i, (x, y, z) in enumerate(xyz.tolist())]
    lines.append(f"*ELEMENT, TYPE={ELEMENT_TYPES[el.name]}, ELSET=EALL")
    per_line = 15  # CalculiX continues an element's node list on the next line
    for i, cell in enumerate(mesh.cells.tolist(), start=1):
        ids = [str(n + 1) for n in cell]
        chunks = [ids[k:k + per_line] for k in range(0, len(ids), per_line)]
        lines.append(f"{i}, " + ", ".join(chunks[0]) + ("," if len(chunks) > 1 else ""))
        lines += [", ".join(c) for c in chunks[1:]]

    factors = np.ones(mesh.n_elements) if scale is None else np.asarray(scale, dtype=float)
    levels, group = np.unique(factors, return_inverse=True)
    section = [_num(thickness)] if d == 2 else []
    for g, factor in enumerate(levels.tolist()):
        members = np.flatnonzero(group == g) + 1
        lines.append(f"*ELSET, ELSET=E{g + 1}")
        lines += [", ".join(map(str, members[k:k + 16])) for k in range(0, members.size, 16)]
        lines += [f"*MATERIAL, NAME=M{g + 1}", "*ELASTIC",
                  f"{_num(youngs_modulus * factor)}, {_num(poisson_ratio)}",
                  f"*SOLID SECTION, ELSET=E{g + 1}, MATERIAL=M{g + 1}", *section]

    for case in cases:
        lines += ["*STEP", "*STATIC", "*BOUNDARY, OP=NEW"]
        lines += [f"{dof // d + 1}, {dof % d + 1}, {dof % d + 1}, {_num(v)}"
                  for dof, v in zip(case.fixed.tolist(), np.asarray(case.values, float).tolist())]
        lines.append("*CLOAD, OP=NEW")
        nz = np.flatnonzero(case.force)
        force = np.asarray(case.force, float)
        lines += [f"{dof // d + 1}, {dof % d + 1}, {_num(force[dof])}" for dof in nz.tolist()]
        lines += ["*NODE PRINT, NSET=NALL", "U", "*NODE PRINT, NSET=NALL", "RF",
                  "*END STEP"]
    return "\n".join(lines) + "\n"


def read_dat(text: str, n_nodes: int, dim: int, n_cases: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """(displacements, reactions), each (n_nodes * dim,), per step, from a
    CalculiX ``.dat`` file."""
    blocks: dict[str, list[np.ndarray]] = {"displacements": [], "forces": []}
    kind, rows = None, []

    def close() -> None:
        if kind is not None:
            field = np.zeros((n_nodes, 3))
            if rows:
                arr = np.array(rows)
                field[arr[:, 0].astype(int) - 1] = arr[:, 1:4]
            blocks[kind].append(field[:, :dim].reshape(-1))

    for line in text.splitlines():
        head = line.strip()
        if head.startswith("displacements (vx,vy,vz)") or head.startswith("forces (fx,fy,fz)"):
            close()
            kind, rows = head.split()[0], []
        elif kind is not None and head:
            parts = head.split()
            try:
                rows.append([float(p) for p in parts[:4]])
            except ValueError:  # another block's header
                close()
                kind = None
    close()
    u, r = blocks["displacements"], blocks["forces"]
    if len(u) != n_cases or len(r) != n_cases:
        raise RuntimeError(f"CalculiX wrote {len(u)} displacement and {len(r)} reaction "
                           f"blocks for {n_cases} load cases")
    return list(zip(u, r))


def run(mesh: Mesh, youngs_modulus: float, poisson_ratio: float, thickness: float,
        cases: Sequence[CcxCase], scale: np.ndarray | None = None,
        workdir: Path | None = None, threads: int = 1,
        ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Solve ``cases`` with CalculiX: (displacements, reactions) per case.

    The deck and CalculiX's files go to ``workdir`` (kept) or a temporary
    directory (removed).  ``threads`` defaults to 1: the multithreaded
    ``ccx`` 2.21 of Ubuntu returned wrong displacements for one step of a
    two-step deck in about 1 run in 20 (the same deck, 4 threads), and never
    in 40 single-threaded runs."""
    if not available():
        raise RuntimeError(
            f"CalculiX ('{CCX}') is not installed: apt install calculix-ccx, or set TOPOCOMBO_CCX"
        )
    deck = write_input(mesh, youngs_modulus, poisson_ratio, thickness, cases, scale)
    n = str(threads)
    env = {**os.environ, "OMP_NUM_THREADS": n, "CCX_NPROC_STIFFNESS": n,
           "CCX_NPROC_EQUATION_SOLVER": n, "NUMBER_OF_CPUS": n}

    def solve_in(folder: Path) -> list[tuple[np.ndarray, np.ndarray]]:
        (folder / "model.inp").write_text(deck)
        done = subprocess.run([CCX, "-i", "model"], cwd=folder, env=env,
                              capture_output=True, text=True)
        dat = folder / "model.dat"
        if done.returncode != 0 or not dat.exists() or "*ERROR" in done.stdout:
            tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-12:])
            raise RuntimeError(f"CalculiX failed (exit {done.returncode}):\n{tail}")
        return read_dat(dat.read_text(), mesh.n_nodes, mesh.dofs_per_node, len(cases))

    if workdir is not None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        return solve_in(workdir)
    with tempfile.TemporaryDirectory(prefix="topocombo-ccx-") as tmp:
        return solve_in(Path(tmp))
