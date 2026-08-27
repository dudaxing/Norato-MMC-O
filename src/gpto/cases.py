"""Paper-faithful definitions of the three Smith-Norato examples."""

from __future__ import annotations

from importlib import resources

import numpy as np

from .config import (
    CaseDefinition,
    ExpectedResult,
    Geometry,
    OptimizationOptions,
    SolverOptions,
)
from .mesh import dof_indices, nodes_at, packaged_mesh, structured_mesh


_CASE_ALIASES = {
    "mbb": "mbb2d",
    "mbb2d": "mbb2d",
    "l": "lbracket2d",
    "lbracket": "lbracket2d",
    "l-bracket": "lbracket2d",
    "lbracket2d": "lbracket2d",
    "3d": "cantilever3d",
    "cantilever": "cantilever3d",
    "cantilever3d": "cantilever3d",
}


def available_cases() -> tuple[str, ...]:
    return ("mbb2d", "lbracket2d", "cantilever3d")


def build_case(name: str, *, profile: str = "paper") -> CaseDefinition:
    """Build one article case.

    Profiles only change mesh resolution. `paper` is the published mesh;
    `smoke` keeps the sample window no larger than the smallest initial bar
    radius; `tiny` is a fast pipeline check and is not a quantitative result.
    """

    normalized = _CASE_ALIASES.get(name.lower())
    if normalized is None:
        raise ValueError(f"unknown case {name!r}; choose from {available_cases()}")
    if profile not in ("paper", "smoke", "tiny"):
        raise ValueError("profile must be 'paper', 'smoke', or 'tiny'")
    builders = {
        "mbb2d": _mbb_case,
        "lbracket2d": _lbracket_case,
        "cantilever3d": _cantilever3d_case,
    }
    return builders[normalized](profile)


def load_reference_geometry(name: str) -> Geometry:
    normalized = _CASE_ALIASES.get(name.lower())
    if normalized is None:
        raise ValueError(f"unknown case {name!r}")
    resource = resources.files("gpto").joinpath(
        "data", f"reference_{normalized}.npz"
    )
    with resources.as_file(resource) as path, np.load(path) as archive:
        return Geometry(
            points=archive["points"],
            bars=archive["bars"],
            size_variables=archive["size_variables"],
            radii=archive["radii"],
        )


def _mbb_case(profile: str) -> CaseDefinition:
    counts = {
        "paper": (200, 50),
        "smoke": (80, 20),
        "tiny": (40, 10),
    }[profile]
    mesh = structured_mesh((20.0, 5.0), counts, name=f"mbb2d-{profile}")
    geometry = _mbb_initial_geometry()

    left = nodes_at(mesh, x=0.0)
    bottom_right = nodes_at(mesh, x=20.0, y=0.0)
    top_left = nodes_at(mesh, x=0.0, y=5.0)
    fixed = np.concatenate(
        (dof_indices(left, 2, (0,)), dof_indices(bottom_right, 2, (1,)))
    )
    forces = np.zeros(mesh.n_nodes * 2)
    forces[dof_indices(top_left, 2, (1,))] = -0.1 / len(top_left)

    return CaseDefinition(
        name="mbb2d",
        title="2D MBB half-beam",
        mesh=mesh,
        geometry=geometry,
        fixed_dofs=fixed,
        forces=forces,
        # The paper calls r=0.25 fixed; the released MATLAB case uses this
        # numerically negligible interval so MMA never receives a zero range.
        radius_bounds=(0.2499, 0.2501),
        volume_fraction_limit=0.45,
        solver=SolverOptions(kind="direct"),
        optimization=OptimizationOptions(
            max_iterations=300,
            move_limit=0.1,
            design_step_tolerance=1.0e-2,
            kkt_tolerance=1.0e-4,
        ),
        expected=ExpectedResult(
            paper_iterations=88,
            paper_compliance=4.201067,
            volume_fraction=0.45,
            notes="Figure 6-8; paper mesh 200x50 and MMA.",
        ),
        profile=profile,
    )


def _lbracket_case(profile: str) -> CaseDefinition:
    filename = "lbracket_fine.npz" if profile == "paper" else "lbracket_coarse.npz"
    mesh = packaged_mesh(filename, name=f"lbracket2d-{profile}")
    geometry = _lbracket_initial_geometry()

    top = nodes_at(mesh, y=100.0)
    loaded = nodes_at(mesh, x=100.0, y=40.0)
    fixed = dof_indices(top, 2, (0, 1))
    forces = np.zeros(mesh.n_nodes * 2)
    forces[dof_indices(loaded, 2, (1,))] = -0.1 / len(loaded)

    return CaseDefinition(
        name="lbracket2d",
        title="2D L-bracket with connected bars",
        mesh=mesh,
        geometry=geometry,
        fixed_dofs=fixed,
        forces=forces,
        radius_bounds=(2.0, 3.0),
        volume_fraction_limit=0.30,
        solver=SolverOptions(kind="direct"),
        optimization=OptimizationOptions(
            max_iterations=300,
            move_limit=0.1,
            design_step_tolerance=5.0e-3,
            kkt_tolerance=1.0e-4,
        ),
        expected=ExpectedResult(
            paper_iterations=64,
            paper_compliance=2.846072,
            volume_fraction=0.30,
            notes="Figures 11-13; fine Gmsh mesh has 6123 Q4 elements.",
        ),
        profile=profile,
    )


def _cantilever3d_case(profile: str) -> CaseDefinition:
    counts = {
        "paper": (80, 40, 40),
        "smoke": (40, 20, 20),
        "tiny": (20, 10, 10),
    }[profile]
    mesh = structured_mesh(
        (20.0, 10.0, 10.0), counts, name=f"cantilever3d-{profile}"
    )
    geometry = _cantilever3d_initial_geometry()

    corners = np.concatenate(
        [
            nodes_at(mesh, x=0.0, y=y, z=z)
            for y in (0.0, 10.0)
            for z in (0.0, 10.0)
        ]
    )
    loaded = nodes_at(mesh, x=20.0, y=5.0, z=5.0)
    fixed = dof_indices(corners, 3, (0, 1, 2))
    forces = np.zeros(mesh.n_nodes * 3)
    forces[dof_indices(loaded, 3, (2,))] = -0.1 / len(loaded)

    return CaseDefinition(
        name="cantilever3d",
        title="3D cantilever with floating cylindrical bars",
        mesh=mesh,
        geometry=geometry,
        fixed_dofs=fixed,
        forces=forces,
        radius_bounds=(0.5, 1.0),
        volume_fraction_limit=0.10,
        solver=SolverOptions(
            kind="cg", relative_tolerance=1.0e-5, max_iterations=10_000
        ),
        optimization=OptimizationOptions(
            max_iterations=200,
            move_limit=0.1,
            design_step_tolerance=1.0e-2,
            kkt_tolerance=1.0e-4,
        ),
        expected=ExpectedResult(
            paper_iterations=106,
            paper_compliance=None,
            volume_fraction=0.10,
            notes="Figures 15-18; paper mesh 80x40x40, reported runtime 63 min.",
        ),
        profile=profile,
    )


def _mbb_initial_geometry() -> Geometry:
    points: list[tuple[float, float]] = []
    bars: list[tuple[int, int]] = []
    for block in range(4):
        left = 0.25 + 5.0 * block
        right = 4.75 + 5.0 * block
        endpoints = (
            ((left, 4.75), (right, 4.75)),
            ((left, 2.75), (right, 4.75)),
            ((left, 4.75), (right, 2.75)),
            ((left, 2.75), (right, 2.75)),
            ((left, 2.25), (right, 2.25)),
            ((left, 0.25), (right, 2.25)),
            ((left, 2.25), (right, 0.25)),
            ((left, 0.25), (right, 0.25)),
        )
        for first, second in endpoints:
            start = len(points)
            points.extend((first, second))
            bars.append((start, start + 1))
    n_bar = len(bars)
    return Geometry(
        points=np.asarray(points),
        bars=np.asarray(bars),
        size_variables=np.full(n_bar, 0.5),
        radii=np.full(n_bar, 0.25),
    )


def _lbracket_initial_geometry() -> Geometry:
    points = np.array(
        [
            [12.5, 12.5],
            [12.5, 37.5],
            [12.5, 62.5],
            [12.5, 87.5],
            [37.5, 12.5],
            [37.5, 37.5],
            [37.5, 62.5],
            [37.5, 87.5],
            [62.5, 12.5],
            [62.5, 37.5],
            [87.5, 12.5],
            [87.5, 37.5],
        ]
    )
    bars = np.array(
        [
            [1, 2], [1, 5], [2, 3], [2, 5], [2, 6], [3, 4], [3, 6],
            [3, 7], [4, 7], [4, 8], [5, 6], [5, 9], [6, 7], [6, 9],
            [6, 10], [7, 8], [9, 10], [9, 11], [10, 11], [10, 12],
            [11, 12],
        ],
        dtype=np.int64,
    ) - 1
    return Geometry(points, bars, np.full(21, 0.5), np.full(21, 2.0))


def _cantilever3d_initial_geometry() -> Geometry:
    points: list[tuple[float, float, float]] = []
    bars: list[tuple[int, int]] = []
    # This nesting reproduces the point order in initial_geometry_cantilever3d.m.
    for z in (2.5, 7.5):
        for y in (2.5, 7.5):
            for center_x in (2.5, 7.5, 12.5, 17.5):
                start = len(points)
                points.extend(
                    (
                        (center_x - 0.1, y, z),
                        (center_x + 0.1, y, z),
                    )
                )
                bars.append((start, start + 1))
    return Geometry(
        np.asarray(points),
        np.asarray(bars),
        np.full(16, 0.5),
        np.full(16, 0.75),
    )
