from __future__ import annotations

from dataclasses import replace

import pytest

from gpto import GPTOProblem, build_case
from gpto.cases import load_reference_geometry
from gpto.config import SolverOptions


def test_paper_case_dimensions_and_design_variable_counts() -> None:
    mbb = build_case("mbb2d")
    assert (mbb.mesh.n_elements, mbb.mesh.n_nodes) == (10_000, 10_251)
    assert GPTOProblem(mbb).design_map.n_variables == 192

    bracket = build_case("lbracket2d")
    assert (bracket.mesh.n_elements, bracket.mesh.n_nodes) == (6_123, 6_320)
    assert GPTOProblem(bracket).design_map.n_variables == 66

    cantilever = build_case("cantilever3d", profile="tiny")
    assert (cantilever.geometry.n_bars, cantilever.geometry.n_points) == (16, 32)
    assert GPTOProblem(cantilever).design_map.n_variables == 128


@pytest.mark.slow
def test_mbb_paper_mesh_initial_state_matches_matlab_golden() -> None:
    problem = GPTOProblem(build_case("mbb2d"))
    evaluation = problem.evaluate(problem.initial_design)
    assert evaluation.compliance == pytest.approx(45.5269634564232, rel=8e-10)
    assert evaluation.volume_fraction == pytest.approx(0.333032744174828, abs=2e-13)


@pytest.mark.slow
def test_lbracket_reference_state_matches_paper_and_matlab_golden() -> None:
    problem = GPTOProblem(build_case("lbracket2d"))
    evaluation = problem.evaluate_geometry(load_reference_geometry("lbracket2d"))
    assert evaluation.compliance == pytest.approx(2.84607178503019, rel=2e-10)
    assert evaluation.volume_fraction == pytest.approx(0.300000636757151, abs=2e-12)


@pytest.mark.slow
def test_reduced_3d_cross_language_golden() -> None:
    case = build_case("cantilever3d", profile="tiny")
    case.solver = replace(SolverOptions(), kind="direct")
    problem = GPTOProblem(case)
    evaluation = problem.evaluate(problem.initial_design)
    assert evaluation.compliance == pytest.approx(28.1179447678968, rel=2e-10)
    assert evaluation.volume_fraction == pytest.approx(0.0227334243383602, abs=2e-13)
