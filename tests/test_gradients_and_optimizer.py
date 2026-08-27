from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gpto import GPTOProblem, build_case
from gpto.optimizer import optimize


def test_mbb_analytical_endpoint_and_size_gradients() -> None:
    problem = GPTOProblem(build_case("mbb2d", profile="smoke"))
    mapping = problem.design_map
    entries = problem.gradient_check(
        indices=[0, 1, mapping.size_start], step=1.0e-6
    )
    assert max(item.relative_error_compliance for item in entries) < 2.0e-5
    assert max(item.relative_error_volume for item in entries) < 2.0e-6


def test_lbracket_shared_endpoint_gradient_is_assembled_once_per_incident_bar() -> None:
    problem = GPTOProblem(build_case("lbracket2d", profile="smoke"))
    # Point 6 (zero-based 5) is shared by six bars in the connected layout.
    point_six_x = 5 * problem.design_map.dim
    entry = problem.gradient_check(indices=[point_six_x], step=1.0e-6)[0]
    assert entry.relative_error_compliance < 5.0e-5
    assert entry.relative_error_volume < 2.0e-6


def test_forward_gradient_check_matches_a_one_sided_perturbation() -> None:
    problem = GPTOProblem(build_case("mbb2d", profile="tiny"))
    design = problem.initial_design.copy()
    index = problem.design_map.size_start
    step = 1.0e-6
    entry = problem.gradient_check(
        design,
        indices=[index],
        step=step,
        method="forward",
        enforce_bounds=False,
    )[0]
    baseline = problem.evaluate(design)
    perturbed = design.copy()
    perturbed[index] += step
    forward = problem.evaluate(perturbed, enforce_bounds=False)
    expected = (forward.compliance - baseline.compliance) / step
    assert entry.finite_difference_compliance == pytest.approx(expected)
    assert np.isfinite(entry.relative_error_compliance)


@pytest.mark.slow
def test_first_mbb_mma_update_matches_matlab_snapshot() -> None:
    problem = GPTOProblem(build_case("mbb2d"))
    options = replace(problem.case.optimization, max_iterations=1)
    result = optimize(problem, options=options)
    assert len(result.history) == 2
    assert result.history[0].compliance == pytest.approx(45.5269634564232, rel=8e-10)
    assert result.history[1].compliance == pytest.approx(35.9482598466751, rel=2e-7)
    # The independent one-constraint subproblem solver uses an analytic dual
    # bisection instead of Svanberg's primal-dual barrier routine.  Its first
    # design differs by only a few scaled-variable micro-units.
    assert result.history[1].volume_fraction == pytest.approx(0.306384868943740, abs=5e-7)
