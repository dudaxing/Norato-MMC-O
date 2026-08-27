"""A compact single-constraint MMA optimizer for the article examples.

All three published examples have exactly one volume inequality. This module
implements that useful MMA special case directly from the separable reciprocal
approximation. It is independent of the GPL Python file bundled in PyGPTO and
keeps the reproduction project license-clean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable

import numpy as np

from .config import FloatArray, OptimizationOptions
from .problem import Evaluation, GPTOProblem


IterationCallback = Callable[["IterationRecord", Evaluation], None]


@dataclass(slots=True)
class IterationRecord:
    iteration: int
    compliance: float
    volume_fraction: float
    constraint_value: float
    constraint_violation: float
    design_change_norm: float
    kkt_norm: float
    dual_multiplier: float
    maximum_volume_density: float
    maximum_stiffness_density: float
    linear_solver_iterations: int
    linear_relative_residual: float
    elapsed_seconds: float


@dataclass(slots=True)
class OptimizationResult:
    design_variables: FloatArray
    evaluation: Evaluation
    history: list[IterationRecord]
    status: str
    message: str
    analysis_count: int

    @property
    def iterations(self) -> int:
        return self.history[-1].iteration if self.history else 0


@dataclass(slots=True)
class _MMAState:
    lower_asymptote: FloatArray
    upper_asymptote: FloatArray
    old_design_1: FloatArray
    old_design_2: FloatArray


class SingleConstraintMMA:
    """Method of moving asymptotes for one smooth inequality constraint."""

    def __init__(
        self,
        *,
        asymptote_initial: float = 0.5,
        asymptote_increase: float = 1.2,
        asymptote_decrease: float = 0.7,
        bound_fraction: float = 0.1,
        regularization: float = 1.0e-5,
    ) -> None:
        self.asymptote_initial = asymptote_initial
        self.asymptote_increase = asymptote_increase
        self.asymptote_decrease = asymptote_decrease
        self.bound_fraction = bound_fraction
        self.regularization = regularization

    def initialize(self, design: FloatArray, lower: FloatArray, upper: FloatArray) -> _MMAState:
        return _MMAState(
            lower_asymptote=lower.copy(),
            upper_asymptote=upper.copy(),
            old_design_1=design.copy(),
            old_design_2=design.copy(),
        )

    def step(
        self,
        iteration: int,
        design: FloatArray,
        objective_gradient: FloatArray,
        constraint_value: float,
        constraint_gradient: FloatArray,
        local_lower: FloatArray,
        local_upper: FloatArray,
        state: _MMAState,
    ) -> tuple[FloatArray, float, _MMAState]:
        x = np.asarray(design, dtype=float)
        xmin = np.asarray(local_lower, dtype=float)
        xmax = np.asarray(local_upper, dtype=float)
        span = xmax - xmin
        active = span > 1.0e-14
        safe_span = np.maximum(span, 1.0e-5)

        if iteration <= 2:
            low = x - self.asymptote_initial * span
            upp = x + self.asymptote_initial * span
        else:
            trend = (x - state.old_design_1) * (
                state.old_design_1 - state.old_design_2
            )
            factor = np.ones_like(x)
            factor[trend > 0] = self.asymptote_increase
            factor[trend < 0] = self.asymptote_decrease
            low = x - factor * (state.old_design_1 - state.lower_asymptote)
            upp = x + factor * (state.upper_asymptote - state.old_design_1)
            low = np.clip(low, x - 10.0 * span, x - 0.01 * span)
            upp = np.clip(upp, x + 0.01 * span, x + 10.0 * span)

        low[~active] = x[~active] - 1.0
        upp[~active] = x[~active] + 1.0
        alpha = np.maximum(
            np.maximum(low + self.bound_fraction * (x - low), x - span), xmin
        )
        beta = np.minimum(
            np.minimum(upp - self.bound_fraction * (upp - x), x + span), xmax
        )
        alpha[~active] = x[~active]
        beta[~active] = x[~active]

        upper_distance = upp - x
        lower_distance = x - low
        inverse_span = 1.0 / safe_span
        objective_positive = np.maximum(objective_gradient, 0.0)
        objective_negative = np.maximum(-objective_gradient, 0.0)
        objective_shift = (
            0.001 * (objective_positive + objective_negative)
            + self.regularization * inverse_span
        )
        p0 = (objective_positive + objective_shift) * upper_distance**2
        q0 = (objective_negative + objective_shift) * lower_distance**2

        constraint_positive = np.maximum(constraint_gradient, 0.0)
        constraint_negative = np.maximum(-constraint_gradient, 0.0)
        constraint_shift = (
            0.001 * (constraint_positive + constraint_negative)
            + self.regularization * inverse_span
        )
        p1 = (constraint_positive + constraint_shift) * upper_distance**2
        q1 = (constraint_negative + constraint_shift) * lower_distance**2
        approximation_rhs = float(
            np.sum(p1 / upper_distance + q1 / lower_distance)
            - constraint_value
        )

        def primal(dual: float) -> FloatArray:
            p = np.sqrt(np.maximum(p0 + dual * p1, np.finfo(float).tiny))
            q = np.sqrt(np.maximum(q0 + dual * q1, np.finfo(float).tiny))
            candidate = (q * upp + p * low) / (p + q)
            candidate = np.minimum(np.maximum(candidate, alpha), beta)
            candidate[~active] = x[~active]
            return candidate

        def approximate_constraint(candidate: FloatArray) -> float:
            return float(
                np.sum(p1 / (upp - candidate) + q1 / (candidate - low))
                - approximation_rhs
            )

        candidate = primal(0.0)
        if approximate_constraint(candidate) <= 0.0:
            dual = 0.0
        else:
            dual_lower = 0.0
            dual_upper = 1.0
            while approximate_constraint(primal(dual_upper)) > 0.0 and dual_upper < 1.0e16:
                dual_upper *= 10.0
            for _ in range(100):
                dual_mid = 0.5 * (dual_lower + dual_upper)
                if approximate_constraint(primal(dual_mid)) > 0.0:
                    dual_lower = dual_mid
                else:
                    dual_upper = dual_mid
                if dual_upper - dual_lower <= 1.0e-12 * max(1.0, dual_upper):
                    break
            dual = dual_upper
            candidate = primal(dual)

        new_state = _MMAState(
            lower_asymptote=low,
            upper_asymptote=upp,
            old_design_1=x.copy(),
            old_design_2=state.old_design_1.copy(),
        )
        return candidate, float(dual), new_state


def optimize(
    problem: GPTOProblem,
    *,
    design_variables: FloatArray | None = None,
    options: OptimizationOptions | None = None,
    callback: IterationCallback | None = None,
) -> OptimizationResult:
    """Optimize compliance subject to the article's volume constraint."""

    settings = options or problem.case.optimization
    x = (
        problem.initial_design.copy()
        if design_variables is None
        else np.asarray(design_variables, dtype=float).reshape(-1).copy()
    )
    lower = problem.lower_bounds
    upper = problem.upper_bounds
    if x.shape != lower.shape:
        raise ValueError("initial design has the wrong size")
    x = np.minimum(np.maximum(x, lower), upper)

    mma = SingleConstraintMMA()
    state = mma.initialize(x, lower, upper)
    move_span = settings.move_limit * np.abs(upper - lower)
    start = perf_counter()
    history: list[IterationRecord] = []
    evaluation = problem.evaluate(x)
    constraint_value = evaluation.volume_fraction - problem.case.volume_fraction_limit
    initial_record = _record(
        0,
        evaluation,
        constraint_value,
        design_change=0.0,
        kkt_norm=np.inf,
        dual=0.0,
        elapsed=perf_counter() - start,
    )
    history.append(initial_record)
    if callback is not None:
        callback(initial_record, evaluation)

    status = "maximum_iterations"
    message = "maximum iteration count reached"
    for iteration in range(1, settings.max_iterations + 1):
        local_lower = np.maximum(lower, x - move_span)
        local_upper = np.minimum(upper, x + move_span)
        candidate, dual, state = mma.step(
            iteration,
            x,
            evaluation.compliance_gradient,
            constraint_value,
            evaluation.volume_gradient,
            local_lower,
            local_upper,
            state,
        )
        design_change = float(np.linalg.norm(candidate - x))
        x = candidate
        evaluation = problem.evaluate(x)
        constraint_value = evaluation.volume_fraction - problem.case.volume_fraction_limit
        kkt_norm = _kkt_norm(
            x,
            lower,
            upper,
            evaluation.compliance_gradient,
            constraint_value,
            evaluation.volume_gradient,
            dual,
        )
        record = _record(
            iteration,
            evaluation,
            constraint_value,
            design_change,
            kkt_norm,
            dual,
            perf_counter() - start,
        )
        history.append(record)
        if callback is not None:
            callback(record, evaluation)

        if iteration > 1 and design_change <= settings.design_step_tolerance:
            status = "design_step_tolerance"
            message = "design step convergence tolerance satisfied"
            break
        if kkt_norm <= settings.kkt_tolerance:
            status = "kkt_tolerance"
            message = "KKT convergence tolerance satisfied"
            break

    return OptimizationResult(
        design_variables=x,
        evaluation=evaluation,
        history=history,
        status=status,
        message=message,
        analysis_count=problem.analysis_count,
    )


def _kkt_norm(
    design: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    objective_gradient: FloatArray,
    constraint_value: float,
    constraint_gradient: FloatArray,
    dual: float,
) -> float:
    lagrangian = objective_gradient + dual * constraint_gradient
    tolerance = 1.0e-8
    residual = lagrangian.copy()
    at_lower = design <= lower + tolerance
    at_upper = design >= upper - tolerance
    residual[at_lower] = np.minimum(residual[at_lower], 0.0)
    residual[at_upper] = np.maximum(residual[at_upper], 0.0)
    fixed = upper <= lower + tolerance
    residual[fixed] = 0.0
    parts = np.concatenate(
        (
            residual,
            np.array(
                [max(constraint_value, 0.0), dual * constraint_value]
            ),
        )
    )
    return float(np.linalg.norm(parts))


def _record(
    iteration: int,
    evaluation: Evaluation,
    constraint_value: float,
    design_change: float,
    kkt_norm: float,
    dual: float,
    elapsed: float,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        compliance=evaluation.compliance,
        volume_fraction=evaluation.volume_fraction,
        constraint_value=float(constraint_value),
        constraint_violation=max(float(constraint_value), 0.0),
        design_change_norm=design_change,
        kkt_norm=float(kkt_norm),
        dual_multiplier=float(dual),
        maximum_volume_density=evaluation.maximum_volume_density,
        maximum_stiffness_density=evaluation.maximum_stiffness_density,
        linear_solver_iterations=evaluation.finite_element.solver_iterations,
        linear_relative_residual=evaluation.finite_element.relative_residual,
        elapsed_seconds=float(elapsed),
    )
