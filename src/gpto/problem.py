"""End-to-end GPTO analysis and analytical design sensitivities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .config import CaseDefinition, FloatArray, Geometry
from .finite_elements import AnalysisResult, FiniteElementModel
from .geometry import ProjectionParameters, ProjectionResult, project_geometry


@dataclass(slots=True)
class Evaluation:
    design_variables: FloatArray
    geometry: Geometry
    projection: ProjectionResult
    finite_element: AnalysisResult
    compliance: float
    compliance_gradient: FloatArray
    volume_fraction: float
    volume_gradient: FloatArray

    @property
    def maximum_volume_density(self) -> float:
        return float(np.max(self.projection.volume_density))

    @property
    def maximum_stiffness_density(self) -> float:
        return float(np.max(self.projection.stiffness_density))


@dataclass(slots=True)
class GradientCheckEntry:
    index: int
    analytic_compliance: float
    finite_difference_compliance: float
    relative_error_compliance: float
    analytic_volume: float
    finite_difference_volume: float
    relative_error_volume: float


class DesignMap:
    """Map shared physical bar parameters to scaled global design variables."""

    def __init__(self, case: CaseDefinition) -> None:
        geometry = case.geometry
        self.dim = geometry.dim
        self.n_points = geometry.n_points
        self.n_bars = geometry.n_bars
        self.point_min = case.mesh.coordinate_min.astype(float)
        self.point_scale = (
            case.mesh.coordinate_max - case.mesh.coordinate_min
        ).astype(float)
        if np.any(self.point_scale <= 0):
            raise ValueError("design-region coordinate range must be positive")
        self.radius_min = float(case.radius_bounds[0])
        delta_radius = float(case.radius_bounds[1] - case.radius_bounds[0])
        self.radius_scale = 1.0 if delta_radius < 1.0e-12 else delta_radius
        self.radius_is_fixed = delta_radius < 1.0e-12

        self.n_point_variables = self.dim * self.n_points
        self.size_start = self.n_point_variables
        self.radius_start = self.size_start + self.n_bars
        self.n_variables = self.radius_start + self.n_bars
        point_variables = np.arange(self.n_point_variables, dtype=np.int64).reshape(
            self.n_points, self.dim
        )
        size_variables = self.size_start + np.arange(self.n_bars, dtype=np.int64)
        radius_variables = self.radius_start + np.arange(self.n_bars, dtype=np.int64)
        self.bar_variables = np.column_stack(
            (
                point_variables[geometry.bars[:, 0]],
                point_variables[geometry.bars[:, 1]],
                size_variables,
                radius_variables,
            )
        )
        self.lower_bounds = np.zeros(self.n_variables)
        self.upper_bounds = np.ones(self.n_variables)
        if self.radius_is_fixed:
            self.upper_bounds[radius_variables] = 0.0

    def encode(self, geometry: Geometry) -> FloatArray:
        if geometry.n_points != self.n_points or geometry.n_bars != self.n_bars:
            raise ValueError("geometry topology does not match the design map")
        design = np.empty(self.n_variables, dtype=float)
        design[: self.n_point_variables] = (
            (geometry.points - self.point_min) / self.point_scale
        ).ravel()
        design[self.size_start : self.radius_start] = geometry.size_variables
        design[self.radius_start :] = (
            geometry.radii - self.radius_min
        ) / self.radius_scale
        return design

    def decode(self, design_variables: FloatArray, bars: np.ndarray) -> Geometry:
        design = np.asarray(design_variables, dtype=float).reshape(-1)
        if design.shape != (self.n_variables,):
            raise ValueError(f"expected {self.n_variables} design variables")
        points = design[: self.n_point_variables].reshape(self.n_points, self.dim)
        points = self.point_min + points * self.point_scale
        sizes = design[self.size_start : self.radius_start].copy()
        radii = self.radius_min + design[self.radius_start :] * self.radius_scale
        return Geometry(points, bars.copy(), sizes, radii)

    def assemble_local_gradient(self, local_gradient: FloatArray) -> FloatArray:
        local = np.asarray(local_gradient, dtype=float)
        expected = (self.n_bars, 2 * self.dim + 2)
        if local.shape != expected:
            raise ValueError(f"local gradient must have shape {expected}")
        scaled = local.copy()
        scaled[:, : self.dim] *= self.point_scale
        scaled[:, self.dim : 2 * self.dim] *= self.point_scale
        scaled[:, -1] *= self.radius_scale
        global_gradient = np.zeros(self.n_variables, dtype=float)
        np.add.at(global_gradient, self.bar_variables.ravel(), scaled.ravel())
        return global_gradient


class GPTOProblem:
    """Paper formulation joined into one cache-aware optimization problem."""

    def __init__(
        self,
        case: CaseDefinition,
        projection_parameters: ProjectionParameters | None = None,
    ) -> None:
        self.case = case
        self.projection_parameters = projection_parameters or ProjectionParameters()
        self.design_map = DesignMap(case)
        self.finite_element = FiniteElementModel(
            case.mesh,
            case.material,
            case.fixed_dofs,
            case.forces,
            case.solver,
        )
        self.initial_design = self.design_map.encode(case.geometry)
        self._cached_design: FloatArray | None = None
        self._cached_evaluation: Evaluation | None = None
        self.analysis_count = 0

    @property
    def lower_bounds(self) -> FloatArray:
        return self.design_map.lower_bounds.copy()

    @property
    def upper_bounds(self) -> FloatArray:
        return self.design_map.upper_bounds.copy()

    def evaluate(
        self, design_variables: FloatArray, *, enforce_bounds: bool = True
    ) -> Evaluation:
        design = np.asarray(design_variables, dtype=float).reshape(-1)
        if (
            self._cached_design is not None
            and np.array_equal(design, self._cached_design)
            and self._cached_evaluation is not None
        ):
            return self._cached_evaluation
        if enforce_bounds and (
            np.any(design < self.design_map.lower_bounds - 1.0e-12)
            or np.any(design > self.design_map.upper_bounds + 1.0e-12)
        ):
            raise ValueError("design variables violate their global bounds")

        geometry = self.design_map.decode(design, self.case.geometry.bars)
        projection = project_geometry(
            self.case.mesh,
            geometry,
            self.case.material.minimum_density,
            self.projection_parameters,
        )
        finite_element = self.finite_element.analyze(projection.stiffness_density)

        compliance_local = projection.local_vjp(
            -finite_element.element_unit_strain_energy, stiffness=True
        )
        compliance_gradient = self.design_map.assemble_local_gradient(compliance_local)

        total_volume = float(np.sum(self.case.mesh.element_volumes))
        volume_weights = self.case.mesh.element_volumes / total_volume
        volume_fraction = float(volume_weights @ projection.volume_density)
        volume_local = projection.local_vjp(volume_weights, stiffness=False)
        volume_gradient = self.design_map.assemble_local_gradient(volume_local)

        evaluation = Evaluation(
            design_variables=design.copy(),
            geometry=geometry,
            projection=projection,
            finite_element=finite_element,
            compliance=finite_element.compliance,
            compliance_gradient=compliance_gradient,
            volume_fraction=volume_fraction,
            volume_gradient=volume_gradient,
        )
        self._cached_design = design.copy()
        self._cached_evaluation = evaluation
        self.analysis_count += 1
        return evaluation

    def evaluate_geometry(self, geometry: Geometry) -> Evaluation:
        return self.evaluate(self.design_map.encode(geometry))

    def objective(self, design_variables: FloatArray) -> tuple[float, FloatArray]:
        evaluation = self.evaluate(design_variables)
        return evaluation.compliance, evaluation.compliance_gradient.copy()

    def constraint(self, design_variables: FloatArray) -> tuple[FloatArray, FloatArray]:
        evaluation = self.evaluate(design_variables)
        value = np.array(
            [evaluation.volume_fraction - self.case.volume_fraction_limit]
        )
        gradient = evaluation.volume_gradient.reshape(1, -1)
        return value, gradient

    def gradient_check(
        self,
        design_variables: FloatArray | None = None,
        *,
        indices: Iterable[int] | None = None,
        step: float = 1.0e-6,
        method: Literal["central", "forward"] = "central",
        enforce_bounds: bool = True,
    ) -> list[GradientCheckEntry]:
        """Finite-difference check of selected global variables.

        The default central, bound-respecting mode is the robust diagnostic.
        Smith and Norato's Figure 14 instead uses a forward difference for
        every scaled variable, even when the perturbation lies infinitesimally
        outside a variable bound; that behavior is available with
        ``method="forward", enforce_bounds=False``.
        """

        if method not in ("central", "forward"):
            raise ValueError("method must be 'central' or 'forward'")

        design = (
            self.initial_design.copy()
            if design_variables is None
            else np.asarray(design_variables, dtype=float).reshape(-1).copy()
        )
        baseline = self.evaluate(design)
        if indices is None:
            indices = range(self.design_map.n_variables)
        entries: list[GradientCheckEntry] = []
        for raw_index in indices:
            index = int(raw_index)
            lower = self.design_map.lower_bounds[index]
            upper = self.design_map.upper_bounds[index]
            if upper <= lower:
                continue
            positive_step = step if not enforce_bounds else min(step, upper - design[index])
            negative_step = step if not enforce_bounds else min(step, design[index] - lower)
            if positive_step <= 0 and negative_step <= 0:
                continue
            if method == "forward" and positive_step > 0:
                plus = design.copy()
                plus[index] += positive_step
                plus_eval = self.evaluate(plus, enforce_bounds=enforce_bounds)
                fd_compliance = (
                    plus_eval.compliance - baseline.compliance
                ) / positive_step
                fd_volume = (
                    plus_eval.volume_fraction - baseline.volume_fraction
                ) / positive_step
            elif method == "forward":
                minus = design.copy()
                minus[index] -= negative_step
                minus_eval = self.evaluate(minus, enforce_bounds=enforce_bounds)
                fd_compliance = (
                    baseline.compliance - minus_eval.compliance
                ) / negative_step
                fd_volume = (
                    baseline.volume_fraction - minus_eval.volume_fraction
                ) / negative_step
            elif positive_step > 0 and negative_step > 0:
                plus = design.copy()
                minus = design.copy()
                plus[index] += positive_step
                minus[index] -= negative_step
                plus_eval = self.evaluate(plus, enforce_bounds=enforce_bounds)
                minus_eval = self.evaluate(minus, enforce_bounds=enforce_bounds)
                denominator = positive_step + negative_step
                fd_compliance = (plus_eval.compliance - minus_eval.compliance) / denominator
                fd_volume = (
                    plus_eval.volume_fraction - minus_eval.volume_fraction
                ) / denominator
            elif positive_step > 0:
                plus = design.copy()
                plus[index] += positive_step
                plus_eval = self.evaluate(plus, enforce_bounds=enforce_bounds)
                fd_compliance = (
                    plus_eval.compliance - baseline.compliance
                ) / positive_step
                fd_volume = (
                    plus_eval.volume_fraction - baseline.volume_fraction
                ) / positive_step
            else:
                minus = design.copy()
                minus[index] -= negative_step
                minus_eval = self.evaluate(minus, enforce_bounds=enforce_bounds)
                fd_compliance = (
                    baseline.compliance - minus_eval.compliance
                ) / negative_step
                fd_volume = (
                    baseline.volume_fraction - minus_eval.volume_fraction
                ) / negative_step

            analytic_compliance = float(baseline.compliance_gradient[index])
            analytic_volume = float(baseline.volume_gradient[index])
            entries.append(
                GradientCheckEntry(
                    index=index,
                    analytic_compliance=analytic_compliance,
                    finite_difference_compliance=float(fd_compliance),
                    relative_error_compliance=_relative_error(
                        analytic_compliance, float(fd_compliance)
                    ),
                    analytic_volume=analytic_volume,
                    finite_difference_volume=float(fd_volume),
                    relative_error_volume=_relative_error(
                        analytic_volume, float(fd_volume)
                    ),
                )
            )
        # Restore the baseline so subsequent objective/constraint calls share it.
        self.evaluate(design)
        return entries


def _relative_error(first: float, second: float) -> float:
    scale = max(abs(first), abs(second), 1.0e-12)
    return abs(first - second) / scale
