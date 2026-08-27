"""Geometry projection operators and their analytical derivatives.

The equation references follow Smith and Norato (2020). A few signs in the
typeset article are inconsistent with the stated signed-distance convention;
this module follows the authors' MATLAB implementation:

    phi = bar_radius - distance_to_medial_axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp

from .config import FloatArray, Geometry, Mesh


class AggregationScheme(str, Enum):
    """Smooth approximations used to combine projected bar densities."""

    LEGACY_MODIFIED_P_NORM = "legacy_modified_p_norm"
    MODIFIED_P_MEAN = "modified_p_mean"
    KS_UPPER = "ks_upper"
    KS_LOWER = "ks_lower"
    PROBABILISTIC_UNION = "probabilistic_union"

    @classmethod
    def parse(cls, value: str | "AggregationScheme") -> "AggregationScheme":
        if isinstance(value, cls):
            return value
        aliases = {
            "legacy": cls.LEGACY_MODIFIED_P_NORM,
            "mod_p-norm": cls.LEGACY_MODIFIED_P_NORM,
            "modified_p_norm": cls.LEGACY_MODIFIED_P_NORM,
            "mod_p-mean": cls.MODIFIED_P_MEAN,
            "modified_p_mean": cls.MODIFIED_P_MEAN,
            "ks": cls.KS_UPPER,
            "ks_under": cls.KS_LOWER,
            "lower_ks": cls.KS_LOWER,
            "union": cls.PROBABILISTIC_UNION,
        }
        normalized = value.lower()
        try:
            if normalized in aliases:
                return aliases[normalized]
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown aggregation scheme {value!r}; choose from {choices}") from exc


@dataclass(frozen=True, slots=True)
class ProjectionParameters:
    """Parameters controlling projection, penalization, and aggregation."""

    aggregation: AggregationScheme = AggregationScheme.LEGACY_MODIFIED_P_NORM
    aggregation_parameter: float = 8.0
    penalization: Literal["SIMP", "RAMP"] = "SIMP"
    penalization_parameter: float = 3.0
    sample_radius: float | FloatArray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "aggregation", AggregationScheme.parse(self.aggregation))
        if self.aggregation_parameter <= 0:
            raise ValueError("aggregation_parameter must be positive")
        if self.penalization not in ("SIMP", "RAMP"):
            raise ValueError("penalization must be 'SIMP' or 'RAMP'")
        if self.penalization_parameter <= 0:
            raise ValueError("penalization_parameter must be positive")


@dataclass(slots=True)
class ProjectionResult:
    """Projected fields plus the factors needed for vector-Jacobian products."""

    volume_density: FloatArray
    stiffness_density: FloatArray
    component_density: FloatArray
    component_endpoint_gradient: FloatArray
    component_radius_gradient: FloatArray
    size_variables: FloatArray
    volume_aggregation_gradient: FloatArray
    stiffness_aggregation_gradient: FloatArray
    stiffness_penalty_gradient: FloatArray
    maximum_component_overlap: int

    def local_vjp(self, element_weights: FloatArray, *, stiffness: bool) -> FloatArray:
        """Return derivatives with respect to each bar's physical parameters.

        The output columns are `[x1, x2, alpha, radius]`, with `dim`
        coordinate columns for each endpoint. Shared endpoint assembly and
        design-variable scaling are deliberately handled by `DesignMap`.
        """

        weights = np.asarray(element_weights, dtype=float).reshape(-1)
        if weights.shape != self.volume_density.shape:
            raise ValueError("element_weights has the wrong number of elements")

        if stiffness:
            aggregation_gradient = self.stiffness_aggregation_gradient
            penalty_gradient = self.stiffness_penalty_gradient
        else:
            aggregation_gradient = self.volume_aggregation_gradient
            penalty_gradient = 1.0

        common = weights[None, :] * aggregation_gradient * penalty_gradient
        alpha = self.size_variables[:, None]

        endpoint_gradient = np.einsum(
            "be,bed->bd",
            common * alpha,
            self.component_endpoint_gradient,
            optimize=True,
        )
        size_gradient = np.sum(common * self.component_density, axis=1)
        radius_gradient = np.sum(
            common * alpha * self.component_radius_gradient, axis=1
        )
        return np.column_stack((endpoint_gradient, size_gradient, radius_gradient))


def segment_distance_and_gradient(
    centroids: FloatArray,
    points: FloatArray,
    bars: NDArray[np.integer],
    *,
    length_tolerance: float = 1.0e-12,
) -> tuple[FloatArray, FloatArray]:
    """Distance from every centroid to every finite medial-axis segment.

    Returns
    -------
    distance
        Array with shape `(n_bar, n_element)`.
    endpoint_gradient
        Physical derivatives with shape `(n_bar, n_element, 2*dim)`, ordered
        first endpoint then second endpoint. The three branches implement
        Eq. (14) and Eq. (30) of the article/code.
    """

    centroids = np.asarray(centroids, dtype=float)
    points = np.asarray(points, dtype=float)
    bars = np.asarray(bars, dtype=np.int64)
    dim = points.shape[1]

    x1 = points[bars[:, 0]]
    x2 = points[bars[:, 1]]
    axis = x2 - x1
    lengths = np.linalg.norm(axis, axis=1)
    safe_lengths = np.where(lengths < length_tolerance, 1.0, lengths)
    unit_axis = axis / safe_lengths[:, None]

    centroid_from_x1 = centroids[None, :, :] - x1[:, None, :]
    centroid_from_x2 = centroids[None, :, :] - x2[:, None, :]
    axial_coordinate = np.einsum(
        "bed,bd->be", centroid_from_x1, unit_axis, optimize=True
    )
    radial_vector = centroid_from_x1 - axial_coordinate[:, :, None] * unit_axis[:, None, :]

    branch_x1 = axial_coordinate <= 0.0
    branch_x2 = axial_coordinate >= lengths[:, None]
    # Degenerate bars follow the authors' implementation: the first endpoint
    # branch represents the collapsed capsule as a circle/sphere.
    branch_x2[lengths < length_tolerance] = False
    branch_middle = ~(branch_x1 | branch_x2)

    distance_x1 = np.linalg.norm(centroid_from_x1, axis=2)
    distance_x2 = np.linalg.norm(centroid_from_x2, axis=2)
    distance_middle = np.linalg.norm(radial_vector, axis=2)
    distance = np.where(
        branch_x1,
        distance_x1,
        np.where(branch_x2, distance_x2, distance_middle),
    )

    inverse_distance = np.zeros_like(distance)
    np.divide(1.0, distance, out=inverse_distance, where=distance > length_tolerance)
    axial_fraction = axial_coordinate / safe_lengths[:, None]

    gradient_x1 = np.zeros_like(centroid_from_x1)
    gradient_x2 = np.zeros_like(centroid_from_x2)
    gradient_x1[branch_x1] = (
        -centroid_from_x1 * inverse_distance[:, :, None]
    )[branch_x1]
    gradient_x2[branch_x2] = (
        -centroid_from_x2 * inverse_distance[:, :, None]
    )[branch_x2]
    middle_x1 = (
        -radial_vector
        * inverse_distance[:, :, None]
        * (1.0 - axial_fraction)[:, :, None]
    )
    middle_x2 = (
        -radial_vector * inverse_distance[:, :, None] * axial_fraction[:, :, None]
    )
    gradient_x1[branch_middle] = middle_x1[branch_middle]
    gradient_x2[branch_middle] = middle_x2[branch_middle]

    return distance, np.concatenate((gradient_x1, gradient_x2), axis=2)


def regularized_heaviside(
    normalized_signed_distance: FloatArray, dim: int
) -> tuple[FloatArray, FloatArray]:
    """Circular/spherical-cap projection of Eqs. (2), (3), and (28)."""

    x = np.asarray(normalized_signed_distance, dtype=float)
    density = np.zeros_like(x)
    derivative = np.zeros_like(x)
    inside = x >= 1.0
    transition = np.abs(x) < 1.0
    density[inside] = 1.0

    xt = x[transition]
    if dim == 2:
        root = np.sqrt(np.maximum(0.0, 1.0 - xt * xt))
        density[transition] = 1.0 + (xt * root - np.arccos(xt)) / np.pi
        derivative[transition] = 2.0 * root / np.pi
    elif dim == 3:
        density[transition] = 0.5 + 0.75 * xt - 0.25 * xt**3
        derivative[transition] = 0.75 * (1.0 - xt**2)
    else:
        raise ValueError("regularized_heaviside supports only 2D and 3D")
    return density, derivative


def penalize(
    x: FloatArray, parameter: float, scheme: Literal["SIMP", "RAMP"]
) -> tuple[FloatArray, FloatArray]:
    """SIMP or RAMP penalization of Eq. (4)."""

    if scheme == "SIMP":
        return x**parameter, parameter * x ** (parameter - 1.0)
    denominator = 1.0 + parameter * (1.0 - x)
    return x / denominator, (1.0 + parameter) / denominator**2


def aggregate(
    x: FloatArray,
    parameter: float,
    scheme: AggregationScheme | str,
    minimum_density: float,
) -> tuple[FloatArray, FloatArray]:
    """Combine bar densities and return the analytical component gradient.

    `LEGACY_MODIFIED_P_NORM` is the exact Eq. (6) implementation used for
    paper reproduction. It can exceed one in overlapping regions; this is
    intentionally *not* clipped. `KS_LOWER` and `PROBABILISTIC_UNION` are
    bounded alternatives and are reported as deviations from the paper.
    """

    x = np.asarray(x, dtype=float)
    scheme = AggregationScheme.parse(scheme)
    if x.ndim != 2:
        raise ValueError("aggregation input must have shape (n_component, n_element)")
    n_component = x.shape[0]
    p = float(parameter)
    rho_min = float(minimum_density)

    if scheme is AggregationScheme.LEGACY_MODIFIED_P_NORM:
        floor_power = rho_min**p
        scale = 1.0 - floor_power
        result = (floor_power + scale * np.sum(x**p, axis=0)) ** (1.0 / p)
        gradient = scale * x ** (p - 1.0) * result[None, :] ** (1.0 - p)
    elif scheme is AggregationScheme.MODIFIED_P_MEAN:
        floor_power = rho_min**p
        scale = 1.0 - floor_power
        result = (
            floor_power + scale * np.mean(x**p, axis=0)
        ) ** (1.0 / p)
        gradient = (
            scale
            / n_component
            * x ** (p - 1.0)
            * result[None, :] ** (1.0 - p)
        )
    elif scheme in (AggregationScheme.KS_UPPER, AggregationScheme.KS_LOWER):
        scaled = p * x
        log_sum = logsumexp(scaled, axis=0)
        if scheme is AggregationScheme.KS_LOWER:
            log_sum = log_sum - np.log(n_component)
        result = rho_min + (1.0 - rho_min) * log_sum / p
        shifted = scaled - logsumexp(scaled, axis=0, keepdims=True)
        gradient = (1.0 - rho_min) * np.exp(shifted)
    elif scheme is AggregationScheme.PROBABILISTIC_UNION:
        complement = 1.0 - x
        product = np.prod(complement, axis=0)
        union = 1.0 - product
        result = rho_min + (1.0 - rho_min) * union
        gradient = np.empty_like(x)
        for component in range(n_component):
            gradient[component] = (1.0 - rho_min) * np.prod(
                np.delete(complement, component, axis=0), axis=0
            )
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(f"unhandled aggregation scheme {scheme}")

    return result, gradient


def project_geometry(
    mesh: Mesh,
    geometry: Geometry,
    material_minimum_density: float,
    parameters: ProjectionParameters,
) -> ProjectionResult:
    """Evaluate the complete geometry -> volume/stiffness density chain."""

    distance, distance_gradient = segment_distance_and_gradient(
        mesh.centroids, geometry.points, geometry.bars
    )

    if parameters.sample_radius is None:
        sample_radius = (
            np.sqrt(mesh.dim)
            / 2.0
            * mesh.element_volumes ** (1.0 / mesh.dim)
        )
    else:
        sample_radius = np.asarray(parameters.sample_radius, dtype=float)
        if sample_radius.ndim == 0:
            sample_radius = np.full(mesh.n_elements, float(sample_radius))
        sample_radius = sample_radius.reshape(-1)
    if sample_radius.shape != (mesh.n_elements,) or np.any(sample_radius <= 0):
        raise ValueError("sample_radius must be positive and scalar or per-element")

    normalized_signed_distance = (
        geometry.radii[:, None] - distance
    ) / sample_radius[None, :]
    component_density, projection_derivative = regularized_heaviside(
        normalized_signed_distance, mesh.dim
    )
    component_endpoint_gradient = (
        projection_derivative[:, :, None]
        * (-1.0 / sample_radius)[None, :, None]
        * distance_gradient
    )
    component_radius_gradient = (
        projection_derivative / sample_radius[None, :]
    )

    effective_density = geometry.size_variables[:, None] * component_density
    volume_density, volume_aggregation_gradient = aggregate(
        effective_density,
        parameters.aggregation_parameter,
        parameters.aggregation,
        material_minimum_density,
    )

    penalized_effective_density, penalty_gradient = penalize(
        effective_density,
        parameters.penalization_parameter,
        parameters.penalization,
    )
    stiffness_density, stiffness_aggregation_gradient = aggregate(
        penalized_effective_density,
        parameters.aggregation_parameter,
        parameters.aggregation,
        material_minimum_density,
    )

    overlap = np.sum(component_density > 0.5, axis=0)
    return ProjectionResult(
        volume_density=volume_density,
        stiffness_density=stiffness_density,
        component_density=component_density,
        component_endpoint_gradient=component_endpoint_gradient,
        component_radius_gradient=component_radius_gradient,
        size_variables=geometry.size_variables.copy(),
        volume_aggregation_gradient=volume_aggregation_gradient,
        stiffness_aggregation_gradient=stiffness_aggregation_gradient,
        stiffness_penalty_gradient=penalty_gradient,
        maximum_component_overlap=int(overlap.max(initial=0)),
    )
