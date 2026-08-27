from __future__ import annotations

import numpy as np
import pytest

from gpto.geometry import (
    AggregationScheme,
    aggregate,
    penalize,
    regularized_heaviside,
    segment_distance_and_gradient,
)


def test_segment_distance_uses_all_three_finite_segment_branches() -> None:
    centroids = np.array([[-1.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
    points = np.array([[0.0, 0.0], [2.0, 0.0]])
    bars = np.array([[0, 1]])

    distance, gradient = segment_distance_and_gradient(centroids, points, bars)

    np.testing.assert_allclose(distance, [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(gradient[0, 0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(gradient[0, 1], [0.0, -0.5, 0.0, -0.5])
    np.testing.assert_allclose(gradient[0, 2], [0.0, 0.0, -1.0, 0.0])


@pytest.mark.parametrize("dim", [2, 3])
def test_regularized_heaviside_has_correct_endpoints_and_midpoint(dim: int) -> None:
    density, derivative = regularized_heaviside(
        np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), dim
    )
    np.testing.assert_allclose(density, [0.0, 0.0, 0.5, 1.0, 1.0])
    assert derivative[2] == pytest.approx(2.0 / np.pi if dim == 2 else 0.75)
    np.testing.assert_allclose(derivative[[0, 1, 3, 4]], 0.0)


def test_heaviside_derivative_matches_finite_difference() -> None:
    coordinates = np.array([-0.73, -0.2, 0.31, 0.82])
    step = 1.0e-7
    for dim in (2, 3):
        _, analytic = regularized_heaviside(coordinates, dim)
        plus, _ = regularized_heaviside(coordinates + step, dim)
        minus, _ = regularized_heaviside(coordinates - step, dim)
        np.testing.assert_allclose(analytic, (plus - minus) / (2 * step), rtol=2e-8)


def test_paper_modified_p_norm_is_intentionally_not_clipped() -> None:
    components = np.ones((2, 1))
    density, gradient = aggregate(
        components,
        8.0,
        AggregationScheme.LEGACY_MODIFIED_P_NORM,
        minimum_density=0.01,
    )
    assert density[0] > 1.0
    assert np.all(gradient > 0.0)


@pytest.mark.parametrize(
    "scheme",
    [
        AggregationScheme.LEGACY_MODIFIED_P_NORM,
        AggregationScheme.MODIFIED_P_MEAN,
        AggregationScheme.KS_UPPER,
        AggregationScheme.KS_LOWER,
        AggregationScheme.PROBABILISTIC_UNION,
    ],
)
def test_aggregation_gradients_match_finite_differences(
    scheme: AggregationScheme,
) -> None:
    components = np.array([[0.14, 0.81], [0.61, 0.37], [0.42, 0.22]])
    value, analytic = aggregate(components, 8.0, scheme, 0.01)
    step = 1.0e-7
    for component in range(components.shape[0]):
        plus = components.copy()
        minus = components.copy()
        plus[component] += step
        minus[component] -= step
        plus_value, _ = aggregate(plus, 8.0, scheme, 0.01)
        minus_value, _ = aggregate(minus, 8.0, scheme, 0.01)
        finite_difference = (plus_value - minus_value) / (2 * step)
        np.testing.assert_allclose(
            analytic[component], finite_difference, rtol=2e-7, atol=2e-9
        )
    assert value.shape == (2,)


@pytest.mark.parametrize("scheme", ["SIMP", "RAMP"])
def test_penalization_gradient_matches_finite_difference(scheme: str) -> None:
    values = np.array([0.15, 0.44, 0.91])
    output, derivative = penalize(values, 3.0, scheme)  # type: ignore[arg-type]
    step = 1.0e-7
    plus, _ = penalize(values + step, 3.0, scheme)  # type: ignore[arg-type]
    minus, _ = penalize(values - step, 3.0, scheme)  # type: ignore[arg-type]
    np.testing.assert_allclose(derivative, (plus - minus) / (2 * step), rtol=2e-8)
    assert np.all(output >= 0.0)
