"""Non-interactive figures for designs, densities, and convergence history."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .optimizer import OptimizationResult
from .problem import Evaluation, GPTOProblem


def plot_evaluation(
    path: str | Path,
    problem: GPTOProblem,
    evaluation: Evaluation,
    *,
    title: str | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if problem.case.mesh.dim == 2:
        _plot_2d(path, problem, evaluation, title=title)
    else:
        _plot_3d(path, problem, evaluation, title=title)


def _plot_2d(
    path: Path,
    problem: GPTOProblem,
    evaluation: Evaluation,
    *,
    title: str | None,
) -> None:
    mesh = problem.case.mesh
    geometry = evaluation.geometry
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    design_axis, density_axis = axes
    for bar_index, (start, end) in enumerate(geometry.bars):
        polygon = _capsule_polygon(
            geometry.points[start], geometry.points[end], geometry.radii[bar_index]
        )
        alpha = float(np.clip(geometry.size_variables[bar_index] ** 2, 0.02, 1.0))
        design_axis.fill(
            polygon[:, 0],
            polygon[:, 1],
            facecolor=(0.9, 0.08, 0.08, alpha),
            edgecolor=(0.2, 0.0, 0.0, min(1.0, alpha + 0.2)),
            linewidth=0.45,
        )
    design_axis.set_title("Explicit bar geometry", fontsize=11)

    vertices = mesh.coordinates[mesh.elements]
    collection = PolyCollection(
        vertices,
        array=evaluation.projection.volume_density,
        cmap="gray_r",
        edgecolors="none",
    )
    collection.set_clim(0.0, max(1.0, evaluation.maximum_volume_density))
    density_axis.add_collection(collection)
    density_axis.set_title(
        f"Projected volume density\nC={evaluation.compliance:.6g}, "
        f"vf={evaluation.volume_fraction:.6g}",
        fontsize=11,
    )
    figure.colorbar(collection, ax=density_axis, shrink=0.78, label=r"$\rho_e^V$")

    minimum = mesh.coordinate_min
    maximum = mesh.coordinate_max
    for axis in axes:
        axis.set_xlim(minimum[0], maximum[0])
        axis.set_ylim(minimum[1], maximum[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    figure.suptitle(title or problem.case.title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_3d(
    path: Path,
    problem: GPTOProblem,
    evaluation: Evaluation,
    *,
    title: str | None,
) -> None:
    geometry = evaluation.geometry
    figure = plt.figure(figsize=(13, 5.8), constrained_layout=True)
    geometry_axis = figure.add_subplot(121, projection="3d")
    density_axis = figure.add_subplot(122, projection="3d")
    for index, (start, end) in enumerate(geometry.bars):
        endpoints = geometry.points[[start, end]]
        alpha = float(np.clip(geometry.size_variables[index], 0.03, 1.0))
        geometry_axis.plot(
            endpoints[:, 0],
            endpoints[:, 1],
            endpoints[:, 2],
            color=(0.85, 0.05, 0.05, alpha),
            linewidth=max(0.5, 6.0 * geometry.radii[index]),
            solid_capstyle="round",
        )

    minimum = problem.case.mesh.coordinate_min
    maximum = problem.case.mesh.coordinate_max
    geometry_axis.set_title("Explicit bar geometry")
    density_title = _draw_density_isosurface(density_axis, problem, evaluation)
    density_axis.set_title(density_title)
    for axis in (geometry_axis, density_axis):
        axis.set_xlim(minimum[0], maximum[0])
        axis.set_ylim(minimum[1], maximum[1])
        axis.set_zlim(minimum[2], maximum[2])
        axis.set_box_aspect(maximum - minimum)
        axis.view_init(elev=22, azim=-130)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_zlabel("z")
    figure.suptitle(
        (title or problem.case.title)
        + f" — C={evaluation.compliance:.6g}, vf={evaluation.volume_fraction:.6g}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _draw_density_isosurface(
    axis: object, problem: GPTOProblem, evaluation: Evaluation
) -> str:
    mesh = problem.case.mesh
    shape = mesh.structured_shape
    density = evaluation.projection.volume_density
    if shape is not None and len(shape) == 3 and density.min() < 0.5 < density.max():
        try:
            from skimage.measure import marching_cubes

            nx, ny, nz = shape
            field = density.reshape(nz, nx, ny)
            dimensions = mesh.coordinate_max - mesh.coordinate_min
            dx, dy, dz = dimensions / np.asarray((nx, ny, nz))
            vertices, faces, _, _ = marching_cubes(
                field, level=0.5, spacing=(dz, dx, dy)
            )
            # marching_cubes follows array axes (z, x, y); plotting uses x/y/z.
            vertices = vertices[:, [1, 2, 0]]
            vertices += mesh.coordinate_min + 0.5 * np.array([dx, dy, dz])
            surface = Poly3DCollection(
                vertices[faces], facecolor="#3c4f65", edgecolor="none", alpha=0.9
            )
            axis.add_collection3d(surface)  # type: ignore[attr-defined]
            return r"Projected density isosurface, $\rho_e^V=0.5$"
        except ImportError:
            pass

    # Dependency-free fallback, also used when the requested level is absent.
    threshold = 0.5 if density.max() >= 0.5 else float(np.quantile(density, 0.98))
    selected = density >= threshold
    centroids = mesh.centroids[selected]
    values = density[selected]
    if len(centroids) > 15_000:
        stride = int(np.ceil(len(centroids) / 15_000))
        centroids = centroids[::stride]
        values = values[::stride]
    axis.scatter(  # type: ignore[attr-defined]
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c=values,
        cmap="gray_r",
        s=2,
        alpha=0.45,
        linewidths=0,
    )
    return rf"High-density element centroids, $\rho_e^V\geq {threshold:.3g}$"


def plot_history(path: str | Path, result: OptimizationResult, volume_limit: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = np.array([item.iteration for item in result.history])
    compliance = np.array([item.compliance for item in result.history])
    volume = np.array([item.volume_fraction for item in result.history])
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True, constrained_layout=True)
    axes[0].semilogy(iterations, compliance, color="#155eef", linewidth=1.8)
    axes[0].set_ylabel("Compliance")
    axes[0].grid(alpha=0.25)
    axes[1].plot(iterations, volume, color="#00956f", linewidth=1.8)
    axes[1].axhline(volume_limit, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("MMA iteration")
    axes[1].set_ylabel("Volume fraction")
    axes[1].grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _capsule_polygon(
    first: np.ndarray, second: np.ndarray, radius: float, samples: int = 36
) -> np.ndarray:
    direction = second - first
    angle = np.arctan2(direction[1], direction[0])
    first_angles = np.linspace(angle + np.pi / 2, angle + 3 * np.pi / 2, samples)
    second_angles = np.linspace(angle - np.pi / 2, angle + np.pi / 2, samples)
    first_cap = first + radius * np.column_stack(
        (np.cos(first_angles), np.sin(first_angles))
    )
    second_cap = second + radius * np.column_stack(
        (np.cos(second_angles), np.sin(second_angles))
    )
    return np.vstack((first_cap, second_cap))
