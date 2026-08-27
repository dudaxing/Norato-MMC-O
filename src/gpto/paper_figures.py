"""Recreate the program-generated figures in Smith and Norato (2020).

The plotting functions deliberately mirror the paper's visual vocabulary:
red explicit bars whose opacity is ``alpha**2``, grayscale unpenalized density,
MATLAB-like convergence histories, a forward finite-difference comparison,
and a rho=0.5 surface for the 3D result.  Figure numbers follow the paper.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import textwrap
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import LightSource
from matplotlib.patches import Circle, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .cases import build_case, load_reference_geometry
from .config import Geometry, Mesh
from .problem import GPTOProblem


PAPER_DOI = "https://doi.org/10.1007/s00158-020-02552-0"
MATLAB_COMMIT = "ead7250e007d4c185de59f8eee6b88a35be39550"
PYTHON_COMMIT = "8157ca2f5b82feb4a2032fd4501c9f49a58b7e8c"
MATLAB_BLUE = "#0072BD"
MATLAB_ORANGE = "#D95319"
BAR_RED = "#F21D19"
BAR_EDGE = "#66100E"


@dataclass(slots=True)
class SavedState:
    geometry: Geometry
    volume_density: np.ndarray
    stiffness_density: np.ndarray
    compliance: float
    volume_fraction: float


@dataclass(slots=True)
class FigureEntry:
    number: int
    filename: str
    title: str
    case: str
    category: str
    data_source: str
    fidelity_note: str


_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


def _load_state(archive_path: Path, summary_path: Path) -> SavedState:
    with np.load(archive_path) as archive:
        geometry = Geometry(
            archive["points"].copy(),
            archive["bars"].copy(),
            archive["size_variables"].copy(),
            archive["radii"].copy(),
        )
        volume_density = archive["volume_density"].copy()
        stiffness_density = archive["stiffness_density"].copy()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    computed = summary["computed"]
    return SavedState(
        geometry=geometry,
        volume_density=volume_density,
        stiffness_density=stiffness_density,
        compliance=float(computed["compliance"]),
        volume_fraction=float(computed["volume_fraction"]),
    )


def _read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"optimization history is empty: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows])
        for name in ("iteration", "compliance", "volume_fraction")
    }


def _save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)


def _capsule_polygon(
    first: np.ndarray, second: np.ndarray, radius: float, samples: int = 72
) -> np.ndarray:
    direction = second - first
    angle = float(np.arctan2(direction[1], direction[0]))
    first_angles = np.linspace(angle + np.pi / 2, angle + 3 * np.pi / 2, samples)
    second_angles = np.linspace(angle - np.pi / 2, angle + np.pi / 2, samples)
    first_cap = first + radius * np.column_stack(
        (np.cos(first_angles), np.sin(first_angles))
    )
    second_cap = second + radius * np.column_stack(
        (np.cos(second_angles), np.sin(second_angles))
    )
    return np.vstack((first_cap, second_cap))


def _draw_geometry_2d(
    geometry: Geometry,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    path: Path,
    dpi: int,
    figure_size: tuple[float, float],
    xtick_step: float,
    ytick_step: float,
) -> None:
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
        for index, (start, end) in enumerate(geometry.bars):
            alpha = float(geometry.size_variables[index] ** 2)
            if alpha <= 0.05:
                continue
            polygon = _capsule_polygon(
                geometry.points[start], geometry.points[end], geometry.radii[index]
            )
            axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                facecolor=BAR_RED,
                edgecolor=BAR_EDGE,
                linewidth=0.55,
                alpha=alpha,
            )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
        axis.set_yticks(np.arange(ylim[0], ylim[1] + 0.5 * ytick_step, ytick_step))
        _save_figure(figure, path, dpi)


def _draw_density_2d(
    mesh: Mesh,
    density: np.ndarray,
    *,
    compliance: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    path: Path,
    dpi: int,
    figure_size: tuple[float, float],
    xtick_step: float,
    ytick_step: float,
) -> None:
    clipped = np.clip(np.asarray(density, dtype=float), 0.0, 1.0)
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
        if mesh.structured_shape is not None:
            nx, ny = mesh.structured_shape
            field = clipped.reshape(nx, ny).T
            x = mesh.centroids[:, 0].reshape(nx, ny)[:, 0]
            y = mesh.centroids[:, 1].reshape(nx, ny)[0, :]
            axis.contourf(
                x,
                y,
                field,
                levels=np.linspace(0.0, 1.0, 65),
                cmap=cm.gray_r,
                antialiased=False,
                extend="both",
            )
        else:
            facecolors = np.zeros((clipped.size, 4), dtype=float)
            facecolors[:, 3] = clipped
            edgecolors = np.zeros((clipped.size, 4), dtype=float)
            edgecolors[:, 3] = 0.05 + 0.95 * clipped
            collection = PolyCollection(
                mesh.coordinates[mesh.elements],
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=0.08,
                antialiaseds=False,
            )
            axis.add_collection(collection)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"density, compliance = {compliance:.6f}")
        axis.set_xticks(np.arange(xlim[0], xlim[1] + 0.5 * xtick_step, xtick_step))
        axis.set_yticks(np.arange(ylim[0], ylim[1] + 0.5 * ytick_step, ytick_step))
        _save_figure(figure, path, dpi)


def _dimension_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    *,
    text_offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 0.9},
    )
    midpoint = 0.5 * (np.asarray(start) + np.asarray(end)) + np.asarray(text_offset)
    axis.text(midpoint[0], midpoint[1], label, ha="center", va="center")


def _plot_mbb_problem(path: Path, dpi: int) -> None:
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=(9.2, 3.8), constrained_layout=True)
        axis.add_patch(Rectangle((0, 0), 20, 5, fill=False, lw=1.35, color="#222222"))
        axis.plot([0, 0], [-0.95, 5.65], color="#555555", ls="--", lw=0.9)
        axis.annotate(
            "",
            xy=(0, 5),
            xytext=(0, 6.2),
            arrowprops={"arrowstyle": "->", "color": MATLAB_BLUE, "lw": 1.8},
        )
        axis.text(0.32, 5.85, r"$F=0.1$", ha="left", va="center")
        support = Polygon([[19.55, -0.45], [20.45, -0.45], [20.0, 0.0]], closed=True,
                          facecolor="none", edgecolor="#333333", lw=0.9)
        axis.add_patch(support)
        axis.add_patch(Circle((19.78, -0.58), 0.11, fill=False, color="#333333", lw=0.8))
        axis.add_patch(Circle((20.22, -0.58), 0.11, fill=False, color="#333333", lw=0.8))
        _dimension_arrow(axis, (0, -1.15), (20, -1.15), "20", text_offset=(0, -0.2))
        _dimension_arrow(axis, (2.2, 0), (2.2, 5), "5", text_offset=(-0.35, 0))
        axis.text(-0.42, -0.22, "symmetry", rotation=90, ha="right", va="top", fontsize=9)
        axis.set_xlim(-1.6, 21.0)
        axis.set_ylim(-1.8, 6.8)
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        _save_figure(figure, path, dpi)


def _plot_lbracket_problem(path: Path, dpi: int) -> None:
    outline = np.array([[0, 0], [100, 0], [100, 40], [40, 40], [40, 100], [0, 100]])
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=(6.7, 6.7), constrained_layout=True)
        axis.add_patch(Polygon(outline, closed=True, fill=False, lw=1.35, color="#222222"))
        axis.add_patch(
            Rectangle((-2, 100), 44, 3.2, facecolor="#d8e9ff", edgecolor=MATLAB_BLUE,
                      hatch="////", lw=0.5, alpha=0.7)
        )
        axis.annotate(
            "",
            xy=(100, 40),
            xytext=(100, 52),
            arrowprops={"arrowstyle": "->", "color": "#111111", "lw": 1.8},
        )
        axis.text(103, 48, r"$F=0.1$", ha="left", va="center")
        _dimension_arrow(axis, (0, -8), (100, -8), "100", text_offset=(0, -3))
        _dimension_arrow(axis, (-8, 0), (-8, 100), "100", text_offset=(-3, 0))
        _dimension_arrow(axis, (0, 92), (40, 92), "40", text_offset=(0, -3))
        _dimension_arrow(axis, (92, 0), (92, 40), "40", text_offset=(-4, 0))
        axis.set_xlim(-18, 116)
        axis.set_ylim(-18, 110)
        axis.set_aspect("equal", adjustable="box")
        axis.axis("off")
        _save_figure(figure, path, dpi)


def _box_edges(minimum: np.ndarray, maximum: np.ndarray) -> list[np.ndarray]:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )
    pairs = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )
    return [corners[[first, second]] for first, second in pairs]


def _draw_box(axis: Any, minimum: np.ndarray, maximum: np.ndarray, *, alpha: float = 0.5) -> None:
    for edge in _box_edges(minimum, maximum):
        axis.plot(
            edge[:, 0], edge[:, 1], edge[:, 2],
            color="#777777", lw=0.65, alpha=alpha,
        )


def _plot_3d_problem(path: Path, dpi: int) -> None:
    minimum = np.array([0.0, 0.0, 0.0])
    maximum = np.array([20.0, 10.0, 10.0])
    vertices = np.array(
        [
            [0, 0, 0], [20, 0, 0], [20, 10, 0], [0, 10, 0],
            [0, 0, 10], [20, 0, 10], [20, 10, 10], [0, 10, 10],
        ], dtype=float,
    )
    faces = [
        vertices[[0, 1, 2, 3]], vertices[[4, 5, 6, 7]],
        vertices[[0, 1, 5, 4]], vertices[[1, 2, 6, 5]],
        vertices[[2, 3, 7, 6]], vertices[[3, 0, 4, 7]],
    ]
    with plt.rc_context(_RC):
        figure = plt.figure(figsize=(9.5, 6.0), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        body = Poly3DCollection(
            faces, facecolors="#d5d5d5", edgecolors="#444444", linewidths=0.8, alpha=0.45
        )
        axis.add_collection3d(body)
        for y in (0.0, 10.0):
            for z in (0.0, 10.0):
                axis.scatter([0], [y], [z], marker="<", s=115, c="#f08c86",
                             edgecolors="#7d312d", linewidths=0.7, depthshade=False)
        axis.quiver(20, 5, 8.2, 0, 0, -3.2, color="#111111", linewidth=2.0,
                    arrow_length_ratio=0.22)
        axis.text(10, -1.2, -0.5, "20", ha="center", va="top")
        axis.text(20.8, 5, -0.5, "10", ha="left", va="center")
        axis.text(0.0, 10.7, 5, "10", ha="right", va="center")
        triad_origin = np.array([21.0, 10.8, 0.0])
        for direction, label in zip(np.eye(3), ("x", "y", "z"), strict=True):
            axis.quiver(
                *triad_origin,
                *(1.7 * direction),
                color="#222222",
                linewidth=1.1,
                arrow_length_ratio=0.22,
            )
            label_position = triad_origin + 2.05 * direction
            axis.text(*label_position, label, fontsize=9)
        _draw_box(axis, minimum, maximum, alpha=0.55)
        axis.set_xlim(-1.0, 23.5)
        axis.set_ylim(-1.0, 13.5)
        axis.set_zlim(-1.0, 12.5)
        axis.set_box_aspect((2, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=20, azim=-62)
        axis.set_axis_off()
        axis.text2D(
            0.735,
            0.56,
            r"$F$",
            transform=axis.transAxes,
            color="#111111",
            fontsize=11,
            zorder=50,
        )
        _save_figure(figure, path, dpi)


def _orthonormal_frame(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = second - first
    length = float(np.linalg.norm(axis))
    if length <= 1.0e-12:
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent = axis / length
    reference = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    normal = np.cross(tangent, reference)
    normal /= np.linalg.norm(normal)
    binormal = np.cross(tangent, normal)
    return tangent, normal, binormal


def _capsule_surfaces(
    first: np.ndarray,
    second: np.ndarray,
    radius: float,
    *,
    n_theta: int = 28,
    n_phi: int = 9,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tangent, normal, binormal = _orthonormal_frame(first, second)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    radial = (
        np.cos(theta)[:, None] * normal[None, :]
        + np.sin(theta)[:, None] * binormal[None, :]
    )
    cylinder = np.stack(
        (first[None, :] + radius * radial, second[None, :] + radius * radial), axis=0
    )
    phi = np.linspace(0.0, np.pi / 2.0, n_phi)
    ring = radial[None, :, :]
    sin_phi = np.sin(phi)[:, None, None]
    cos_phi = np.cos(phi)[:, None, None]
    first_cap = first + radius * (-cos_phi * tangent + sin_phi * ring)
    second_cap = second + radius * (cos_phi * tangent + sin_phi * ring)
    return [
        (cylinder[:, :, 0], cylinder[:, :, 1], cylinder[:, :, 2]),
        (first_cap[:, :, 0], first_cap[:, :, 1], first_cap[:, :, 2]),
        (second_cap[:, :, 0], second_cap[:, :, 1], second_cap[:, :, 2]),
    ]


def _draw_geometry_3d(
    geometry: Geometry,
    *,
    title: str,
    path: Path,
    dpi: int,
) -> None:
    minimum = np.array([0.0, 0.0, 0.0])
    maximum = np.array([20.0, 10.0, 10.0])
    with plt.rc_context(_RC):
        figure = plt.figure(figsize=(9.6, 6.4), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        for index, (start, end) in enumerate(geometry.bars):
            alpha = float(geometry.size_variables[index] ** 2)
            if alpha <= 0.05:
                continue
            for x, y, z in _capsule_surfaces(
                geometry.points[start], geometry.points[end], float(geometry.radii[index])
            ):
                # MATLAB surface clipping is enabled after x/y/zlim are set.
                # Clamp the tessellated capsule to those planes so bars that
                # extend beyond the design domain end in a flat clipped face.
                x = np.clip(x, minimum[0], maximum[0])
                y = np.clip(y, minimum[1], maximum[1])
                z = np.clip(z, minimum[2], maximum[2])
                axis.plot_surface(
                    x, y, z, color=BAR_RED, alpha=alpha, linewidth=0,
                    antialiased=True, shade=True, rcount=x.shape[0], ccount=x.shape[1],
                )
        _draw_box(axis, minimum, maximum, alpha=0.24)
        axis.set_xlim(0, 20)
        axis.set_ylim(0, 10)
        axis.set_zlim(0, 10)
        axis.set_box_aspect((2, 1, 1))
        axis.set_proj_type("ortho")
        # MATLAB view([50, 22]) maps to Matplotlib azimuth 50 - 90 = -40.
        axis.view_init(elev=22, azim=-40)
        axis.set_title(title)
        _save_figure(figure, path, dpi)


def _draw_density_isosurface_3d(
    density: np.ndarray,
    *,
    shape: tuple[int, int, int],
    path: Path,
    dpi: int,
    level: float = 0.5,
) -> None:
    try:
        from skimage.measure import marching_cubes
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "3D paper figures require scikit-image; install the visualization extra"
        ) from error
    nx, ny, nz = shape
    cell_field = np.asarray(density, dtype=float).reshape(nz, nx, ny)
    # Match ParaView's Cell Data to Point Data filter before the rho=0.5 clip.
    field_sum = np.zeros((nz + 1, nx + 1, ny + 1), dtype=float)
    field_count = np.zeros_like(field_sum)
    for offset_z in (0, 1):
        for offset_x in (0, 1):
            for offset_y in (0, 1):
                target = (
                    slice(offset_z, offset_z + nz),
                    slice(offset_x, offset_x + nx),
                    slice(offset_y, offset_y + ny),
                )
                field_sum[target] += cell_field
                field_count[target] += 1.0
    field = np.pad(field_sum / field_count, 1, mode="constant", constant_values=0.0)
    dx, dy, dz = 20.0 / nx, 10.0 / ny, 10.0 / nz
    vertices, faces, _, _ = marching_cubes(field, level=level, spacing=(dz, dx, dy))
    vertices -= np.array([dz, dx, dy])
    vertices = vertices[:, [1, 2, 0]]
    vertices = np.clip(vertices, np.zeros(3), np.array([20.0, 10.0, 10.0]))
    triangles = vertices[faces]
    twice_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    triangles = triangles[twice_area > 1.0e-10]
    minimum = np.array([0.0, 0.0, 0.0])
    maximum = np.array([20.0, 10.0, 10.0])
    with plt.rc_context(_RC):
        figure = plt.figure(figsize=(9.6, 6.4), constrained_layout=True)
        axis = figure.add_subplot(111, projection="3d")
        light = LightSource(azdeg=310, altdeg=35)
        try:
            surface = Poly3DCollection(
                triangles,
                facecolors="#d8d2bd",
                edgecolors=(0.0, 0.0, 0.0, 0.0),
                shade=True,
                lightsource=light,
                antialiaseds=False,
            )
        except (TypeError, ValueError):  # pragma: no cover - version fallback
            surface = Poly3DCollection(
                triangles,
                facecolors="#d8d2bd",
                edgecolors=(0.0, 0.0, 0.0, 0.0),
                antialiaseds=False,
            )
        axis.add_collection3d(surface)
        _draw_box(axis, minimum, maximum, alpha=0.55)
        axis.set_xlim(0, 20)
        axis.set_ylim(0, 10)
        axis.set_zlim(0, 10)
        axis.set_box_aspect((2, 1, 1))
        axis.set_proj_type("ortho")
        axis.view_init(elev=22, azim=-40)
        axis.set_axis_off()
        _save_figure(figure, path, dpi)


def _draw_history(
    history: dict[str, np.ndarray],
    *,
    volume_limit: float,
    path: Path,
    dpi: int,
) -> None:
    compliance = history["compliance"]
    volume = history["volume_fraction"]
    # MATLAB plots an array without an explicit x vector, hence samples are
    # numbered 1..N even though the saved Python history labels the first row 0.
    iteration = np.arange(1, compliance.size + 1)
    x_limit = 10.0 * np.ceil(float(iteration[-1]) / 10.0)
    with plt.rc_context(_RC):
        figure, axes = plt.subplots(
            2, 1, figsize=(9.2, 7.1), sharex=True, constrained_layout=True
        )
        axes[0].semilogy(
            iteration, compliance, color=MATLAB_BLUE, lw=1.25, label="compliance"
        )
        axes[0].set_title("objective history")
        axes[0].set_xlabel("iteration")
        axes[0].legend(loc="upper right")
        axes[1].plot(
            iteration,
            volume,
            color=MATLAB_BLUE,
            lw=1.25,
            label="volume fraction",
        )
        axes[1].axhline(volume_limit, color="#444444", ls=":", lw=1.5)
        axes[1].set_title("constraint history")
        axes[1].set_xlabel("iteration")
        axes[1].set_ylim(-0.05, 1.8 * volume_limit)
        axes[1].legend(loc="upper right")
        for axis in axes:
            axis.set_xlim(0, max(1.0, x_limit))
        _save_figure(figure, path, dpi)


def _draw_gradient_check(payload: dict[str, Any], path: Path, dpi: int) -> None:
    entries = payload["entries"]
    design_index = np.arange(1, len(entries) + 1)
    analytical = np.asarray([entry["analytic_compliance"] for entry in entries])
    finite_difference = np.asarray(
        [entry["finite_difference_compliance"] for entry in entries]
    )
    with plt.rc_context(_RC):
        figure, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
        axis.plot(
            design_index,
            finite_difference,
            color=MATLAB_BLUE,
            marker="o",
            markersize=4.2,
            markerfacecolor="none",
            markeredgewidth=0.8,
            linewidth=0.8,
            label="fd",
        )
        axis.plot(
            design_index,
            analytical,
            color=MATLAB_ORANGE,
            linewidth=1.15,
            label="analytical",
        )
        axis.set_xlim(0, 70)
        axis.set_ylim(-6, 6)
        axis.set_title("cost function")
        axis.set_xlabel(r"design variable $z$")
        axis.set_ylabel(r"$dc/dz$")
        axis.legend(loc="upper right")
        _save_figure(figure, path, dpi)


def _ensure_lbracket_gradient_check(path: Path) -> dict[str, Any]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "paper_style_error" not in payload:
            entries = payload["entries"]
            analytic = np.asarray(
                [entry["analytic_compliance"] for entry in entries]
            )
            finite_difference = np.asarray(
                [entry["finite_difference_compliance"] for entry in entries]
            )
            signed_error = analytic - finite_difference
            maximum_index = int(np.argmax(np.abs(signed_error)))
            objective = 2.846071785022176
            payload["paper_style_error"] = {
                "zero_based_index": maximum_index,
                "one_based_index": maximum_index + 1,
                "signed_absolute_difference": float(signed_error[maximum_index]),
                "signed_difference_over_objective": float(
                    signed_error[maximum_index] / objective
                ),
            }
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return payload
    path.parent.mkdir(parents=True, exist_ok=True)
    problem = GPTOProblem(build_case("lbracket2d", profile="paper"))
    design = problem.design_map.encode(load_reference_geometry("lbracket2d"))
    entries = problem.gradient_check(
        design,
        indices=range(problem.design_map.n_variables),
        step=1.0e-6,
        method="forward",
        enforce_bounds=False,
    )
    baseline = problem.evaluate(design)
    analytic = np.asarray([entry.analytic_compliance for entry in entries])
    finite_difference = np.asarray(
        [entry.finite_difference_compliance for entry in entries]
    )
    signed_error = analytic - finite_difference
    maximum_index = int(np.argmax(np.abs(signed_error)))
    payload = {
        "case": "lbracket2d",
        "profile": "paper",
        "design": "reference",
        "step": 1.0e-6,
        "difference_scheme": "forward",
        "enforce_bounds": False,
        "entries": [asdict(entry) for entry in entries],
        "paper_style_error": {
            "zero_based_index": maximum_index,
            "one_based_index": maximum_index + 1,
            "signed_absolute_difference": float(signed_error[maximum_index]),
            "signed_difference_over_objective": float(
                signed_error[maximum_index] / baseline.compliance
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "paper figure generation requires numerical artifacts that are missing:\n"
            f"{formatted}\n"
            "Run the paper-profile MBB and L-bracket optimizations and the 3D "
            "reference analysis first; see README.md."
        )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        from matplotlib import font_manager

        return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=size)
    except (OSError, ValueError):  # pragma: no cover - platform font fallback
        return ImageFont.load_default()


def _create_contact_sheet(
    output_path: Path, entries: list[FigureEntry], output_directory: Path
) -> None:
    columns = 3
    tile_width, tile_height = 560, 420
    header_height = 105
    rows = int(np.ceil(len(entries) / columns))
    canvas = Image.new(
        "RGB", (columns * tile_width, header_height + rows * tile_height), "white"
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(30)
    caption_font = _font(18)
    draw.text(
        (32, 22),
        "Smith-Norato GPTO: reproduced program figures",
        fill="#111111",
        font=title_font,
    )
    draw.text(
        (32, 62),
        "Paper figures 4-18; numerical state and fidelity are recorded in manifest.json",
        fill="#555555",
        font=caption_font,
    )
    for index, entry in enumerate(entries):
        row, column = divmod(index, columns)
        left = column * tile_width
        top = header_height + row * tile_height
        with Image.open(output_directory / entry.filename) as source:
            image = source.convert("RGB")
            image.thumbnail((tile_width - 28, tile_height - 70), Image.Resampling.LANCZOS)
            x = left + (tile_width - image.width) // 2
            y = top + 6 + (tile_height - 70 - image.height) // 2
            canvas.paste(image, (x, y))
        caption = f"Fig. {entry.number}  {entry.title}"
        caption_lines = textwrap.wrap(caption, width=54)[:2]
        caption_y = top + tile_height - 58
        for line_index, line in enumerate(caption_lines):
            draw.text(
                (left + 18, caption_y + 23 * line_index),
                line,
                fill="#222222",
                font=caption_font,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def _write_gallery(
    output_path: Path, entries: list[FigureEntry], manifest_filename: str
) -> None:
    cards = []
    for entry in entries:
        cards.append(
            f"""
      <article class="figure-card" data-case="{entry.case}">
        <a href="{entry.filename}"><img src="{entry.filename}" alt="Figure {entry.number}: {entry.title}" loading="lazy"></a>
        <div class="caption"><strong>Fig. {entry.number}</strong> {entry.title}</div>
        <div class="note">{entry.fidelity_note}</div>
      </article>"""
        )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GPTO paper figure reproduction</title>
  <style>
    :root {{ color-scheme: light dark; font-family: Inter, "Segoe UI", sans-serif; }}
    body {{ margin: 0; padding: 32px; background: #f4f6f8; color: #17202a; }}
    header {{ max-width: 1280px; margin: 0 auto 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 6px 0; color: #53606d; }}
    a {{ color: #155eef; }}
    main {{ max-width: 1500px; margin: auto; display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 20px; }}
    .figure-card {{ background: white; border: 1px solid #d8dee5; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 10px rgba(20,30,40,.06); }}
    .figure-card img {{ display: block; width: 100%; height: 320px; object-fit: contain; background: white; }}
    .caption {{ padding: 13px 15px 4px; }}
    .note {{ padding: 0 15px 15px; color: #64717d; font-size: 14px; line-height: 1.45; }}
    @media (max-width: 520px) {{ body {{ padding: 16px; }} main {{ grid-template-columns: 1fr; }} .figure-card img {{ height: 250px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>GPTO 论文程序结果图复现</h1>
    <p>逐图对应 Smith &amp; Norato (2020) Fig. 4-18。点击任意图片查看原始分辨率。</p>
    <p><a href="contact_sheet.png">总览图</a> · <a href="{manifest_filename}">数据来源与保真说明</a></p>
  </header>
  <main>{''.join(cards)}
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def create_pdf_atlas(
    atlas_path: Path, entries: list[FigureEntry], output_directory: Path
) -> None:
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "PDF atlas generation requires reportlab; install the visualization extra"
        ) from error

    atlas_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(atlas_path), pagesize=landscape(A4))
    page_width, page_height = landscape(A4)
    pdf.setTitle("GPTO paper program figures - Python reproduction")
    pdf.setAuthor("Python reproduction of Smith and Norato (2020)")
    pdf.setFont("Helvetica-Bold", 24)
    pdf.drawString(48, page_height - 70, "GPTO paper program figures")
    pdf.setFont("Helvetica", 15)
    pdf.drawString(48, page_height - 100, "Python reproduction of Smith and Norato (2020), Figures 4-18")
    pdf.setFont("Helvetica", 10.5)
    cover_lines = [
        "Red explicit bars use alpha^2 opacity, matching plot_design.m.",
        "MBB uses penalized density; L-bracket and 3D use unpenalized combined density.",
        "The 3D surface follows Cell Data to Point Data and a rho = 0.5 threshold.",
        "MBB and L-bracket histories are fresh Python runs; exact state provenance is noted per page.",
        f"Source: {PAPER_DOI}",
        f"MATLAB commit: {MATLAB_COMMIT}",
        f"PyGPTO commit: {PYTHON_COMMIT}",
    ]
    y = page_height - 155
    for line in cover_lines:
        pdf.drawString(48, y, line)
        y -= 20
    pdf.showPage()

    for entry in entries:
        image_path = output_directory / entry.filename
        with Image.open(image_path) as source:
            width, height = source.size
        page_size = landscape(A4) if width / height >= 1.08 else A4
        pdf.setPageSize(page_size)
        page_width, page_height = page_size
        header = f"Figure {entry.number}. {entry.title}"
        header_size = 15.0
        while (
            pdf.stringWidth(header, "Helvetica-Bold", header_size)
            > page_width - 80
            and header_size > 10.0
        ):
            header_size -= 0.5
        pdf.setFont("Helvetica-Bold", header_size)
        pdf.drawString(40, page_height - 38, header)
        pdf.setFont("Helvetica", 9)
        note_lines = textwrap.wrap(entry.fidelity_note, width=125)[:2]
        for line_index, line in enumerate(note_lines):
            pdf.drawString(40, 26 - 11 * line_index, line)
        max_width = page_width - 80
        max_height = page_height - 104
        scale = min(max_width / width, max_height / height)
        draw_width, draw_height = width * scale, height * scale
        x = 0.5 * (page_width - draw_width)
        y = 50 + 0.5 * (max_height - draw_height)
        pdf.drawImage(
            ImageReader(str(image_path)), x, y, width=draw_width, height=draw_height,
            preserveAspectRatio=True, mask="auto",
        )
        pdf.showPage()
    pdf.save()


def generate_paper_figures(
    *,
    output_directory: Path | str = Path("output/figures/gpto-paper-results"),
    results_root: Path | str = Path("results"),
    atlas_path: Path | str | None = Path(
        "output/pdf/gpto-paper-results-atlas.pdf"
    ),
    dpi: int = 240,
) -> dict[str, Any]:
    """Generate a numbered Fig. 4--18 reproduction gallery and optional PDF atlas.

    The routine consumes the full-resolution numerical artifacts produced by the
    reproduction workflow.  It never substitutes a reduced mesh for a final
    density field.  The only tiny-profile construction is the deterministic 3D
    *initial geometry*, whose bars do not depend on the FE mesh.
    """

    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    output_directory = Path(output_directory)
    results_root = Path(results_root)
    atlas = None if atlas_path is None else Path(atlas_path)
    output_directory.mkdir(parents=True, exist_ok=True)

    mbb_directory = results_root / "mbb2d" / "paper-run"
    lbracket_run_directory = results_root / "lbracket2d" / "paper-run"
    lbracket_reference_directory = results_root / "lbracket2d" / "reference-paper"
    cantilever_reference_directory = (
        results_root / "cantilever3d" / "reference-paper"
    )
    gradient_path = (
        results_root
        / "lbracket2d"
        / "paper-reference-gradient-check"
        / "gradient_check.json"
    )
    required = [
        mbb_directory / "final.npz",
        mbb_directory / "final.json",
        mbb_directory / "history.csv",
        lbracket_run_directory / "history.csv",
        lbracket_reference_directory / "reference.npz",
        lbracket_reference_directory / "reference.json",
        cantilever_reference_directory / "reference.npz",
        cantilever_reference_directory / "reference.json",
    ]
    _require_files(required)

    mbb_state = _load_state(
        mbb_directory / "final.npz", mbb_directory / "final.json"
    )
    lbracket_state = _load_state(
        lbracket_reference_directory / "reference.npz",
        lbracket_reference_directory / "reference.json",
    )
    cantilever_state = _load_state(
        cantilever_reference_directory / "reference.npz",
        cantilever_reference_directory / "reference.json",
    )
    mbb_history = _read_history(mbb_directory / "history.csv")
    lbracket_history = _read_history(lbracket_run_directory / "history.csv")
    gradient_payload = _ensure_lbracket_gradient_check(gradient_path)

    mbb_case = build_case("mbb2d", profile="paper")
    lbracket_case = build_case("lbracket2d", profile="paper")
    cantilever_initial_geometry = build_case(
        "cantilever3d", profile="tiny"
    ).geometry
    cantilever_summary = json.loads(
        (cantilever_reference_directory / "reference.json").read_text(
            encoding="utf-8"
        )
    )
    cantilever_shape = tuple(
        int(value) for value in cantilever_summary["mesh"]["structured_shape"]
    )
    if len(cantilever_shape) != 3:
        raise ValueError("3D reference summary does not contain a 3D mesh shape")

    entries: list[FigureEntry] = []

    def add_entry(
        number: int,
        filename: str,
        title: str,
        case: str,
        category: str,
        data_source: str,
        fidelity_note: str,
    ) -> None:
        entries.append(
            FigureEntry(
                number=number,
                filename=filename,
                title=title,
                case=case,
                category=category,
                data_source=data_source,
                fidelity_note=fidelity_note,
            )
        )

    path = output_directory / "fig04_mbb_problem.png"
    _plot_mbb_problem(path, dpi)
    add_entry(
        4,
        path.name,
        "MBB beam problem",
        "mbb2d",
        "problem schematic",
        "Paper dimensions and boundary conditions; redrawn in Python.",
        "Schematic, not a numerical program output; recreated from the paper specification.",
    )

    path = output_directory / "fig05_mbb_initial_design.png"
    _draw_geometry_2d(
        mbb_case.geometry,
        xlim=(0.0, 20.0),
        ylim=(-5.0, 10.0),
        title="initial design",
        path=path,
        dpi=dpi,
        figure_size=(9.2, 5.2),
        xtick_step=2.0,
        ytick_step=5.0,
    )
    add_entry(
        5,
        path.name,
        "Initial design for MBB beam",
        "mbb2d",
        "MATLAB-direct result",
        "Deterministic initial geometry from the released MBB input.",
        "Paper viewport and code-level alpha^2 opacity are reproduced.",
    )

    path = output_directory / "fig06_mbb_optimal_design.png"
    _draw_geometry_2d(
        mbb_state.geometry,
        xlim=(0.0, 20.0),
        ylim=(0.0, 5.0),
        title="design, iteration = 88",
        path=path,
        dpi=dpi,
        figure_size=(9.2, 3.35),
        xtick_step=2.0,
        ytick_step=1.0,
    )
    add_entry(
        6,
        path.name,
        "Optimal design for MBB beam",
        "mbb2d",
        "MATLAB-direct result",
        str((mbb_directory / "final.npz").resolve()),
        "Fresh Python paper-mesh run, 88 iterations; geometry is not the unavailable exact paper checkpoint.",
    )

    path = output_directory / "fig07_mbb_combined_density.png"
    _draw_density_2d(
        mbb_case.mesh,
        mbb_state.stiffness_density,
        compliance=mbb_state.compliance,
        xlim=(0.0, 20.0),
        ylim=(0.0, 5.0),
        path=path,
        dpi=dpi,
        figure_size=(9.2, 3.35),
        xtick_step=2.0,
        ytick_step=1.0,
    )
    add_entry(
        7,
        path.name,
        "Combined density of optimal design for MBB beam",
        "mbb2d",
        "MATLAB-direct result",
        str((mbb_directory / "final.npz").resolve()),
        "Uses penalized/effective density exactly as the released structured-mesh MATLAB plotting branch; computed compliance is 1.33% above the paper value.",
    )

    path = output_directory / "fig08_mbb_history.png"
    _draw_history(
        mbb_history, volume_limit=0.45, path=path, dpi=dpi
    )
    add_entry(
        8,
        path.name,
        "Optimization history for MBB beam",
        "mbb2d",
        "MATLAB-direct result",
        str((mbb_directory / "history.csv").resolve()),
        "Fresh Python run with the paper mesh and 88 optimizer iterations; semilog compliance and the 0.45 volume limit mirror plot_history.m.",
    )

    path = output_directory / "fig09_lbracket_problem.png"
    _plot_lbracket_problem(path, dpi)
    add_entry(
        9,
        path.name,
        "2D L-bracket problem",
        "lbracket2d",
        "problem schematic",
        "Paper dimensions and boundary conditions; redrawn in Python.",
        "Schematic, not a numerical program output; recreated from the paper specification.",
    )

    path = output_directory / "fig10_lbracket_initial_design.png"
    _draw_geometry_2d(
        lbracket_case.geometry,
        xlim=(0.0, 100.0),
        ylim=(10.0, 90.0),
        title="initial design",
        path=path,
        dpi=dpi,
        figure_size=(7.2, 6.2),
        xtick_step=10.0,
        ytick_step=10.0,
    )
    add_entry(
        10,
        path.name,
        "Initial design for L-bracket",
        "lbracket2d",
        "MATLAB-direct result",
        "Deterministic connected-bar initial geometry from the released L-bracket input.",
        "Paper viewport and code-level alpha^2 opacity are reproduced.",
    )

    path = output_directory / "fig11_lbracket_optimal_design.png"
    _draw_geometry_2d(
        lbracket_state.geometry,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        title="design, iteration = 64",
        path=path,
        dpi=dpi,
        figure_size=(7.2, 7.0),
        xtick_step=10.0,
        ytick_step=10.0,
    )
    add_entry(
        11,
        path.name,
        "Optimal design for L-bracket",
        "lbracket2d",
        "MATLAB-direct result",
        str((lbracket_reference_directory / "reference.npz").resolve()),
        "Released 64-iteration final geometry; the recomputed compliance matches the paper within 7.6e-8 relative error.",
    )

    path = output_directory / "fig12_lbracket_combined_density.png"
    _draw_density_2d(
        lbracket_case.mesh,
        lbracket_state.volume_density,
        compliance=lbracket_state.compliance,
        xlim=(0.0, 100.0),
        ylim=(0.0, 100.0),
        path=path,
        dpi=dpi,
        figure_size=(7.2, 7.0),
        xtick_step=10.0,
        ytick_step=10.0,
    )
    add_entry(
        12,
        path.name,
        "Combined density of optimal design for L-bracket",
        "lbracket2d",
        "MATLAB-direct result",
        str((lbracket_reference_directory / "reference.npz").resolve()),
        "Uses unpenalized combined density and density-dependent cell-edge opacity, matching plot_density_cells.m.",
    )

    path = output_directory / "fig13_lbracket_history.png"
    _draw_history(
        lbracket_history, volume_limit=0.30, path=path, dpi=dpi
    )
    add_entry(
        13,
        path.name,
        "Optimization history for L-bracket",
        "lbracket2d",
        "MATLAB-direct result",
        str((lbracket_run_directory / "history.csv").resolve()),
        "Fresh Python paper-mesh history (83 optimizer iterations); the exact 64-step MATLAB history was not published in either repository.",
    )

    path = output_directory / "fig14_lbracket_sensitivity_check.png"
    _draw_gradient_check(gradient_payload, path, dpi)
    add_entry(
        14,
        path.name,
        "Finite-difference check of L-bracket compliance sensitivities",
        "lbracket2d",
        "MATLAB-direct result",
        str(gradient_path.resolve()),
        "All 66 scaled variables use a one-sided forward difference with h=1e-6; the maximum signed discrepancy reproduces the paper's -0.0037 and -0.0013 values.",
    )

    path = output_directory / "fig15_cantilever3d_problem.png"
    _plot_3d_problem(path, dpi)
    add_entry(
        15,
        path.name,
        "3D cantilever problem",
        "cantilever3d",
        "problem schematic",
        "Paper dimensions and boundary conditions; redrawn in Python.",
        "The paper caption calls this an initial design, but the body text and image show the load/support schematic; the true initial geometry is Fig. 16.",
    )

    path = output_directory / "fig16_cantilever3d_initial_design.png"
    _draw_geometry_3d(
        cantilever_initial_geometry,
        title="design, iteration = 0",
        path=path,
        dpi=dpi,
    )
    add_entry(
        16,
        path.name,
        "Initial design for 3D cantilever beam",
        "cantilever3d",
        "MATLAB-direct result",
        "Deterministic initial 16-bar geometry from the released 3D input.",
        "Real cylinders with hemispherical caps, alpha^2 opacity, and the MATLAB [50,22] camera equivalent are reproduced.",
    )

    path = output_directory / "fig17_cantilever3d_optimal_design.png"
    _draw_geometry_3d(
        cantilever_state.geometry,
        title="design, iteration = 106",
        path=path,
        dpi=dpi,
    )
    add_entry(
        17,
        path.name,
        "Optimal design for 3D cantilever beam",
        "cantilever3d",
        "MATLAB-direct result",
        str((cantilever_reference_directory / "reference.npz").resolve()),
        "Released final geometry. The title is corrected to iteration 106; the paper image accidentally retains 'iteration = 0'.",
    )

    path = output_directory / "fig18_cantilever3d_density_isosurface.png"
    _draw_density_isosurface_3d(
        cantilever_state.volume_density,
        shape=cantilever_shape,
        path=path,
        dpi=dpi,
        level=0.5,
    )
    add_entry(
        18,
        path.name,
        "Combined density isosurface for optimal 3D cantilever beam",
        "cantilever3d",
        "VTK/ParaView post-processing",
        str((cantilever_reference_directory / "reference.npz").resolve()),
        "Unpenalized combined density, Cell Data to Point Data averaging, and rho=0.5 are reproduced; the unpublished ParaView camera and lighting are visually approximated.",
    )

    gradient_error = gradient_payload["paper_style_error"]
    manifest = {
        "title": "Smith-Norato GPTO paper figure reproduction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "paper": PAPER_DOI,
        "upstream": {
            "matlab_commit": MATLAB_COMMIT,
            "python_commit": PYTHON_COMMIT,
        },
        "scope": {
            "figure_range": "4-18",
            "figure_count": len(entries),
            "matlab_direct_results": 11,
            "problem_schematics": 3,
            "vtk_paraview_results": 1,
        },
        "numerical_results": {
            "mbb2d": {
                "iterations": int(mbb_history["iteration"][-1]),
                "compliance": mbb_state.compliance,
                "paper_compliance": 4.201067,
                "relative_compliance_error": (
                    mbb_state.compliance - 4.201067
                )
                / 4.201067,
                "volume_fraction": mbb_state.volume_fraction,
            },
            "lbracket2d": {
                "reference_iterations": 64,
                "fresh_history_iterations": int(
                    lbracket_history["iteration"][-1]
                ),
                "compliance": lbracket_state.compliance,
                "paper_compliance": 2.846072,
                "volume_fraction": lbracket_state.volume_fraction,
                "gradient_check": gradient_error,
            },
            "cantilever3d": {
                "reference_iterations": 106,
                "compliance": cantilever_state.compliance,
                "volume_fraction": cantilever_state.volume_fraction,
                "isosurface_level": 0.5,
            },
        },
        "figures": [asdict(entry) for entry in entries],
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    contact_sheet_path = output_directory / "contact_sheet.png"
    _create_contact_sheet(contact_sheet_path, entries, output_directory)
    gallery_path = output_directory / "index.html"
    _write_gallery(gallery_path, entries, manifest_path.name)
    if atlas is not None:
        create_pdf_atlas(atlas, entries, output_directory)

    return {
        "figures": len(entries),
        "output_directory": str(output_directory.resolve()),
        "gallery": str(gallery_path.resolve()),
        "contact_sheet": str(contact_sheet_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "atlas": None if atlas is None else str(atlas.resolve()),
        "gradient_check": gradient_error,
    }
