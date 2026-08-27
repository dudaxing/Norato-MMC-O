"""Typed configuration and state objects used by the reproduction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(slots=True)
class Mesh:
    """Finite-element mesh with zero-based connectivity."""

    coordinates: FloatArray
    elements: IntArray
    structured_shape: tuple[int, ...] | None = None
    name: str = "mesh"
    centroids: FloatArray = field(init=False)
    element_volumes: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        self.elements = np.asarray(self.elements, dtype=np.int64)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] not in (2, 3):
            raise ValueError("coordinates must have shape (n_node, 2|3)")
        expected_nodes = 2 ** self.dim
        if self.elements.ndim != 2 or self.elements.shape[1] != expected_nodes:
            raise ValueError(
                f"elements must have shape (n_element, {expected_nodes}) for {self.dim}D"
            )
        if self.elements.size and (
            self.elements.min() < 0 or self.elements.max() >= self.n_nodes
        ):
            raise ValueError("element connectivity is outside the node range")
        element_coordinates = self.coordinates[self.elements]
        self.centroids = element_coordinates.mean(axis=1)
        self.element_volumes = _element_measures(element_coordinates)
        if np.any(self.element_volumes <= 0):
            raise ValueError("mesh contains an element with non-positive measure")

    @property
    def dim(self) -> int:
        return int(self.coordinates.shape[1])

    @property
    def n_nodes(self) -> int:
        return int(self.coordinates.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def coordinate_min(self) -> FloatArray:
        return self.coordinates.min(axis=0)

    @property
    def coordinate_max(self) -> FloatArray:
        return self.coordinates.max(axis=0)


def _element_measures(element_coordinates: FloatArray) -> FloatArray:
    dim = element_coordinates.shape[2]
    if dim == 2:
        x = element_coordinates[:, :, 0]
        y = element_coordinates[:, :, 1]
        signed_twice_area = np.sum(
            x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1
        )
        return 0.5 * np.abs(signed_twice_area)

    # All 3D article examples use structured parallelepiped H8 elements.
    edge_x = element_coordinates[:, 1] - element_coordinates[:, 0]
    edge_y = element_coordinates[:, 3] - element_coordinates[:, 0]
    edge_z = element_coordinates[:, 4] - element_coordinates[:, 0]
    return np.abs(np.einsum("ni,ni->n", edge_x, np.cross(edge_y, edge_z)))


@dataclass(slots=True)
class Geometry:
    """Explicit offset-bar geometry.

    `bars[:, 0]` and `bars[:, 1]` contain zero-based endpoint indices.
    """

    points: FloatArray
    bars: IntArray
    size_variables: FloatArray
    radii: FloatArray

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=float)
        self.bars = np.asarray(self.bars, dtype=np.int64)
        self.size_variables = np.asarray(self.size_variables, dtype=float)
        self.radii = np.asarray(self.radii, dtype=float)
        if self.points.ndim != 2 or self.points.shape[1] not in (2, 3):
            raise ValueError("points must have shape (n_point, 2|3)")
        if self.bars.ndim != 2 or self.bars.shape[1] != 2:
            raise ValueError("bars must have shape (n_bar, 2)")
        if self.bars.size and (self.bars.min() < 0 or self.bars.max() >= self.n_points):
            raise ValueError("bar endpoint index is outside the point range")
        if self.size_variables.shape != (self.n_bars,):
            raise ValueError("size_variables must contain one value per bar")
        if self.radii.shape != (self.n_bars,):
            raise ValueError("radii must contain one value per bar")

    @property
    def dim(self) -> int:
        return int(self.points.shape[1])

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_bars(self) -> int:
        return int(self.bars.shape[0])

    def copy(self) -> "Geometry":
        return Geometry(
            self.points.copy(),
            self.bars.copy(),
            self.size_variables.copy(),
            self.radii.copy(),
        )


@dataclass(frozen=True, slots=True)
class Material:
    young_modulus: float = 1.0
    poisson_ratio: float = 0.3
    minimum_density: float = 1.0e-2


@dataclass(frozen=True, slots=True)
class SolverOptions:
    kind: Literal["direct", "cg"] = "direct"
    relative_tolerance: float = 1.0e-5
    max_iterations: int = 10_000
    use_jacobi_preconditioner: bool = True


@dataclass(frozen=True, slots=True)
class OptimizationOptions:
    max_iterations: int = 300
    move_limit: float = 0.1
    design_step_tolerance: float = 1.0e-2
    kkt_tolerance: float = 1.0e-4


@dataclass(frozen=True, slots=True)
class ExpectedResult:
    paper_iterations: int | None = None
    paper_compliance: float | None = None
    volume_fraction: float | None = None
    notes: str = ""


@dataclass(slots=True)
class CaseDefinition:
    name: str
    title: str
    mesh: Mesh
    geometry: Geometry
    fixed_dofs: IntArray
    forces: FloatArray
    radius_bounds: tuple[float, float]
    volume_fraction_limit: float
    material: Material = field(default_factory=Material)
    solver: SolverOptions = field(default_factory=SolverOptions)
    optimization: OptimizationOptions = field(default_factory=OptimizationOptions)
    expected: ExpectedResult = field(default_factory=ExpectedResult)
    profile: str = "paper"
    reference_design_path: Path | None = None

    def __post_init__(self) -> None:
        self.fixed_dofs = np.unique(np.asarray(self.fixed_dofs, dtype=np.int64))
        self.forces = np.asarray(self.forces, dtype=float).reshape(-1)
        n_dof = self.mesh.dim * self.mesh.n_nodes
        if self.geometry.dim != self.mesh.dim:
            raise ValueError("mesh and geometry dimensions differ")
        if self.forces.shape != (n_dof,):
            raise ValueError(f"forces must have shape ({n_dof},)")
        if self.fixed_dofs.size and (
            self.fixed_dofs.min() < 0 or self.fixed_dofs.max() >= n_dof
        ):
            raise ValueError("fixed dof index is outside the global dof range")
        if not (0 < self.volume_fraction_limit <= 1):
            raise ValueError("volume fraction limit must be in (0, 1]")
        if self.radius_bounds[0] > self.radius_bounds[1]:
            raise ValueError("radius lower bound exceeds upper bound")

    def with_geometry(self, geometry: Geometry) -> "CaseDefinition":
        return replace(self, geometry=geometry)
