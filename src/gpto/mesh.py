"""Mesh generation and loading utilities."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import Mesh


def structured_mesh(
    dimensions: Sequence[float],
    elements_per_side: Sequence[int],
    *,
    name: str = "structured",
) -> Mesh:
    """Generate the Q4/H8 mesh ordering used by the reference codes."""

    dimensions_array = np.asarray(dimensions, dtype=float)
    counts = np.asarray(elements_per_side, dtype=np.int64)
    if dimensions_array.ndim != 1 or dimensions_array.size not in (2, 3):
        raise ValueError("dimensions must contain two or three lengths")
    if counts.shape != dimensions_array.shape:
        raise ValueError("elements_per_side must match dimensions")
    if np.any(dimensions_array <= 0) or np.any(counts <= 0):
        raise ValueError("mesh dimensions and element counts must be positive")

    axes = [
        np.linspace(0.0, length, int(count) + 1)
        for length, count in zip(dimensions_array, counts, strict=True)
    ]
    if len(axes) == 2:
        x, y = axes
        xx, yy = np.meshgrid(x, y, indexing="ij")
        coordinates = np.column_stack((xx.ravel(), yy.ravel()))
        nx, ny = map(int, counts)
        ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        n1 = ix * (ny + 1) + iy
        n2 = (ix + 1) * (ny + 1) + iy
        elements = np.column_stack(
            (
                n1.ravel(),
                n2.ravel(),
                (n2 + 1).ravel(),
                (n1 + 1).ravel(),
            )
        )
    else:
        x, y, z = axes
        # Reference ordering: y varies fastest, followed by x, then z.
        zz, xx, yy = np.meshgrid(z, x, y, indexing="ij")
        coordinates = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        nx, ny, nz = map(int, counts)
        iz, ix, iy = np.meshgrid(
            np.arange(nz), np.arange(nx), np.arange(ny), indexing="ij"
        )
        layer = (nx + 1) * (ny + 1)
        n1 = iz * layer + ix * (ny + 1) + iy
        n2 = iz * layer + (ix + 1) * (ny + 1) + iy
        n3 = n2 + 1
        n4 = n1 + 1
        n5 = n1 + layer
        n6 = n2 + layer
        n7 = n6 + 1
        n8 = n5 + 1
        elements = np.column_stack(
            tuple(item.ravel() for item in (n1, n2, n3, n4, n5, n6, n7, n8))
        )

    return Mesh(
        coordinates=coordinates,
        elements=elements,
        structured_shape=tuple(int(value) for value in counts),
        name=name,
    )


def packaged_mesh(filename: str, *, name: str | None = None) -> Mesh:
    """Load a mesh converted from an official GPTO/PyGPTO input file."""

    resource = resources.files("gpto").joinpath("data", filename)
    with resources.as_file(resource) as path:
        return load_mesh_npz(path, name=name or Path(filename).stem)


def load_mesh_npz(path: str | Path, *, name: str = "mesh") -> Mesh:
    with np.load(path) as archive:
        coordinates = np.asarray(archive["coordinates"], dtype=float)
        elements = np.asarray(archive["elements"], dtype=np.int64)

    if coordinates.shape[1] == 2:
        signed_twice_area = _signed_quad_area(coordinates[elements])
        clockwise = signed_twice_area < 0
        if np.any(clockwise):
            corrected = elements.copy()
            corrected[clockwise] = corrected[clockwise][:, [0, 3, 2, 1]]
            elements = corrected
    return Mesh(coordinates, elements, name=name)


def _signed_quad_area(element_coordinates: np.ndarray) -> np.ndarray:
    x = element_coordinates[:, :, 0]
    y = element_coordinates[:, :, 1]
    return np.sum(
        x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1
    )


def nodes_at(
    mesh: Mesh,
    *,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    tolerance: float | None = None,
) -> np.ndarray:
    """Return node indices matching any specified coordinates."""

    targets = (x, y, z)
    if tolerance is None:
        scale = max(1.0, float(np.max(np.abs(mesh.coordinates))))
        tolerance = 1.0e-9 * scale
    mask = np.ones(mesh.n_nodes, dtype=bool)
    for axis, target in enumerate(targets[: mesh.dim]):
        if target is not None:
            mask &= np.abs(mesh.coordinates[:, axis] - target) <= tolerance
    return np.flatnonzero(mask)


def dof_indices(nodes: np.ndarray, dim: int, components: Sequence[int]) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.int64).reshape(-1)
    components_array = np.asarray(components, dtype=np.int64).reshape(-1)
    if np.any(components_array < 0) or np.any(components_array >= dim):
        raise ValueError("dof component is outside the mesh dimension")
    return (nodes[:, None] * dim + components_array[None, :]).ravel()
