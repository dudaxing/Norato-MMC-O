"""Q4/H8 linear-elastic finite-element analysis for GPTO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from .config import FloatArray, Material, Mesh, SolverOptions


@dataclass(slots=True)
class AnalysisResult:
    displacement: FloatArray
    compliance: float
    element_unit_strain_energy: FloatArray
    solver_info: int
    solver_iterations: int
    relative_residual: float


class FiniteElementModel:
    """Reusable FE model whose stiffness is updated by element densities."""

    def __init__(
        self,
        mesh: Mesh,
        material: Material,
        fixed_dofs: np.ndarray,
        forces: FloatArray,
        solver: SolverOptions,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.solver = solver
        self.n_dof = mesh.dim * mesh.n_nodes
        self.fixed_dofs = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        fixed_mask = np.zeros(self.n_dof, dtype=bool)
        fixed_mask[self.fixed_dofs] = True
        self.free_dofs = np.flatnonzero(~fixed_mask)
        self.forces = np.asarray(forces, dtype=float).reshape(self.n_dof)
        if np.any(self.forces[self.fixed_dofs] != 0.0):
            raise ValueError("a prescribed load acts on a fixed degree of freedom")

        components = np.arange(mesh.dim, dtype=np.int64)
        self.element_dofs = (
            mesh.elements[:, :, None] * mesh.dim + components[None, None, :]
        ).reshape(mesh.n_elements, -1)
        self.n_element_dof = int(self.element_dofs.shape[1])
        self.constitutive = constitutive_matrix(mesh.dim, material)

        element_coordinates = mesh.coordinates[mesh.elements]
        if mesh.structured_shape is not None:
            one = element_stiffness(element_coordinates[:1], self.constitutive)
            self.unit_stiffness: FloatArray = one[0]
            self.uniform_stiffness = True
        else:
            self.unit_stiffness = element_stiffness(
                element_coordinates, self.constitutive
            )
            self.uniform_stiffness = False

        # Assemble only the upper triangle and mirror it. This cuts the large
        # 3D article case's assembly arrays almost in half.
        self._tri_row, self._tri_col = np.triu_indices(self.n_element_dof)
        self._assembly_rows = self.element_dofs[:, self._tri_row].astype(
            np.int32, copy=False
        ).ravel()
        self._assembly_cols = self.element_dofs[:, self._tri_col].astype(
            np.int32, copy=False
        ).ravel()
        self._last_displacement = np.zeros(self.n_dof, dtype=float)

    def assemble(self, stiffness_density: FloatArray) -> sparse.csr_matrix:
        density = np.asarray(stiffness_density, dtype=float).reshape(-1)
        if density.shape != (self.mesh.n_elements,):
            raise ValueError("stiffness_density has the wrong element count")
        if np.any(density <= 0):
            raise ValueError("stiffness densities must be strictly positive")

        if self.uniform_stiffness:
            local_upper = self.unit_stiffness[self._tri_row, self._tri_col]
            values = (density[:, None] * local_upper[None, :]).ravel()
        else:
            local_upper = self.unit_stiffness[:, self._tri_row, self._tri_col]
            values = (density[:, None] * local_upper).ravel()

        upper = sparse.coo_matrix(
            (values, (self._assembly_rows, self._assembly_cols)),
            shape=(self.n_dof, self.n_dof),
        ).tocsr()
        upper.sum_duplicates()
        matrix = upper + upper.T - sparse.diags(upper.diagonal(), format="csr")
        matrix.sum_duplicates()
        return matrix

    def analyze(self, stiffness_density: FloatArray) -> AnalysisResult:
        matrix = self.assemble(stiffness_density)
        reduced = matrix[self.free_dofs][:, self.free_dofs].tocsr()
        rhs = self.forces[self.free_dofs]
        solver_iterations = 0

        if self.solver.kind == "direct":
            free_displacement = sparse_linalg.spsolve(reduced, rhs)
            solver_info = 0
        elif self.solver.kind == "cg":
            diagonal = reduced.diagonal()
            if np.any(diagonal <= 0):
                raise RuntimeError("reduced stiffness matrix has a non-positive diagonal")
            preconditioner = None
            if self.solver.use_jacobi_preconditioner:
                inverse_diagonal = 1.0 / diagonal
                preconditioner = sparse_linalg.LinearOperator(
                    reduced.shape, matvec=lambda vector: inverse_diagonal * vector
                )

            def count_iteration(_: np.ndarray) -> None:
                nonlocal solver_iterations
                solver_iterations += 1

            free_displacement, solver_info = sparse_linalg.cg(
                reduced,
                rhs,
                x0=self._last_displacement[self.free_dofs],
                rtol=self.solver.relative_tolerance,
                atol=0.0,
                maxiter=self.solver.max_iterations,
                M=preconditioner,
                callback=count_iteration,
            )
            if solver_info < 0:
                raise RuntimeError(f"conjugate gradient failed with info={solver_info}")
        else:  # pragma: no cover - Literal guard
            raise ValueError(f"unsupported solver kind {self.solver.kind!r}")

        displacement = np.zeros(self.n_dof, dtype=float)
        displacement[self.free_dofs] = free_displacement
        self._last_displacement = displacement.copy()
        residual = reduced @ free_displacement - rhs
        denominator = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        relative_residual = float(np.linalg.norm(residual) / denominator)
        compliance = float(self.forces @ displacement)
        element_displacement = displacement[self.element_dofs]
        if self.uniform_stiffness:
            element_energy = np.einsum(
                "ei,ij,ej->e",
                element_displacement,
                self.unit_stiffness,
                element_displacement,
                optimize=True,
            )
        else:
            element_energy = np.einsum(
                "ei,eij,ej->e",
                element_displacement,
                self.unit_stiffness,
                element_displacement,
                optimize=True,
            )

        return AnalysisResult(
            displacement=displacement,
            compliance=compliance,
            element_unit_strain_energy=element_energy,
            solver_info=int(solver_info),
            solver_iterations=solver_iterations,
            relative_residual=relative_residual,
        )


def constitutive_matrix(dim: int, material: Material) -> FloatArray:
    """Plane-stress (2D) or isotropic 3D elasticity matrix."""

    young = float(material.young_modulus)
    nu = float(material.poisson_ratio)
    if dim == 2:
        factor = young / (1.0 - nu**2)
        matrix = factor * np.array(
            [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1.0 - nu) / 2.0]]
        )
    elif dim == 3:
        factor = young / ((1.0 + nu) * (1.0 - 2.0 * nu))
        matrix = factor * np.array(
            [
                [1.0 - nu, nu, nu, 0.0, 0.0, 0.0],
                [nu, 1.0 - nu, nu, 0.0, 0.0, 0.0],
                [nu, nu, 1.0 - nu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
            ]
        )
    else:
        raise ValueError("only 2D and 3D elasticity are supported")
    return 0.5 * (matrix + matrix.T)


def element_stiffness(
    element_coordinates: FloatArray, constitutive: FloatArray
) -> FloatArray:
    """Fully integrated Q4 or H8 unit-material stiffness matrices."""

    coordinates = np.asarray(element_coordinates, dtype=float)
    dim = int(coordinates.shape[2])
    n_element = coordinates.shape[0]
    n_node = 2**dim
    n_edof = dim * n_node
    stiffness = np.zeros((n_element, n_edof, n_edof), dtype=float)
    gauss_points = (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0))

    if dim == 2:
        for xi in gauss_points:
            for eta in gauss_points:
                gradient = q4_shape_gradient(xi, eta)
                jacobian = np.einsum(
                    "ai,nib->nab", gradient, coordinates, optimize=True
                )
                determinant = np.linalg.det(jacobian)
                if np.any(determinant <= 0):
                    raise ValueError("Q4 element has a non-positive Jacobian")
                physical_gradient = np.einsum(
                    "nab,bi->nai",
                    np.linalg.inv(jacobian),
                    gradient,
                    optimize=True,
                )
                strain = q4_strain_displacement(physical_gradient)
                stiffness += np.einsum(
                    "nai,ab,nbj,n->nij",
                    strain,
                    constitutive,
                    strain,
                    determinant,
                    optimize=True,
                )
    elif dim == 3:
        for xi in gauss_points:
            for eta in gauss_points:
                for zeta in gauss_points:
                    gradient = h8_shape_gradient(xi, eta, zeta)
                    jacobian = np.einsum(
                        "ai,nib->nab", gradient, coordinates, optimize=True
                    )
                    determinant = np.linalg.det(jacobian)
                    if np.any(determinant <= 0):
                        raise ValueError("H8 element has a non-positive Jacobian")
                    physical_gradient = np.einsum(
                        "nab,bi->nai",
                        np.linalg.inv(jacobian),
                        gradient,
                        optimize=True,
                    )
                    strain = h8_strain_displacement(physical_gradient)
                    stiffness += np.einsum(
                        "nai,ab,nbj,n->nij",
                        strain,
                        constitutive,
                        strain,
                        determinant,
                        optimize=True,
                    )
    else:
        raise ValueError("only Q4 and H8 elements are supported")
    return 0.5 * (stiffness + stiffness.transpose(0, 2, 1))


def q4_shape_gradient(xi: float, eta: float) -> FloatArray:
    return 0.25 * np.array(
        [
            [eta - 1.0, 1.0 - eta, 1.0 + eta, -1.0 - eta],
            [xi - 1.0, -1.0 - xi, 1.0 + xi, 1.0 - xi],
        ]
    )


def h8_shape_gradient(xi: float, eta: float, zeta: float) -> FloatArray:
    return 0.125 * np.array(
        [
            [
                -(1 - zeta) * (1 - eta),
                (1 - zeta) * (1 - eta),
                (1 - zeta) * (1 + eta),
                -(1 - zeta) * (1 + eta),
                -(1 + zeta) * (1 - eta),
                (1 + zeta) * (1 - eta),
                (1 + zeta) * (1 + eta),
                -(1 + zeta) * (1 + eta),
            ],
            [
                -(1 - zeta) * (1 - xi),
                -(1 - zeta) * (1 + xi),
                (1 - zeta) * (1 + xi),
                (1 - zeta) * (1 - xi),
                -(1 + zeta) * (1 - xi),
                -(1 + zeta) * (1 + xi),
                (1 + zeta) * (1 + xi),
                (1 + zeta) * (1 - xi),
            ],
            [
                -(1 - eta) * (1 - xi),
                -(1 - eta) * (1 + xi),
                -(1 + eta) * (1 + xi),
                -(1 + eta) * (1 - xi),
                (1 - eta) * (1 - xi),
                (1 - eta) * (1 + xi),
                (1 + eta) * (1 + xi),
                (1 + eta) * (1 - xi),
            ],
        ]
    )


def q4_strain_displacement(gradient: FloatArray) -> FloatArray:
    n_element = gradient.shape[0]
    strain = np.zeros((n_element, 3, 8), dtype=float)
    strain[:, 0, 0::2] = gradient[:, 0]
    strain[:, 1, 1::2] = gradient[:, 1]
    strain[:, 2, 0::2] = gradient[:, 1]
    strain[:, 2, 1::2] = gradient[:, 0]
    return strain


def h8_strain_displacement(gradient: FloatArray) -> FloatArray:
    n_element = gradient.shape[0]
    strain = np.zeros((n_element, 6, 24), dtype=float)
    strain[:, 0, 0::3] = gradient[:, 0]
    strain[:, 1, 1::3] = gradient[:, 1]
    strain[:, 2, 2::3] = gradient[:, 2]
    strain[:, 3, 0::3] = gradient[:, 1]
    strain[:, 3, 1::3] = gradient[:, 0]
    strain[:, 4, 1::3] = gradient[:, 2]
    strain[:, 4, 2::3] = gradient[:, 1]
    strain[:, 5, 0::3] = gradient[:, 2]
    strain[:, 5, 2::3] = gradient[:, 0]
    return strain
