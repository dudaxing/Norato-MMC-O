"""Persist reproducible numerical outputs and VTK fields."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import ProjectionParameters
from .optimizer import OptimizationResult
from .problem import Evaluation, GPTOProblem


def save_evaluation(
    output_directory: str | Path,
    problem: GPTOProblem,
    evaluation: Evaluation,
    *,
    label: str,
) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{label}.npz",
        design_variables=evaluation.design_variables,
        points=evaluation.geometry.points,
        bars=evaluation.geometry.bars,
        size_variables=evaluation.geometry.size_variables,
        radii=evaluation.geometry.radii,
        volume_density=evaluation.projection.volume_density,
        stiffness_density=evaluation.projection.stiffness_density,
        displacement=evaluation.finite_element.displacement,
        compliance_gradient=evaluation.compliance_gradient,
        volume_gradient=evaluation.volume_gradient,
    )
    summary = evaluation_summary(problem, evaluation)
    (output / f"{label}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_vtk(
        output / f"{label}.vtk",
        problem,
        evaluation.projection.volume_density,
        evaluation.projection.stiffness_density,
    )
    return output


def save_optimization(
    output_directory: str | Path,
    problem: GPTOProblem,
    result: OptimizationResult,
) -> Path:
    output = save_evaluation(
        output_directory, problem, result.evaluation, label="final"
    )
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(result.history[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in result.history)
    payload = {
        "status": result.status,
        "message": result.message,
        "iterations": result.iterations,
        "analysis_count": result.analysis_count,
        "final": evaluation_summary(problem, result.evaluation),
    }
    (output / "optimization.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output


def evaluation_summary(problem: GPTOProblem, evaluation: Evaluation) -> dict[str, Any]:
    case = problem.case
    parameters: ProjectionParameters = problem.projection_parameters
    expected_compliance = case.expected.paper_compliance
    return {
        "case": case.name,
        "title": case.title,
        "profile": case.profile,
        "mesh": {
            "dimension": case.mesh.dim,
            "nodes": case.mesh.n_nodes,
            "elements": case.mesh.n_elements,
            "structured_shape": case.mesh.structured_shape,
        },
        "method": {
            "aggregation": parameters.aggregation.value,
            "aggregation_parameter": parameters.aggregation_parameter,
            "penalization": parameters.penalization,
            "penalization_parameter": parameters.penalization_parameter,
            "minimum_density": case.material.minimum_density,
        },
        "computed": {
            "compliance": evaluation.compliance,
            "volume_fraction": evaluation.volume_fraction,
            "volume_constraint_limit": case.volume_fraction_limit,
            "maximum_volume_density": evaluation.maximum_volume_density,
            "maximum_stiffness_density": evaluation.maximum_stiffness_density,
            "maximum_component_overlap": evaluation.projection.maximum_component_overlap,
            "linear_solver_info": evaluation.finite_element.solver_info,
            "linear_solver_iterations": evaluation.finite_element.solver_iterations,
            "linear_relative_residual": evaluation.finite_element.relative_residual,
        },
        "paper": {
            "iterations": case.expected.paper_iterations,
            "compliance": expected_compliance,
            "volume_fraction": case.expected.volume_fraction,
            "notes": case.expected.notes,
        },
        "comparison": {
            "absolute_compliance_error": None
            if expected_compliance is None
            else abs(evaluation.compliance - expected_compliance),
            "relative_compliance_error": None
            if expected_compliance is None
            else abs(evaluation.compliance - expected_compliance)
            / abs(expected_compliance),
        },
    }


def write_vtk(
    path: str | Path,
    problem: GPTOProblem,
    volume_density: np.ndarray,
    stiffness_density: np.ndarray,
) -> None:
    """Write an ASCII VTK unstructured grid readable by ParaView."""

    path = Path(path)
    mesh = problem.case.mesh
    coordinates = np.zeros((mesh.n_nodes, 3), dtype=float)
    coordinates[:, : mesh.dim] = mesh.coordinates
    nodes_per_element = mesh.elements.shape[1]
    vtk_type = 9 if mesh.dim == 2 else 12
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write(f"GPTO {problem.case.name}\n")
        stream.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {mesh.n_nodes} double\n")
        for point in coordinates:
            stream.write(f"{point[0]:.16g} {point[1]:.16g} {point[2]:.16g}\n")
        stream.write(
            f"CELLS {mesh.n_elements} {mesh.n_elements * (nodes_per_element + 1)}\n"
        )
        for element in mesh.elements:
            values = " ".join(str(int(node)) for node in element)
            stream.write(f"{nodes_per_element} {values}\n")
        stream.write(f"CELL_TYPES {mesh.n_elements}\n")
        stream.writelines(f"{vtk_type}\n" for _ in range(mesh.n_elements))
        stream.write(f"CELL_DATA {mesh.n_elements}\n")
        for name, values in (
            ("volume_density", volume_density),
            ("stiffness_density", stiffness_density),
        ):
            stream.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            stream.writelines(f"{float(value):.16g}\n" for value in values)
