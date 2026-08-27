"""Command-line interface for the GPTO reproduction."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .artifacts import evaluation_summary, save_evaluation, save_optimization
from .cases import available_cases, build_case, load_reference_geometry
from .geometry import AggregationScheme, ProjectionParameters
from .optimizer import IterationRecord, optimize
from .plotting import plot_evaluation, plot_history
from .problem import GPTOProblem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpto",
        description=(
            "Reproduce the geometry-projection topology-optimization cases "
            "from Smith and Norato (2020)."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("cases", help="list the packaged article cases")

    analyze = subparsers.add_parser(
        "analyze", help="analyze an initial or packaged reference design"
    )
    _add_problem_arguments(analyze)
    _add_design_argument(analyze)
    _add_output_arguments(analyze)

    gradient = subparsers.add_parser(
        "gradient-check", help="compare analytical sensitivities with finite differences"
    )
    _add_problem_arguments(gradient, default_profile="smoke")
    _add_design_argument(gradient)
    gradient.add_argument(
        "--indices",
        help="comma-separated zero-based design-variable indices; default selects representatives",
    )
    gradient.add_argument(
        "--step", type=float, default=1.0e-6, help="finite-difference step in scaled variables"
    )
    gradient.add_argument(
        "--difference-scheme",
        choices=("central", "forward"),
        default="central",
        help="finite-difference formula",
    )
    gradient.add_argument(
        "--allow-outside-bounds",
        action="store_true",
        help="allow the infinitesimal perturbation to cross a design bound",
    )
    gradient.add_argument("--output", type=Path, help="output directory")

    optimization = subparsers.add_parser(
        "optimize", help="run the paper formulation with single-constraint MMA"
    )
    _add_problem_arguments(optimization)
    _add_design_argument(optimization)
    _add_output_arguments(optimization)
    optimization.add_argument("--max-iterations", type=int)
    optimization.add_argument("--move-limit", type=float)
    optimization.add_argument("--design-step-tolerance", type=float)
    optimization.add_argument("--kkt-tolerance", type=float)
    optimization.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="print every Nth iteration (0 disables iteration output)",
    )

    figures = subparsers.add_parser(
        "figures", help="recreate the paper's numbered Figure 4-18 gallery"
    )
    figures.add_argument(
        "--output",
        type=Path,
        default=Path("output/figures/gpto-paper-results"),
        help="directory for numbered PNGs, HTML gallery, and manifest",
    )
    figures.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="root containing the full-resolution numerical artifacts",
    )
    figures.add_argument(
        "--atlas",
        type=Path,
        default=Path("output/pdf/gpto-paper-results-atlas.pdf"),
        help="PDF atlas output path",
    )
    figures.add_argument(
        "--no-atlas", action="store_true", help="skip PDF atlas generation"
    )
    figures.add_argument("--dpi", type=int, default=240, help="PNG resolution")
    return parser


def _add_problem_arguments(
    parser: argparse.ArgumentParser, *, default_profile: str = "paper"
) -> None:
    parser.add_argument("case", choices=available_cases())
    parser.add_argument(
        "--profile",
        choices=("paper", "smoke", "tiny"),
        default=default_profile,
        help="paper mesh or a reduced verification mesh",
    )
    parser.add_argument(
        "--aggregation",
        choices=tuple(item.value for item in AggregationScheme),
        default=AggregationScheme.LEGACY_MODIFIED_P_NORM.value,
    )
    parser.add_argument("--aggregation-parameter", type=float, default=8.0)
    parser.add_argument("--penalization", choices=("SIMP", "RAMP"), default="SIMP")
    parser.add_argument("--penalization-parameter", type=float, default=3.0)
    parser.add_argument(
        "--sample-radius",
        type=float,
        help="override the element-derived projection sampling-window radius",
    )


def _add_design_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--design",
        choices=("initial", "reference"),
        default="initial",
        help="initial paper geometry or the design bundled with the upstream repository",
    )


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, help="output directory")
    parser.add_argument("--no-plot", action="store_true", help="skip PNG plots")


def _problem_from_arguments(arguments: argparse.Namespace) -> GPTOProblem:
    case = build_case(arguments.case, profile=arguments.profile)
    parameters = ProjectionParameters(
        aggregation=AggregationScheme.parse(arguments.aggregation),
        aggregation_parameter=arguments.aggregation_parameter,
        penalization=arguments.penalization,
        penalization_parameter=arguments.penalization_parameter,
        sample_radius=arguments.sample_radius,
    )
    return GPTOProblem(case, parameters)


def _design_from_arguments(
    problem: GPTOProblem, arguments: argparse.Namespace
) -> np.ndarray:
    if arguments.design == "initial":
        return problem.initial_design.copy()
    geometry = load_reference_geometry(arguments.case)
    design = problem.design_map.encode(geometry)
    if np.any(design < problem.lower_bounds - 1.0e-10) or np.any(
        design > problem.upper_bounds + 1.0e-10
    ):
        raise ValueError(
            "the upstream reference design lies outside this case's paper bounds"
        )
    return np.minimum(np.maximum(design, problem.lower_bounds), problem.upper_bounds)


def _default_output(arguments: argparse.Namespace, operation: str) -> Path:
    aggregation = arguments.aggregation.replace("_", "-")
    return (
        Path("results")
        / arguments.case
        / f"{operation}-{arguments.profile}-{arguments.design}-{aggregation}"
    )


def _print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def _run_analyze(arguments: argparse.Namespace) -> int:
    problem = _problem_from_arguments(arguments)
    design = _design_from_arguments(problem, arguments)
    evaluation = problem.evaluate(design)
    output = arguments.output or _default_output(arguments, "analysis")
    save_evaluation(output, problem, evaluation, label=arguments.design)
    if not arguments.no_plot:
        plot_evaluation(
            output / f"{arguments.design}.png",
            problem,
            evaluation,
            title=f"{problem.case.title} — {arguments.design} design",
        )
    summary = evaluation_summary(problem, evaluation)
    summary["artifacts"] = str(output.resolve())
    _print_summary(summary)
    return 0


def _parse_indices(raw: str | None, problem: GPTOProblem) -> list[int]:
    if raw and raw.strip().lower() == "all":
        indices = list(range(problem.design_map.n_variables))
    elif raw:
        indices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        mapping = problem.design_map
        indices = [0]
        if mapping.dim > 1:
            indices.append(1)
        indices.append(mapping.size_start)
        if problem.upper_bounds[mapping.radius_start] > problem.lower_bounds[
            mapping.radius_start
        ]:
            indices.append(mapping.radius_start)
    if not indices:
        raise ValueError("no gradient-check indices were supplied")
    upper = problem.design_map.n_variables
    if any(index < 0 or index >= upper for index in indices):
        raise ValueError(f"gradient-check indices must be in [0, {upper - 1}]")
    return indices


def _run_gradient_check(arguments: argparse.Namespace) -> int:
    if arguments.step <= 0:
        raise ValueError("--step must be positive")
    problem = _problem_from_arguments(arguments)
    design = _design_from_arguments(problem, arguments)
    indices = _parse_indices(arguments.indices, problem)
    entries = problem.gradient_check(
        design,
        indices=indices,
        step=arguments.step,
        method=arguments.difference_scheme,
        enforce_bounds=not arguments.allow_outside_bounds,
    )
    payload = {
        "case": problem.case.name,
        "profile": problem.case.profile,
        "design": arguments.design,
        "step": arguments.step,
        "difference_scheme": arguments.difference_scheme,
        "enforce_bounds": not arguments.allow_outside_bounds,
        "entries": [asdict(entry) for entry in entries],
        "max_relative_error_compliance": max(
            (entry.relative_error_compliance for entry in entries), default=None
        ),
        "max_relative_error_volume": max(
            (entry.relative_error_volume for entry in entries), default=None
        ),
    }
    output = arguments.output or _default_output(arguments, "gradient-check")
    output.mkdir(parents=True, exist_ok=True)
    (output / "gradient_check.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    payload["artifacts"] = str(output.resolve())
    _print_summary(payload)
    return 0


def _run_optimize(arguments: argparse.Namespace) -> int:
    problem = _problem_from_arguments(arguments)
    design = _design_from_arguments(problem, arguments)
    options = problem.case.optimization
    replacements = {
        name: getattr(arguments, name)
        for name in (
            "max_iterations",
            "move_limit",
            "design_step_tolerance",
            "kkt_tolerance",
        )
        if getattr(arguments, name) is not None
    }
    if replacements:
        options = replace(options, **replacements)
    if options.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive")
    if arguments.print_every < 0:
        raise ValueError("--print-every cannot be negative")

    def report(record: IterationRecord, _: object) -> None:
        every = arguments.print_every
        if every and (record.iteration == 0 or record.iteration % every == 0):
            print(
                f"iter={record.iteration:4d} "
                f"C={record.compliance:.10g} "
                f"vf={record.volume_fraction:.8f} "
                f"|dx|={record.design_change_norm:.3e} "
                f"KKT={record.kkt_norm:.3e} "
                f"t={record.elapsed_seconds:.2f}s",
                flush=True,
            )

    result = optimize(problem, design_variables=design, options=options, callback=report)
    output = arguments.output or _default_output(arguments, "optimization")
    save_optimization(output, problem, result)
    if not arguments.no_plot:
        plot_evaluation(output / "final.png", problem, result.evaluation, title="Final design")
        plot_history(output / "history.png", result, problem.case.volume_fraction_limit)
    payload = {
        "status": result.status,
        "message": result.message,
        "iterations": result.iterations,
        "analysis_count": result.analysis_count,
        "final": evaluation_summary(problem, result.evaluation),
        "artifacts": str(output.resolve()),
    }
    _print_summary(payload)
    return 0


def _run_figures(arguments: argparse.Namespace) -> int:
    from .paper_figures import generate_paper_figures

    payload = generate_paper_figures(
        output_directory=arguments.output,
        results_root=arguments.results_root,
        atlas_path=None if arguments.no_atlas else arguments.atlas,
        dpi=arguments.dpi,
    )
    _print_summary(payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "cases":
            for case in available_cases():
                print(case)
            return 0
        if arguments.command == "analyze":
            return _run_analyze(arguments)
        if arguments.command == "gradient-check":
            return _run_gradient_check(arguments)
        if arguments.command == "optimize":
            return _run_optimize(arguments)
        if arguments.command == "figures":
            return _run_figures(arguments)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    raise AssertionError(f"unhandled command {arguments.command!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
