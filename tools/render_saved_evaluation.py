"""Render a saved evaluation without repeating an expensive FE solve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpto.cases import build_case
from gpto.config import Geometry
from gpto.plotting import plot_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("archive", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("paper", "smoke", "tiny"), default="paper")
    arguments = parser.parse_args()

    case = build_case(arguments.case, profile=arguments.profile)
    with np.load(arguments.archive) as data:
        geometry = Geometry(
            points=data["points"].copy(),
            bars=data["bars"].copy(),
            size_variables=data["size_variables"].copy(),
            radii=data["radii"].copy(),
        )
        volume_density = data["volume_density"].copy()
    summary = json.loads(arguments.summary.read_text(encoding="utf-8"))
    computed = summary["computed"]

    problem = SimpleNamespace(case=case)
    evaluation = SimpleNamespace(
        geometry=geometry,
        projection=SimpleNamespace(volume_density=volume_density),
        compliance=float(computed["compliance"]),
        volume_fraction=float(computed["volume_fraction"]),
        maximum_volume_density=float(np.max(volume_density)),
    )
    plot_evaluation(
        arguments.output,
        problem,  # type: ignore[arg-type]
        evaluation,  # type: ignore[arg-type]
        title=f"{case.title} — saved design",
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
