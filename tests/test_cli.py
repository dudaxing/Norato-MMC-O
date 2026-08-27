from __future__ import annotations

import json

from gpto.cli import main


def test_cases_command(capsys) -> None:
    assert main(["cases"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "mbb2d",
        "lbracket2d",
        "cantilever3d",
    ]


def test_tiny_analysis_writes_machine_readable_artifacts(tmp_path) -> None:
    assert (
        main(
            [
                "analyze",
                "mbb2d",
                "--profile",
                "tiny",
                "--no-plot",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    payload = json.loads((tmp_path / "initial.json").read_text(encoding="utf-8"))
    assert payload["case"] == "mbb2d"
    assert payload["mesh"]["elements"] == 400
    assert (tmp_path / "initial.npz").is_file()
    assert (tmp_path / "initial.vtk").is_file()
