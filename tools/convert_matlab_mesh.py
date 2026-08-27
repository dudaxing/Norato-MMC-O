"""Convert a GPTO Gmsh-to-MATLAB mesh file into the package NPZ format.

The released PyGPTO L-bracket mesh rounds coordinates to four decimal
places.  This converter reads the original MATLAB export so paper-profile
analyses retain the full coordinate precision used by GPTO.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _numeric_block(text: str, name: str) -> np.ndarray:
    marker = f"msh.{name}"
    marker_start = text.find(marker)
    if marker_start < 0:
        raise ValueError(f"section {marker!r} was not found")
    block_start = text.find("[", marker_start)
    block_end = text.find("];", block_start)
    if block_start < 0 or block_end < 0:
        raise ValueError(f"section {marker!r} is not a complete matrix")
    rows: list[list[float]] = []
    for raw_line in text[block_start + 1 : block_end].splitlines():
        line = raw_line.strip().rstrip(";")
        if line:
            rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError(f"section {marker!r} is empty")
    return np.asarray(rows)


def convert(source: Path, destination: Path) -> None:
    text = source.read_text(encoding="utf-8", errors="strict")
    coordinates = _numeric_block(text, "POS")[:, :2]
    elements = _numeric_block(text, "QUADS")[:, :4].astype(np.int64) - 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, coordinates=coordinates, elements=elements
    )
    print(
        f"wrote {destination}: {len(coordinates)} nodes, "
        f"{len(elements)} Q4 elements"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
