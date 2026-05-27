"""CLI entry point: python -m aligner_gen path/to/scan.stl [--qt]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .io import load_stl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aligner_gen")
    parser.add_argument("stl", type=Path, help="ścieżka do pliku STL łuku zębowego")
    parser.add_argument(
        "--qt",
        action="store_true",
        help="Użyj nowego viewera PySide6 + pyvistaqt (Faza 1 scaffold). "
        "Bez flagi — legacy PyVista viewer.",
    )
    args = parser.parse_args(argv)

    if not args.stl.exists():
        print(f"[error] Plik nie istnieje: {args.stl}", file=sys.stderr)
        return 2

    if args.qt:
        from .qt_viewer import run as run_qt
        return run_qt(args.stl)

    print(f"[main] Ładuję STL: {args.stl}")
    loaded = load_stl(args.stl)
    print(
        f"[main] Mesh: {len(loaded.mesh.vertices)} verts, "
        f"{len(loaded.mesh.faces)} faces, "
        f"watertight={loaded.mesh.is_watertight}"
    )

    from .viewer import SelectionViewer
    viewer = SelectionViewer(loaded)
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
