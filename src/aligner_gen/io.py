"""STL loading and selection persistence."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

SELECTIONS_DIR = Path(__file__).resolve().parents[2] / "data" / "selections"


@dataclass
class LoadedMesh:
    mesh: trimesh.Trimesh
    stl_path: Path
    stl_hash: str


def stl_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stl(path: str | Path) -> LoadedMesh:
    p = Path(path).resolve()
    mesh = trimesh.load(p, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a Trimesh: {type(mesh)}")
    # Merge close vertices — STL has none initially, every triangle has separate verts.
    mesh.merge_vertices()
    return LoadedMesh(mesh=mesh, stl_path=p, stl_hash=stl_sha256(p))


def selection_path_for(stl_path: Path) -> Path:
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SELECTIONS_DIR / f"{stl_path.stem}.npz"


def save_selection(loaded: LoadedMesh, mask: np.ndarray) -> Path:
    out = selection_path_for(loaded.stl_path)
    np.savez_compressed(
        out,
        mask=mask.astype(bool),
        stl_hash=loaded.stl_hash,
        stl_name=loaded.stl_path.name,
        n_vertices=len(loaded.mesh.vertices),
    )
    return out


def load_selection(loaded: LoadedMesh) -> np.ndarray | None:
    """Return saved mask if it exists and matches the STL. Else None."""
    path = selection_path_for(loaded.stl_path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    saved_hash = str(data["stl_hash"])
    saved_n = int(data["n_vertices"])
    n_verts = len(loaded.mesh.vertices)
    if saved_n != n_verts:
        print(
            f"[io] Selekcja ma {saved_n} wierzchołków, mesh {n_verts} — pomijam load."
        )
        return None
    if saved_hash != loaded.stl_hash:
        print("[io] Hash STL się zmienił — selekcja może być nieaktualna, ale ładuję.")
    return data["mask"].astype(bool)
