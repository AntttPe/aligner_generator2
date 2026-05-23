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


def load_stl(path: str | Path, *, repair: bool = True) -> LoadedMesh:
    p = Path(path).resolve()
    mesh = trimesh.load(p, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a Trimesh: {type(mesh)}")
    # Merge close vertices — STL has none initially, every triangle has separate verts.
    mesh.merge_vertices()

    if repair:
        mesh = _repair_mesh(mesh)

    return LoadedMesh(mesh=mesh, stl_path=p, stl_hash=stl_sha256(p))


def _repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair dążący do watertight — KRYTYCZNE dla voxelizacji.

    Niewatertight mesh (np. otwarta podstawa modelu) powoduje że
    trimesh.voxelized().fill() nie wypełnia wnętrza → SDF z cienkiej skorupy
    → poszarpana krawędź nakładki, wyrostki, widoczne voxele.

    Strategia (szybka, nieinwazyjna — NIE remeshuje):
      1. Lekkie czyszczenie (degenerate/duplicate faces).
      2. Jeśli już watertight — koniec (geometria nietknięta).
      3. trimesh.fill_holes() — tania próba na małe dziury.
      4. Capping boundary loops (fan z centroidu) — zaślepia duże otwory
         (podstawa modelu). DOKLEJA nowe verts NA KOŃCU → oryginalne indeksy
         0..N-1 nietknięte → zapisana selekcja (na zębach) pozostaje ważna.
    """
    wt_before = mesh.is_watertight
    n_before = len(mesh.vertices)

    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if mesh.is_watertight:
        print(f"[io] repair: watertight=True (czysty, {len(mesh.vertices)} verts)")
        return mesh

    mesh.fill_holes()
    if mesh.is_watertight:
        print(
            f"[io] repair: watertight True przez fill_holes ({len(mesh.vertices)} verts)"
        )
        return mesh

    capped = _cap_boundary_loops(mesh)
    # Kryterium sukcesu: NIE is_watertight (czapa ma T-junctions które
    # voxelizacji nie szkodzą), tylko czy mesh daje wypełnioną BRYŁĘ.
    solid_ok = _verify_solid(capped)
    print(
        f"[io] repair: capping, verts {n_before} → {len(capped.vertices)}, "
        f"bryła={'OK' if solid_ok else 'NIE'}"
    )
    if not solid_ok:
        print(
            "[io] UWAGA: mesh nie daje wypełnionej bryły (głębsze defekty niż "
            "otwarte dziury). Voxelizacja da artefakty. Napraw STL zewnętrznie "
            "(Meshmixer 'Make Solid' / Blender 'Make Manifold') i wczytaj ponownie."
        )
    return capped


def _verify_solid(mesh: trimesh.Trimesh, *, n_div: int = 60) -> bool:
    """Szybki coarse test: czy mesh wypełnia się jako bryła (nie cienka skorupa).

    Voxelizuje zgrubnie i sprawdza stosunek wypełnienia. Cienka skorupa
    (fill nie zadziałał) ma ~1-2% wypełnienia, prawdziwa bryła znacznie więcej.
    """
    import numpy as np

    try:
        max_dim = float(np.max(mesh.bounds[1] - mesh.bounds[0]))
        pitch = max(max_dim / n_div, 1e-3)
        vg = mesh.voxelized(pitch=pitch).fill(method="holes")
        ratio = float(vg.matrix.sum()) / max(vg.matrix.size, 1)
        return ratio > 0.04
    except Exception as e:  # pragma: no cover
        print(f"[io] _verify_solid błąd: {e}")
        return False


def _cap_boundary_loops(
    mesh: trimesh.Trimesh, *, cap_max_edge: float = 0.5
) -> trimesh.Trimesh:
    """Zaślep otwarte pętle brzegowe — czapa z MAŁYCH trójkątów.

    Wachlarz z centroidu daje wielkie trójkąty (centroid→brzeg), które
    rozwalają subdivide-voxelizację (dzieli każdy trójkąt do rozmiaru
    voxela → eksplozja pamięci). Dlatego czapę budujemy osobno i
    subdividujemy do `cap_max_edge` mm zanim dokleimy.

    Oryginalne wierzchołki/indeksy zachowane (czapa doklejana NA KOŃCU),
    więc selekcja na zębach pozostaje ważna. Czapa to geometrycznie
    przylegający patch (współdzielone pozycje brzegu) — dla voxelizacji
    wystarcza (pokrywa powierzchnię), nawet jeśli topologicznie osobny.
    """
    import numpy as np
    from trimesh.remesh import subdivide_to_size

    try:
        outline = mesh.outline()
    except Exception as e:  # pragma: no cover
        print(f"[io] outline() błąd: {e} — pomijam capping.")
        return mesh

    entities = getattr(outline, "entities", [])
    if len(entities) == 0:
        return mesh

    verts_blocks = [np.asarray(mesh.vertices, dtype=float)]
    faces_blocks = [np.asarray(mesh.faces, dtype=np.int64)]
    offset = len(mesh.vertices)
    n_capped = 0

    for ent in entities:
        try:
            ring = np.asarray(outline.vertices[ent.points], dtype=float)
        except Exception:
            continue
        # usuń kolejne duplikaty + ewentualne domknięcie
        keep = np.ones(len(ring), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(ring, axis=0)) > 1e-9, axis=1)
        ring = ring[keep]
        if len(ring) >= 2 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        if len(ring) < 3:
            continue

        centroid = ring.mean(axis=0)
        cap_v = np.vstack([ring, centroid])
        ci = len(ring)
        cap_f = np.array(
            [[i, (i + 1) % len(ring), ci] for i in range(len(ring))],
            dtype=np.int64,
        )
        # subdivide czapę do małych trójkątów (kluczowe dla voxelizacji)
        sv, sf = subdivide_to_size(cap_v, cap_f, max_edge=cap_max_edge)

        verts_blocks.append(sv)
        faces_blocks.append(sf + offset)
        offset += len(sv)
        n_capped += 1

    if n_capped == 0:
        return mesh

    capped = trimesh.Trimesh(
        vertices=np.vstack(verts_blocks),
        faces=np.vstack(faces_blocks),
        process=False,
    )
    added = offset - len(mesh.vertices)
    # zespaw szew czapy z oryginałem (welduje pokrywające się brzegi) →
    # prawdziwy watertight. Deterministyczne, więc selekcja persystuje
    # między uruchomieniami (zapisywana zawsze na meshu po repair).
    capped.merge_vertices()
    try:
        trimesh.repair.fix_normals(capped)
    except Exception as e:  # pragma: no cover
        print(f"[io] fix_normals po capping: {e}")
    print(
        f"[io] capping: zaślepiono {n_capped} pętli (czapa subdiv do "
        f"{cap_max_edge}mm, +{added} verts przed merge)"
    )
    return capped


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
