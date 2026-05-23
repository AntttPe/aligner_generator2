"""Operacje voxelowe i SDF (backend: scipy + skimage).

Architektura: VoxelGrid trzyma dane 3D + transformację (origin, pitch).
Wszystkie operacje SDF/morfologiczne działają w voxelach, ale parametry
podajemy w milimetrach.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy import ndimage as ndi


@dataclass
class VoxelGrid:
    data: np.ndarray  # 3D array (bool lub float32)
    origin: np.ndarray  # (3,) world-coord min corner
    pitch: float  # voxel size [mm]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    def voxel_centers(self) -> np.ndarray:
        """(N, 3) world coords of all voxel centers (C-order flatten)."""
        nx, ny, nz = self.shape
        ix, iy, iz = np.mgrid[0:nx, 0:ny, 0:nz]
        return np.stack(
            [
                self.origin[0] + (ix + 0.5) * self.pitch,
                self.origin[1] + (iy + 0.5) * self.pitch,
                self.origin[2] + (iz + 0.5) * self.pitch,
            ],
            axis=-1,
        ).reshape(-1, 3)

    def world_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        nx, ny, nz = self.shape
        return (
            self.origin.copy(),
            self.origin + np.array([nx, ny, nz]) * self.pitch,
        )


# ---------- voxelizacja ----------

def voxelize_solid(
    mesh: trimesh.Trimesh,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    pitch: float,
) -> VoxelGrid:
    """Voxelizacja watertight mesha jako bool grid (True = wnętrze).

    Wynikowy grid ma DOKŁADNIE rozmiar bbox-a w voxelach. Tam gdzie bbox
    wystaje poza zakres mesha (np. ~2mm nad chewing surface — żeby closing
    miał gdzie zaokrąglić cuspy), brakujące voxele są dopełniane False
    ("outside teeth"). Bez tego closing był ucinany przy granicy gridu
    i nakładka miała dziury na cusp tops.
    """
    # Memory guard: trimesh voxelizuje CAŁY mesh. Jeśli bbox mesha jest
    # absurdalnie duży (outlier vertex daleko od modelu), grid eksploduje
    # i proces zostaje zabity przez OOM. Lepiej jasny błąd.
    full_extent = mesh.bounds[1] - mesh.bounds[0]
    full_voxels = float(np.prod(np.ceil(full_extent / pitch)))
    if full_voxels > 3e9:
        raise MemoryError(
            f"Voxelizacja calego mesha = {full_voxels / 1e9:.1f}G voxeli "
            f"(bbox {np.round(full_extent, 1)} mm). Prawdopodobnie outlier "
            f"daleko od modelu. Sprawdz/napraw STL."
        )

    vg = mesh.voxelized(pitch=pitch)
    vg = vg.fill(method="holes")
    full = vg.matrix
    grid_origin = np.asarray(vg.bounds[0], dtype=float)
    grid_shape = np.array(full.shape)

    # bbox-aligned indices w układzie trimesh-grida (NIE klipujemy do shape)
    i_min = np.floor((bbox_min - grid_origin) / pitch).astype(np.int64)
    i_max = np.ceil((bbox_max - grid_origin) / pitch).astype(np.int64)
    target_shape = tuple(int(x) for x in (i_max - i_min))

    out = np.zeros(target_shape, dtype=bool)

    # overlap między mesh-grid a target bbox-grid
    src_lo = np.maximum(i_min, 0)
    src_hi = np.minimum(i_max, grid_shape)
    if np.all(src_hi > src_lo):
        dst_lo = src_lo - i_min
        dst_hi = src_hi - i_min
        out[
            dst_lo[0]:dst_hi[0],
            dst_lo[1]:dst_hi[1],
            dst_lo[2]:dst_hi[2],
        ] = full[
            src_lo[0]:src_hi[0],
            src_lo[1]:src_hi[1],
            src_lo[2]:src_hi[2],
        ]

    crop_origin = grid_origin + i_min.astype(float) * pitch
    return VoxelGrid(data=out, origin=crop_origin, pitch=pitch)


# ---------- SDF ----------

def sdf_from_binary(binary: VoxelGrid) -> VoxelGrid:
    """Signed distance field: negatywne wewnątrz, pozytywne na zewnątrz."""
    inside = binary.data
    dist_in = ndi.distance_transform_edt(inside)
    dist_out = ndi.distance_transform_edt(~inside)
    sdf = (dist_out - dist_in).astype(np.float32) * binary.pitch
    return VoxelGrid(data=sdf, origin=binary.origin.copy(), pitch=binary.pitch)


# ---------- morfologia ----------

def binary_close(binary: VoxelGrid, radius_mm: float) -> VoxelGrid:
    """Morphological closing: dylatacja + erozja. Wypełnia embrasury.

    Pad False-em przed operacją, żeby krawędzie gridu nie działały jak
    bariera. Bez tego konweksne kształty stykające się z brzegiem gridu
    są niesymetrycznie erodowane.
    """
    if radius_mm <= 0:
        return VoxelGrid(
            data=binary.data.copy(),
            origin=binary.origin.copy(),
            pitch=binary.pitch,
        )
    iters = max(1, int(round(radius_mm / binary.pitch)))
    pad = iters + 1
    padded = np.pad(binary.data, pad, mode="constant", constant_values=False)
    dilated = ndi.binary_dilation(padded, iterations=iters)
    closed = ndi.binary_erosion(dilated, iterations=iters)
    closed = closed[pad:-pad, pad:-pad, pad:-pad]
    return VoxelGrid(
        data=closed.copy(),
        origin=binary.origin.copy(),
        pitch=binary.pitch,
    )


# ---------- pole odległości do punktów ----------

def distance_to_points(template: VoxelGrid, points: np.ndarray) -> VoxelGrid:
    """Dla każdego voxela: odległość do najbliższego z `points` (mm).

    Implementacja przez EDT: voxelize `points` jako binary seed-mask
    (False na pozycjach voxeli-ziaren, True w pozostałych), potem
    distance_transform_edt zwraca odległość do najbliższego False.
    To jest O(N) gdzie N = liczba voxeli, dużo szybsze niż cKDTree.query
    przy dużych gridach (sekundy zamiast minut).

    Dokładność: pitch (każde ziarno jest 'snapowane' do najbliższego voxela).
    """
    shape = np.array(template.shape)
    origin = np.asarray(template.origin, dtype=float)
    pitch = float(template.pitch)

    if points.size == 0:
        return VoxelGrid(
            data=np.full(template.shape, np.inf, dtype=np.float32),
            origin=template.origin.copy(),
            pitch=template.pitch,
        )

    indices = np.floor((np.asarray(points, dtype=float) - origin) / pitch).astype(np.int64)
    in_bounds = np.all((indices >= 0) & (indices < shape[None, :]), axis=1)
    indices = indices[in_bounds]

    seed_mask = np.ones(template.shape, dtype=bool)
    if indices.size > 0:
        seed_mask[indices[:, 0], indices[:, 1], indices[:, 2]] = False

    dist_voxels = ndi.distance_transform_edt(seed_mask)
    return VoxelGrid(
        data=(dist_voxels * pitch).astype(np.float32),
        origin=template.origin.copy(),
        pitch=template.pitch,
    )


# ---------- voxelize surface + selection proximity ----------

def voxelize_surface_to_template(
    mesh: trimesh.Trimesh, template: VoxelGrid
) -> np.ndarray:
    """Voxelizuj mesh jako shell (BEZ fillu) i zmapuj voxele do gridu `template`.

    Zwraca bool mask z kształtem template.data.shape. True = voxel zawiera
    fragment powierzchni mesha.
    """
    if len(mesh.faces) == 0:
        return np.zeros(template.shape, dtype=bool)
    vg = mesh.voxelized(pitch=template.pitch)
    surface_pts = np.asarray(vg.points, dtype=float)
    indices = np.floor((surface_pts - template.origin) / template.pitch).astype(np.int64)
    shape_arr = np.array(template.shape)
    valid = np.all((indices >= 0) & (indices < shape_arr[None, :]), axis=1)
    indices = indices[valid]
    mask = np.zeros(template.shape, dtype=bool)
    if indices.size > 0:
        mask[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return mask


def selection_proximity(
    mesh: trimesh.Trimesh,
    selected_vert_mask: np.ndarray,
    template: VoxelGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Owner-based proximity dla każdego voxela w `template`.

    Zwraca:
        dist_to_selected: float32 grid — odległość od najbliższego voxela
            **na zaznaczonej powierzchni** (faces gdzie wszystkie 3 verts
            są w selekcji)
        owner: bool grid — True jeśli najbliższy mesh-surface voxel
            jest zaznaczony (czyli ten voxel "należy" do zaznaczonego obszaru
            powierzchni). To naturalnie cutuje aligner na granicy selekcji
            (np. gingival margin).
    """
    shape = template.shape
    pitch = template.pitch

    face_in_selection = selected_vert_mask[mesh.faces].all(axis=1)
    n_sel_faces = int(face_in_selection.sum())
    if n_sel_faces == 0:
        return (
            np.full(shape, np.inf, dtype=np.float32),
            np.zeros(shape, dtype=bool),
        )

    sel_submesh = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[face_in_selection],
        process=False,
    )
    sel_voxels = voxelize_surface_to_template(sel_submesh, template)

    # Pełna powierzchnia mesha (selected + unselected)
    full_voxels = voxelize_surface_to_template(mesh, template)
    # selected voxels POWINNY być podzbiorem full_voxels; jeśli nie, dorzuć je
    # (artefakty voxelizacji submeshu).
    full_voxels = full_voxels | sel_voxels

    if not sel_voxels.any():
        return (
            np.full(shape, np.inf, dtype=np.float32),
            np.zeros(shape, dtype=bool),
        )

    # 1) distance to selected surface voxels
    dist_sel_v = ndi.distance_transform_edt(~sel_voxels).astype(np.float32) * pitch

    # 2) nearest full-surface voxel index → check if it's selected
    _, indices = ndi.distance_transform_edt(~full_voxels, return_indices=True)
    owner = sel_voxels[indices[0], indices[1], indices[2]]
    del indices  # free big array

    return dist_sel_v, owner


# ---------- marching cubes ----------

def marching_cubes_mesh(
    field: VoxelGrid,
    level: float = 0.0,
    *,
    gradient_direction: str = "ascent",
) -> trimesh.Trimesh:
    """Iso-surface z pola skalarnego → trimesh w world coords.

    gradient_direction='ascent' oznacza: normalne na zewnątrz w miejscach
    gdzie field rośnie. Dla pól "TSDF-like" (pozytywne wewnątrz, ujemne
    na zewnątrz) chcemy 'descent'.
    """
    from skimage import measure

    verts, faces, _normals, _values = measure.marching_cubes(
        field.data,
        level=level,
        spacing=(field.pitch, field.pitch, field.pitch),
        gradient_direction=gradient_direction,
        allow_degenerate=False,
    )
    verts = verts + field.origin
    return trimesh.Trimesh(vertices=verts, faces=faces, process=True)


# ---------- helpery ----------

def grid_memory_estimate(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    pitch: float,
    *,
    bytes_per_voxel: int = 4,
) -> tuple[tuple[int, int, int], int]:
    """Przybliżona ocena rozmiaru gridu i pamięci. Zwraca (shape, MB)."""
    shape = tuple(int(np.ceil((b - a) / pitch)) for a, b in zip(bbox_min, bbox_max))
    n_voxels = int(np.prod(shape))
    mb = n_voxels * bytes_per_voxel // (1024 * 1024)
    return shape, mb  # type: ignore[return-value]
