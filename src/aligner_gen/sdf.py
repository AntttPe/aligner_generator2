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

    Voxelizujemy CAŁY mesh (trimesh API tego wymaga), a potem cropujemy
    do bboxa. Tańsze pamięciowo niż wydaje się: voxel binary jest cheap.
    """
    vg = mesh.voxelized(pitch=pitch)
    vg = vg.fill(method="holes")  # zamknięte wnętrze
    full = vg.matrix
    grid_origin = np.asarray(vg.bounds[0], dtype=float)

    shape = np.array(full.shape)
    i_min = np.floor((bbox_min - grid_origin) / pitch).astype(int)
    i_max = np.ceil((bbox_max - grid_origin) / pitch).astype(int)
    i_min = np.clip(i_min, 0, shape)
    i_max = np.clip(i_max, i_min + 1, shape)

    cropped = full[i_min[0]:i_max[0], i_min[1]:i_max[1], i_min[2]:i_max[2]].copy()
    crop_origin = grid_origin + i_min.astype(float) * pitch
    return VoxelGrid(data=cropped, origin=crop_origin, pitch=pitch)


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
    """Dla każdego voxela: odległość do najbliższego z `points` (mm)."""
    from scipy.spatial import cKDTree

    centers = template.voxel_centers()
    tree = cKDTree(points)
    dist, _ = tree.query(centers, k=1)
    return VoxelGrid(
        data=dist.reshape(template.shape).astype(np.float32),
        origin=template.origin.copy(),
        pitch=template.pitch,
    )


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
