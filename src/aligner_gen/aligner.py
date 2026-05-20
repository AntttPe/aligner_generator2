"""Pipeline generowania nakładki: mesh + selekcja → mesh nakładki (STL ready).

Etapy:
  1. Bbox z padding wokół zaznaczonej powierzchni
  2. Voxelizacja całego mesha → binary (inside teeth)
  3. Morphological closing → wypełnia embrasury
  4. SDF z closed binary
  5. Pole odległości od zaznaczonych wierzchołków (restrict do selekcji)
  6. TSDF-like field aligner: min(sdf - inner_iso, outer_iso - sdf, sel_radius - sel_dist)
  7. Marching cubes → mesh nakładki
  8. Post-process: largest component, opcjonalny smoothing
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import trimesh

from .sdf import (
    VoxelGrid,
    binary_close,
    distance_to_points,
    grid_memory_estimate,
    marching_cubes_mesh,
    sdf_from_binary,
    voxelize_solid,
)


@dataclass
class AlignerParams:
    thickness: float = 0.6           # grubość ścianki nakładki [mm]
    inner_clearance: float = 0.05    # luz między zębem a wnętrzem nakładki [mm]
    close_radius: float = 1.5        # R closing dla bridgowania embrasur [mm]
    selection_radius: float = 1.5    # softness trim line — extent poza zaznaczone vertices [mm]
    voxel_pitch: float = 0.15        # rozdzielczość siatki voxelowej [mm]
    bbox_padding: float = 5.0        # dodatkowy margines bboxa [mm]
    keep_largest_component: bool = True


@dataclass
class GenerationReport:
    grid_shape: tuple[int, int, int] = (0, 0, 0)
    grid_memory_mb: int = 0
    voxelize_seconds: float = 0.0
    close_seconds: float = 0.0
    sdf_seconds: float = 0.0
    sel_distance_seconds: float = 0.0
    marching_seconds: float = 0.0
    aligner_verts: int = 0
    aligner_faces: int = 0
    aligner_voxels: int = 0
    components_found: int = 0
    total_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def generate_aligner(
    mesh: trimesh.Trimesh,
    selected_vertex_mask: np.ndarray,
    params: AlignerParams | None = None,
) -> tuple[trimesh.Trimesh | None, GenerationReport]:
    """Pipeline mesh + selekcja → mesh nakładki."""
    if params is None:
        params = AlignerParams()
    rep = GenerationReport()
    t_total = time.time()

    if not selected_vertex_mask.any():
        rep.notes.append("Pusta selekcja — nic do wygenerowania.")
        return None, rep

    selected_pts = np.asarray(mesh.vertices[selected_vertex_mask], dtype=float)

    # ---- bbox ----
    padding = (
        params.bbox_padding
        + params.thickness
        + 2.0 * params.close_radius
    )
    bbox_min = selected_pts.min(axis=0) - padding
    bbox_max = selected_pts.max(axis=0) + padding
    shape_est, mem_mb = grid_memory_estimate(
        bbox_min, bbox_max, params.voxel_pitch, bytes_per_voxel=4
    )
    rep.grid_shape = shape_est
    rep.grid_memory_mb = mem_mb
    print(
        f"[aligner] bbox: {bbox_max - bbox_min} mm  →  grid {shape_est}  "
        f"≈ {mem_mb} MB / float array"
    )

    # ---- voxelizacja ----
    t = time.time()
    print(f"[aligner] voxelizuję (pitch={params.voxel_pitch} mm)...")
    binary = voxelize_solid(mesh, bbox_min, bbox_max, params.voxel_pitch)
    rep.voxelize_seconds = time.time() - t
    rep.grid_shape = tuple(binary.shape)
    inside_count = int(binary.data.sum())
    print(
        f"[aligner]   grid: {binary.shape}, inside: {inside_count} / "
        f"{binary.data.size}  ({rep.voxelize_seconds:.1f}s)"
    )
    if inside_count == 0:
        rep.notes.append(
            "Po voxelizacji 0 voxeli wewnątrz — sprawdź czy mesh jest watertight "
            "i czy bbox pokrywa zaznaczoną powierzchnię."
        )
        return None, rep

    # ---- closing (embrasure bridging) ----
    t = time.time()
    print(f"[aligner] closing (R={params.close_radius} mm)...")
    binary_closed = binary_close(binary, params.close_radius)
    rep.close_seconds = time.time() - t
    inside_after = int(binary_closed.data.sum())
    delta = inside_after - inside_count
    print(
        f"[aligner]   inside po close: {inside_after}  "
        f"(Δ={delta:+d}, {rep.close_seconds:.1f}s)"
    )

    # ---- SDF ----
    t = time.time()
    print("[aligner] SDF z closed binary...")
    sdf = sdf_from_binary(binary_closed)
    rep.sdf_seconds = time.time() - t
    print(
        f"[aligner]   SDF range: [{float(sdf.data.min()):.2f}, "
        f"{float(sdf.data.max()):.2f}] mm  ({rep.sdf_seconds:.1f}s)"
    )

    # ---- pole odległości od selekcji ----
    t = time.time()
    print(f"[aligner] distance do selekcji ({len(selected_pts)} pkt)...")
    sel_dist = distance_to_points(sdf, selected_pts)
    rep.sel_distance_seconds = time.time() - t
    print(
        f"[aligner]   sel_dist range: [{float(sel_dist.data.min()):.2f}, "
        f"{float(sel_dist.data.max()):.2f}] mm  ({rep.sel_distance_seconds:.1f}s)"
    )

    # ---- TSDF-like field aligner ----
    inner_iso = params.inner_clearance
    outer_iso = params.inner_clearance + params.thickness
    aligner_field_data = np.minimum.reduce(
        [
            sdf.data - inner_iso,                       # ≥0 gdy na zewnątrz wnętrza
            outer_iso - sdf.data,                       # ≥0 gdy wewnątrz zewnętrza
            params.selection_radius - sel_dist.data,    # ≥0 gdy w obszarze selekcji
        ]
    ).astype(np.float32)
    n_aligner_voxels = int((aligner_field_data > 0).sum())
    rep.aligner_voxels = n_aligner_voxels
    print(f"[aligner]   aligner voxels (field > 0): {n_aligner_voxels}")
    if n_aligner_voxels == 0:
        rep.notes.append(
            "0 voxeli nakładki — parametry zbyt restrykcyjne albo selekcja "
            "nie generuje sensownej powłoki."
        )
        return None, rep

    # padding "negatywnymi" wartościami żeby marching cubes domykał na brzegach gridu
    padded = np.pad(
        aligner_field_data,
        pad_width=1,
        mode="constant",
        constant_values=-1.0,
    )

    aligner_field = VoxelGrid(
        data=padded,
        origin=sdf.origin - np.array([sdf.pitch] * 3),
        pitch=sdf.pitch,
    )

    # ---- marching cubes ----
    t = time.time()
    print("[aligner] marching cubes...")
    try:
        mesh_out = marching_cubes_mesh(
            aligner_field, level=0.0, gradient_direction="descent"
        )
    except ValueError as e:
        rep.notes.append(f"marching_cubes ValueError: {e}")
        return None, rep
    rep.marching_seconds = time.time() - t
    print(
        f"[aligner]   mesh: {len(mesh_out.vertices)} verts, "
        f"{len(mesh_out.faces)} faces  ({rep.marching_seconds:.1f}s)"
    )

    # ---- post-process: największa komponenta ----
    if params.keep_largest_component:
        comps = mesh_out.split(only_watertight=False)
        rep.components_found = len(comps)
        if len(comps) > 1:
            sizes = [len(c.vertices) for c in comps]
            biggest_idx = int(np.argmax(sizes))
            mesh_out = comps[biggest_idx]
            print(
                f"[aligner]   znaleziono {len(comps)} komponent, wybrano "
                f"największą ({len(mesh_out.vertices)} verts)"
            )

    rep.aligner_verts = len(mesh_out.vertices)
    rep.aligner_faces = len(mesh_out.faces)
    rep.total_seconds = time.time() - t_total
    print(
        f"[aligner] GOTOWE w {rep.total_seconds:.1f}s — "
        f"watertight={mesh_out.is_watertight}, "
        f"verts={rep.aligner_verts}, faces={rep.aligner_faces}"
    )
    return mesh_out, rep
