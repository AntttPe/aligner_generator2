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
from scipy import ndimage as ndi

from .sdf import (
    VoxelGrid,
    binary_close,
    grid_memory_estimate,
    marching_cubes_mesh,
    sdf_from_binary,
    selection_proximity,
    voxelize_solid,
)


@dataclass
class AlignerParams:
    thickness: float = 1.0              # grubość ścianki nakładki [mm] (klinicznie 0.5–1.0; sztywniejsza nakładka = 1.0)
    inner_clearance: float = 0.05       # luz między zębem a wnętrzem nakładki [mm]
    close_radius: float = 1.5           # R closing dla bridgowania embrasur [mm]
    selection_radius: float = 3.0       # max odległość voxela od zaznaczonej powierzchni [mm]
    use_owner_check: bool = True        # voxel musi być bliżej zaznaczonej części mesha niż niezaznaczonej
    fillet_radius: float = 0.6          # zaokrąglenie krawędzi trim/owner (smooth-min k) [mm]
    trim_smooth_sigma: float = 5.0      # gauss na polach trim (owner/sel_dist) [voxele] — wygładza PRZEBIEG krawędzi (frędzle). NIE rusza sdf → fit inner surface zachowany
    voxel_pitch: float = 0.10           # rozdzielczość siatki voxelowej [mm] — 0.10 dla direct-print quality
    bbox_padding: float = 5.0           # dodatkowy margines bboxa [mm]
    field_smooth_sigma: float = 1.0     # gauss smoothing field-u przed MC (voxel units)
    fill_holes: bool = True             # trimesh.repair.fill_holes po MC (otwarte krawędzie)
    taubin_iters: int = 15              # Taubin smoothing wynikowego mesha (volume-preserving)
    taubin_lambda: float = 0.5
    taubin_nu: float = 0.53
    keep_largest_component: bool = True


def _smin(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Polynomial smooth minimum (Iñigo Quilez).

    Gdy `k=0` daje twardy min. Dla `k>0` gładko zaokrągla zbieg dwóch
    powierzchni iso-zerowych z efektywnym promieniem ≈ k/4.

    https://iquilezles.org/articles/smin/
    """
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


@dataclass
class GenerationReport:
    grid_shape: tuple[int, int, int] = (0, 0, 0)
    grid_memory_mb: int = 0
    voxelize_seconds: float = 0.0
    close_seconds: float = 0.0
    sdf_seconds: float = 0.0
    sel_distance_seconds: float = 0.0
    field_smooth_seconds: float = 0.0
    marching_seconds: float = 0.0
    smoothing_seconds: float = 0.0
    aligner_verts: int = 0
    aligner_faces: int = 0
    aligner_voxels: int = 0
    components_found: int = 0
    holes_filled: int = 0
    total_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    # S1 (support generation): per-vertex maska na mesh nakładki — True dla
    # vertices na krawędzi trim (cięcie sel/owner), False dla face. Używane
    # przez `supports.py` do umieszczania kotwic podpór dokładnie na krawędzi.
    rim_mask: np.ndarray | None = None


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
    try:
        binary = voxelize_solid(mesh, bbox_min, bbox_max, params.voxel_pitch)
    except MemoryError as e:
        rep.notes.append(str(e))
        print(f"[aligner] ABORT: {e}")
        return None, rep
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

    # ---- owner-based selection proximity ----
    t = time.time()
    print(
        f"[aligner] selection proximity (voxelize selected vs full surface)..."
    )
    sel_dist_data, owner = selection_proximity(mesh, selected_vertex_mask, sdf)
    rep.sel_distance_seconds = time.time() - t
    print(
        f"[aligner]   sel_dist range: [{float(sel_dist_data.min()):.2f}, "
        f"{float(sel_dist_data.max()):.2f}] mm, owner-voxels: "
        f"{int(owner.sum())} ({rep.sel_distance_seconds:.1f}s)"
    )

    # ---- TSDF-like field aligner z smooth-min (fillet na krawędziach) ----
    inner_iso = params.inner_clearance
    outer_iso = params.inner_clearance + params.thickness
    k = params.fillet_radius

    # Grubość ścianki (inner vs outer) — TWARDY min, żeby fillet NIE
    # ścieńczał ścianki (smin na tej parze przewęża środek ścianki do zera).
    wall = np.minimum(sdf.data - inner_iso, outer_iso - sdf.data)

    # --- pola definiujące TRIM (krawędź przy dziąśle) ---
    sel_field = params.selection_radius - sel_dist_data
    if params.use_owner_check:
        dist_in = ndi.distance_transform_edt(owner).astype(np.float32) * sdf.pitch
        dist_out = ndi.distance_transform_edt(~owner).astype(np.float32) * sdf.pitch
        owner_signed = (dist_in - dist_out).astype(np.float32)  # >0 wewn. owner
    else:
        owner_signed = None

    # Wygładź PRZEBIEG krawędzi: gauss na polach trim. To likwiduje frędzle
    # (jagged trim kopiujący ręczny kontur), a NIE rusza sdf → fit inner
    # surface zostaje precyzyjny.
    if params.trim_smooth_sigma > 0:
        t = time.time()
        sel_field = ndi.gaussian_filter(sel_field, sigma=params.trim_smooth_sigma)
        if owner_signed is not None:
            owner_signed = ndi.gaussian_filter(
                owner_signed, sigma=params.trim_smooth_sigma
            )
        print(
            f"[aligner]   trim smoothing σ={params.trim_smooth_sigma} voxeli "
            f"({time.time() - t:.1f}s)"
        )

    # Fillet (smooth-min) TYLKO na otwartych krawędziach: trim + owner cut.
    aligner_field_data = _smin(wall, sel_field, k)
    if owner_signed is not None:
        aligner_field_data = _smin(aligner_field_data, owner_signed, k)
    aligner_field_data = aligner_field_data.astype(np.float32)

    n_aligner_voxels = int((aligner_field_data > 0).sum())
    rep.aligner_voxels = n_aligner_voxels
    print(f"[aligner]   aligner voxels (field > 0): {n_aligner_voxels}")
    if n_aligner_voxels == 0:
        rep.notes.append(
            "0 voxeli nakładki — parametry zbyt restrykcyjne albo selekcja "
            "nie generuje sensownej powłoki."
        )
        return None, rep

    # ---- gauss smoothing field-u (eliminuje drobne ujemne wyspy → dziury) ----
    if params.field_smooth_sigma > 0:
        t = time.time()
        aligner_field_data = ndi.gaussian_filter(
            aligner_field_data, sigma=params.field_smooth_sigma
        ).astype(np.float32)
        rep.field_smooth_seconds = time.time() - t
        n_after_smooth = int((aligner_field_data > 0).sum())
        print(
            f"[aligner]   po gauss σ={params.field_smooth_sigma}: "
            f"{n_after_smooth} voxeli  ({rep.field_smooth_seconds:.1f}s)"
        )

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

    # ---- fill holes (otwarte krawędzie) ----
    if params.fill_holes:
        try:
            before_faces = len(mesh_out.faces)
            trimesh.repair.fill_holes(mesh_out)
            after_faces = len(mesh_out.faces)
            rep.holes_filled = max(0, after_faces - before_faces)
            if rep.holes_filled > 0:
                print(f"[aligner]   fill_holes: dodano {rep.holes_filled} triangles")
        except Exception as e:
            print(f"[aligner]   fill_holes warning: {e}")

    # ---- Taubin smoothing (volume-preserving, likwiduje schodkowanie) ----
    if params.taubin_iters > 0:
        t = time.time()
        trimesh.smoothing.filter_taubin(
            mesh_out,
            lamb=params.taubin_lambda,
            nu=params.taubin_nu,
            iterations=params.taubin_iters,
        )
        rep.smoothing_seconds = time.time() - t
        print(
            f"[aligner]   Taubin smoothing ({params.taubin_iters} iter): "
            f"{rep.smoothing_seconds:.1f}s"
        )

    rep.aligner_verts = len(mesh_out.vertices)
    rep.aligner_faces = len(mesh_out.faces)

    # ---- S1: rim detection (krawędź trim dla podpór druku) ----
    # Trilinear-sample pól komponentowych na finalnych vertices nakładki.
    # Vertex jest "rim" gdy sel/owner wiąże (nie wall) → cięcie wzdłuż gingival.
    try:
        from .supports import compute_rim_mask
        t = time.time()
        rep.rim_mask = compute_rim_mask(
            np.asarray(mesh_out.vertices, dtype=float),
            sdf,
            wall,
            sel_field,
            owner_signed,
        )
        n_rim = int(rep.rim_mask.sum())
        print(
            f"[aligner]   rim detection: {n_rim}/{rep.aligner_verts} verts "
            f"({100.0 * n_rim / max(rep.aligner_verts, 1):.1f}%) "
            f"({time.time() - t:.2f}s)"
        )
    except Exception as e:
        print(f"[aligner]   rim detection warning: {e}")
        rep.rim_mask = None

    rep.total_seconds = time.time() - t_total
    print(
        f"[aligner] GOTOWE w {rep.total_seconds:.1f}s — "
        f"watertight={mesh_out.is_watertight}, "
        f"verts={rep.aligner_verts}, faces={rep.aligner_faces}"
    )
    return mesh_out, rep
