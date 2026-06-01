"""Moduł generowania podpór druku 3D dla nakładek.

Zakres:
  S1 ✅ rim detection — wall vs sel_field/owner constraint binding check.
  S2 ✅ apex line + sampling — sub-mesh outline (boundary loops) + arc-length.
  S3 (TEN PLIK ↓): cone-tip + cylinder body pillars, concat z aligner.
  S4 (TODO): perforowany raft z grawerem ID pacjenta + nr kroku.
  S5 (TODO): auto-orient PCA (rim → -Z), collision check.
  S6 (TODO): dock panel w qt_viewer (rozszerzony), export combined STL.

Specyfikacja klinicza: [[edge-support-generation]] w pamięci projektu.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.ndimage import map_coordinates

from .sdf import VoxelGrid


def compute_rim_mask(
    mesh_verts: np.ndarray,
    sdf_grid: VoxelGrid,
    wall: np.ndarray,
    sel_field: np.ndarray,
    owner_signed: np.ndarray | None,
) -> np.ndarray:
    """Per-vertex mask na meshu nakładki: True = krawędź trim, False = face.

    Idea: aligner_field = min(wall, sel_field, owner_signed). Vertex leży na
    surface gdzie min ≈ 0. Który człon jest wiążący, to determinuje typ
    powierzchni w tym miejscu:
      - wall wiążący (najmniejszy) → face inner lub outer (powierzchnia gładka),
      - sel_field/owner wiążący → krawędź trim (cięcie wzdłuż gingival margin).

    To jest "uprzywilejowana" detekcja — generic slicer dostaje gotowy STL i
    musi zgadywać, my robimy mesh i znamy dokładnie wszystkie pola.

    Parametry
    ---------
    mesh_verts : (N, 3) world coords vertices nakładki (po Taubin smoothing)
    sdf_grid   : VoxelGrid pierwotny SDF zębów (dostarcza origin/pitch dla
                 mapowania world→voxel; wall/sel/owner są w tej samej siatce)
    wall, sel_field, owner_signed : 3D ndarray pól komponentowych. owner_signed
                 może być None gdy use_owner_check=False.

    Zwraca
    ------
    rim_mask : (N,) bool — True dla vertices na krawędzi trim.

    Notes
    -----
    Vertices nakładki są lekko przesunięte po Taubin smoothing (off-iso o sub-mm),
    ale relatywne porównanie pól (które jest najmniejsze) zostaje stabilne —
    trilinearna interpolacja daje sensowną odpowiedź.
    """
    origin = np.asarray(sdf_grid.origin, dtype=float)
    pitch = float(sdf_grid.pitch)

    # world → voxel index (float). map_coordinates oczekuje shape (3, N).
    ijk = ((np.asarray(mesh_verts, dtype=float) - origin) / pitch).T

    wall_v = map_coordinates(wall, ijk, order=1, mode="nearest")
    sel_v = map_coordinates(sel_field, ijk, order=1, mode="nearest")

    if owner_signed is not None:
        owner_v = map_coordinates(owner_signed, ijk, order=1, mode="nearest")
        trim_v = np.minimum(sel_v, owner_v)
    else:
        trim_v = sel_v

    # rim: vertex leży na krawędzi cięcia trim (sel/owner najmniejsze).
    # Dodajemy mały epsilon żeby uniknąć granicznych przypadków numerycznych.
    rim_mask = trim_v < (wall_v - 1e-4)
    return rim_mask.astype(bool)


# ===========================================================================
# S2: apex line + sampling — kotwice podpór wzdłuż grzbietu filletu
# ===========================================================================

def _erode_rim_to_medial(
    aligner_mesh,
    rim_mask: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Skeletonizacja pasma rim → cienka 1D medial line środkiem pasma.

    Algorytm (klasyczna morphological skeleton z distance transform):
      1. **Multi-source BFS** od non-rim vertices w mesh adjacency.
         `d[v]` = liczba hopów od najbliższego non-rim vert.
         d == 0 dla non-rim; d > 0 dla rim (rośnie w głąb pasma).
      2. **Lokalne maksimum** d wśród rim: vertex v jest medial jeśli żaden
         jego mesh-sąsiad nie ma większego d. Te leżą **dokładnie w środku
         pasma w 3D** — ridge skeleton.
      3. Fallback gdy plateau (cienkie pasmo, max_d=1): weź wszystkie
         z d == max_d.

    Poprzednia "naive erosion" zawodziła bo oznaczała vert jako border przy
    JAKIMKOLWIEK non-rim sąsiedzie. Na triangle mesh każdy vert pasma ma
    sąsiadów po obu stronach (inner+outer face) → 1 iter wymazywała wszystko
    oprócz losowego centrum.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    eu = np.asarray(aligner_mesh.edges_unique, dtype=np.int64)
    n = len(aligner_mesh.vertices)

    # mesh adjacency (symmetric, binary)
    row = np.concatenate([eu[:, 0], eu[:, 1]])
    col = np.concatenate([eu[:, 1], eu[:, 0]])
    data = np.ones(len(row), dtype=np.float32)
    G = csr_matrix((data, (row, col)), shape=(n, n))

    non_rim_idx = np.where(~rim_mask)[0]
    if non_rim_idx.size == 0:
        return rim_mask.copy()

    # multi-source BFS: d[v] = hops od najbliższego non-rim
    d = dijkstra(G, indices=non_rim_idx, min_only=True, unweighted=True)
    d = np.where(np.isfinite(d), d, 0).astype(np.float32)
    d_rim = d[rim_mask]
    if d_rim.size == 0:
        return rim_mask.copy()
    max_d = int(d_rim.max())

    if max_d == 0:
        return rim_mask.copy()

    # local max: vertex v jest medial gdy żaden mesh-sąsiad nie ma d > d[v]
    larger_neighbor = np.zeros(n, dtype=bool)
    larger_neighbor[eu[d[eu[:, 1]] > d[eu[:, 0]], 0]] = True
    larger_neighbor[eu[d[eu[:, 0]] > d[eu[:, 1]], 1]] = True
    medial = rim_mask & ~larger_neighbor & (d > 0)

    # fallback: cienkie pasmo (max_d=1) → plateau, weź wszystkie z d==max_d
    if int(medial.sum()) < 20:
        medial = rim_mask & (d == max_d)
    if int(medial.sum()) < 8:
        medial = rim_mask & (d >= max(1, max_d - 1))
    if int(medial.sum()) < 3:
        medial = rim_mask.copy()

    if verbose:
        print(
            f"[supports] _erode: max_d={max_d}, medial={int(medial.sum())} "
            f"verts ({100.0 * int(medial.sum()) / max(int(rim_mask.sum()),1):.1f}% rim)"
        )
    return medial


def _traverse_loop_through_graph(
    aligner_mesh,
    medial_mask: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Zbuduj adjacency na medial vertices, wybierz **largest connected
    component** i traverse → uporządkowana polyline.

    Strategia:
      - subgraph mesh adjacency restricted to medial verts,
      - connected_components → wybierz największą (medial może mieć rozłączne
        kawałki przy szumie),
      - start z endpoint (degree=1) jeśli open path, dowolny gdy cycle,
      - greedy walk po nieodwiedzonych sąsiadach.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    eu = np.asarray(aligner_mesh.edges_unique, dtype=np.int64)
    v0, v1 = eu[:, 0], eu[:, 1]
    both = medial_mask[v0] & medial_mask[v1]
    medial_edges = eu[both]
    medial_idx = np.where(medial_mask)[0]
    if len(medial_idx) < 3:
        if verbose:
            print(f"[supports] _traverse: medial za mały ({len(medial_idx)} verts)")
        return np.zeros((0, 3), dtype=float)

    # mapowanie global → local (0..M-1)
    pos_map = np.full(len(aligner_mesh.vertices), -1, dtype=np.int64)
    pos_map[medial_idx] = np.arange(len(medial_idx))
    M = len(medial_idx)

    # local-index edges
    ed_a = pos_map[medial_edges[:, 0]]
    ed_b = pos_map[medial_edges[:, 1]]

    # connected components na medial subgraph
    if len(ed_a) > 0:
        sub_rows = np.concatenate([ed_a, ed_b])
        sub_cols = np.concatenate([ed_b, ed_a])
        sub_data = np.ones(len(sub_rows), dtype=np.float32)
        sub = csr_matrix((sub_data, (sub_rows, sub_cols)), shape=(M, M))
    else:
        sub = csr_matrix((M, M))
    n_comp, labels = connected_components(sub, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    biggest = int(np.argmax(sizes))
    in_biggest = labels == biggest
    L = int(in_biggest.sum())
    if verbose:
        print(
            f"[supports] _traverse: medial {M}v, {n_comp} comp(s), "
            f"largest={L}v ({100.0*L/M:.0f}%)"
        )
    if L < 3:
        return np.zeros((0, 3), dtype=float)

    # adjacency w obrębie biggest component (local indeksy 0..L-1)
    keep_local = np.where(in_biggest)[0]
    local_remap = np.full(M, -1, dtype=np.int64)
    local_remap[keep_local] = np.arange(L)
    keep_edges = in_biggest[ed_a] & in_biggest[ed_b]
    ea = local_remap[ed_a[keep_edges]]
    eb = local_remap[ed_b[keep_edges]]

    adj: list[list[int]] = [[] for _ in range(L)]
    for a, b in zip(ea, eb):
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    degrees = np.array([len(x) for x in adj], dtype=np.int32)
    endpoints = np.where(degrees == 1)[0]
    start = int(endpoints[0]) if endpoints.size > 0 else 0

    visited = np.zeros(L, dtype=bool)
    order: list[int] = [start]
    visited[start] = True
    cur = start
    while True:
        next_v = -1
        for nb in adj[cur]:
            if not visited[nb]:
                next_v = nb
                break
        if next_v < 0:
            break
        order.append(next_v)
        visited[next_v] = True
        cur = next_v

    if len(order) < 3:
        return np.zeros((0, 3), dtype=float)

    # local biggest → global mesh idx
    global_idx = medial_idx[keep_local[np.asarray(order, dtype=np.int64)]]
    return np.asarray(aligner_mesh.vertices, dtype=float)[global_idx]


def extract_apex_loop(
    aligner_mesh,
    rim_mask: np.ndarray,
    verbose: bool = True,
) -> np.ndarray:
    """Z rim band wyciągnij **apex polyline** — uporządkowaną zamkniętą krzywą
    1D wzdłuż krawędzi trim nakładki.

    Insight: rim_mask = powierzchnia 2D **side wall** (boczna ścianka shell-a
    łącząca inner face z outer face). Topologicznie to cylinder → ma DWIE
    zamknięte pętle brzegowe:
      - outer trim edge (gdzie side wall styka się z outer face),
      - inner trim edge (gdzie side wall styka się z inner face).

    Wykorzystujemy `trimesh.Trimesh.outline()` na sub-meshu z **rim faces**
    (face gdzie wszystkie 3 verts są rim) — wyciąga te boundary loops bardzo
    czysto bez morphologicznej skeletyzacji (która zawodziła na trójkątnym
    paśmie o nierównomiernym width — plateau d_max fragmentowało się na
    setki rozłącznych komponentów).

    Wynik: longest closed loop = zwykle outer trim edge (lekko poniżej
    geometrycznego apex zaokrąglenia, ale wystarczająco blisko dla
    umieszczenia kotwic podpór). Refine do prawdziwego 3D apex w S3+.

    Parametry
    ---------
    aligner_mesh : trimesh.Trimesh nakładki
    rim_mask     : (V,) bool — per-vertex rim flag (z compute_rim_mask)

    Zwraca
    ------
    apex_pts : (M, 3) — uporządkowane wierzchołki zamkniętej pętli (bez
               powtórzonego ostatniego punktu). Pusta przy patologii.
    """
    import trimesh as _tm

    if int(rim_mask.sum()) < 10:
        return np.zeros((0, 3), dtype=float)

    # 1) Sub-mesh: tylko twarze gdzie wszystkie 3 verts są rim
    faces = np.asarray(aligner_mesh.faces, dtype=np.int64)
    rim_face_mask = rim_mask[faces].all(axis=1)
    n_rim_faces = int(rim_face_mask.sum())
    if n_rim_faces == 0:
        if verbose:
            print("[supports] outline: 0 rim faces (rim verts nie tworzą twarzy)")
        return np.zeros((0, 3), dtype=float)

    sub = _tm.Trimesh(
        vertices=np.asarray(aligner_mesh.vertices),
        faces=faces[rim_face_mask],
        process=False,
    )

    # 2) Boundary edges sub-mesha → outline (Path3D z entitiamI = zamknięte pętle)
    try:
        outline = sub.outline()
    except Exception as e:  # pragma: no cover
        if verbose:
            print(f"[supports] outline error: {e}")
        return np.zeros((0, 3), dtype=float)

    if outline is None or not hasattr(outline, "entities"):
        return np.zeros((0, 3), dtype=float)

    # 3) Iteruj entity, wybierz najdłuższą zamkniętą pętlę (= zwykle outer trim)
    best_pts: np.ndarray | None = None
    best_len = 0.0
    n_loops = 0
    n_open = 0
    out_verts = np.asarray(outline.vertices, dtype=float)
    for ent in outline.entities:
        try:
            pts = out_verts[np.asarray(ent.points, dtype=np.int64)]
        except Exception:
            continue
        if len(pts) < 3:
            continue
        # closed loop: ostatni punkt = pierwszy
        is_closed = bool(np.allclose(pts[0], pts[-1]))
        if not is_closed:
            n_open += 1
            continue
        n_loops += 1
        diffs = np.diff(pts, axis=0)
        L = float(np.linalg.norm(diffs, axis=1).sum())
        if L > best_len:
            best_len = L
            best_pts = pts[:-1]  # drop dup last pt

    if verbose:
        print(
            f"[supports] outline: rim_faces={n_rim_faces}, "
            f"closed loops={n_loops}, open={n_open}, "
            f"best perimeter={best_len:.1f}mm, "
            f"best verts={len(best_pts) if best_pts is not None else 0}"
        )

    if best_pts is None or len(best_pts) < 3:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(best_pts, dtype=float)


def sample_apex_loop(
    apex_pts: np.ndarray,
    spacing_mm: float = 3.0,
) -> np.ndarray:
    """Resample apex polyline w równomiernym arc-length spacing (closed loop).

    Buduje cumulative arc length zamkniętej polyline, kładzie próbki co
    `spacing_mm` wzdłuż obwodu. Liniowa interpolacja między sąsiednimi
    apex points.

    Zwraca (N, 3) gdzie N = round(perimeter / spacing_mm), min 8.
    """
    if len(apex_pts) < 3:
        return apex_pts.copy()
    closed = np.vstack([apex_pts, apex_pts[:1]])
    diffs = np.diff(closed, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total = float(seg_lengths.sum())
    if total < spacing_mm * 2:
        return apex_pts.copy()
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    n = max(8, int(round(total / spacing_mm)))
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 3), dtype=float)
    for i, s in enumerate(targets):
        j = int(np.searchsorted(arc, s) - 1)
        j = max(0, min(j, len(seg_lengths) - 1))
        denom = max(arc[j + 1] - arc[j], 1e-9)
        t = (s - arc[j]) / denom
        out[i] = closed[j] + t * (closed[j + 1] - closed[j])
    return out


def detect_rim_local_minima(
    apex_pts: np.ndarray,
    transform: np.ndarray,
    existing_anchors_print: np.ndarray | None = None,
    xy_radius_mm: float = 2.0,
    min_dip_mm: float = 0.05,
    dedupe_dist_mm: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """Znajdź punkty rim które są **local Z-minima w XY-neighborhood**.

    PreForm "Unsupported Minima Detected" = punkt rim który jest najniżej
    w print Z spośród wszystkich sąsiadów w pobliżu w XY → drukuje pierwszy
    bez nic poniżej. Wymaga bezpośredniej podpory.

    Algorytm:
      1. Transformuj apex_pts do print space (XY = build plane, Z = print).
      2. Per apex point: znajdź XY-sąsiadów w promieniu `xy_radius_mm`.
      3. Jeśli mam najniższy Z spośród sąsiadów I peak nad mną ≥ `min_dip_mm`
         → jestem unsupported minimum.
      4. Dedupe vs `existing_anchors_print` (poniżej dist ⇒ pokryty przez
         pobliski pillar; powyżej ⇒ standalone unsupported minimum).

    Wcześniejszy bug: porównywałem do MEAN sąsiedztwa wzdłuż arc length —
    w wąskim oknie mean ≈ z_i, dip depth zawsze ~0. XY-neighbor MAX porównanie
    matchuje semantikę PreForm.
    """
    if apex_pts is None or len(apex_pts) < 8:
        return np.zeros((0, 3), dtype=float)
    from scipy.spatial import cKDTree

    pts = (np.asarray(transform, dtype=float)[:3, :3] @ apex_pts.T).T \
        + np.asarray(transform, dtype=float)[:3, 3]
    n = len(pts)
    tree = cKDTree(pts[:, :2])

    minima_idx: list[int] = []
    for i in range(n):
        neighbors = tree.query_ball_point(pts[i, :2], xy_radius_mm)
        if len(neighbors) < 3:
            continue
        z_neighbors = pts[neighbors, 2]
        z_i = pts[i, 2]
        if z_i <= z_neighbors.min() + 1e-6 and (z_neighbors.max() - z_i) >= min_dip_mm:
            minima_idx.append(i)

    if not minima_idx:
        if verbose:
            print(
                f"[supports] rim minima: 0 znalezionych "
                f"(xy_radius={xy_radius_mm}mm, dip≥{min_dip_mm}mm)"
            )
        return np.zeros((0, 3), dtype=float)

    candidates = pts[np.asarray(minima_idx, dtype=np.int64)]

    # Dedupe vs istniejące pillary — minimum w odległości ≤dedupe od pillara
    # jest "pokryte" jego stożkiem (tip radius ~0.2mm + cot(angle) tolerance).
    if existing_anchors_print is not None and len(existing_anchors_print) > 0:
        d, _ = cKDTree(existing_anchors_print[:, :2]).query(candidates[:, :2], k=1)
        candidates = candidates[d > dedupe_dist_mm]

    # Dedupe wewnętrzny (klastry minimów blisko siebie → jeden)
    if len(candidates) > 1:
        keys = np.floor(candidates[:, :2] / max(dedupe_dist_mm, 0.5)).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        candidates = candidates[idx]

    if verbose:
        print(
            f"[supports] rim minima: {len(minima_idx)} surowych → "
            f"{len(candidates)} po dedupe (xy_radius={xy_radius_mm}mm, "
            f"dip≥{min_dip_mm}mm, dedupe≥{dedupe_dist_mm}mm)"
        )
    return candidates


# ===========================================================================
# S3: pillar geometry — cone-tip + cylinder body per kotwica
# ===========================================================================

@dataclass
class PillarParams:
    """Parametry pojedynczej podpory druku.

    Kliniczne wytyczne (z [[edge-support-generation]]):
      - tip kontaktowy mały (~0.3–0.5mm) → snap-off bez śladu na fillet,
      - tip wysokości ~1mm żeby zwęzić od body do contact point,
      - body 0.8–1.2mm — wystarczająco wytrzymały na siły print,
      - **tip_penetration**: tip wbija się w part o ~0.1mm → pewny overlap
        (PreForm bez tego liczy support jako "ungrounded" → flagi
        "unsupported minima"). Po snap-off zostaje sub-mm nub.
      - długość pillara wyznacza odległość apex → top raftu (S4); na razie
        stała wartość do podglądu geometrii w S3.
    """
    tip_diameter: float = 0.4        # mm — średnica kontaktu (snap-off interface)
    tip_height: float = 1.0          # mm — wysokość zwężenia od body do tip
    body_diameter: float = 1.0       # mm — średnica trzonu
    tip_penetration: float = 0.12    # mm — overlap z part (anti "ungrounded")
    use_ball_tip: bool = True        # sfera na kontakcie zamiast krążka:
                                     # uniform 360° kontakt na zakrzywionym
                                     # rim (cone-flat-circle traci kontakt
                                     # gdy rim się odchyla). Plus sphere-cone
                                     # joint = naturalny stress riser dla
                                     # clean snap-off (Formlabs-style).
    pillar_height: float = 8.0   # mm — całkowita długość pillara (S4 ustawi do raftu)
    n_sides: int = 8             # ilość boków cylindra/stożka (8 = wystarczająco gładkie)


def compute_pillar_direction(
    aligner_mesh,
    apex_loop: np.ndarray,
) -> np.ndarray:
    """Kierunek "w dół" pillarów: od apex, prostopadle do płaszczyzny trim,
    w stronę PRZECIWNĄ do centroidu nakładki.

    PCA na apex_loop → normal najmniejszej wariancji = normalna płaszczyzny.
    Orient: jeśli wskazuje ku środkowi aligner → flip. Wynik: pillary rosną
    od apex w "outside" kierunku, czyli ku platformie druku (S5 zorientuje
    nakładkę żeby to było -Z).
    """
    if len(apex_loop) < 3:
        return np.array([0.0, 0.0, -1.0])
    apex_c = apex_loop.mean(axis=0)
    centered = apex_loop - apex_c
    cov = (centered.T @ centered) / max(len(centered), 1)
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    normal = normal / max(np.linalg.norm(normal), 1e-9)
    aligner_c = np.asarray(aligner_mesh.vertices, dtype=float).mean(axis=0)
    # orient: normal wskazuje OD aligner centroid (outward)
    if float(np.dot(normal, apex_c - aligner_c)) < 0:
        normal = -normal
    return normal.astype(float)


def _make_pillar(
    top_pt: np.ndarray,
    dir_down: np.ndarray,
    params: PillarParams,
    height: float | None = None,
) -> trimesh.Trimesh:
    """Pojedynczy pillar: stożek (tip→body) sklejony z cylindrem (body→bottom).

    Geometria:
      ring0 (top, tip diameter)  ─── kontakt z apex (tu nub po snap-off)
        ↓ cone tapering
      ring1 (po `tip_height`, body diameter)
        ↓ cylindrical body
      ring2 (po `height`, body diameter) ─── do platformy / raftu

    `height` (None → params.pillar_height) pozwala na zmienną długość per
    pillar (S5: każda noga ma inną długość — do płaszczyzny Z=0 raftu).

    Caps na top (mały krążek z tip diameter) i bottom (krążek z body diameter)
    → manifold closed solid. Konkatenowane (NIE boolean union — slicer łyka
    multi-solid STL).
    """
    total_h = float(params.pillar_height if height is None else height)
    d = np.asarray(dir_down, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-9)

    # baza prostopadła do d
    helper = np.array([1.0, 0.0, 0.0])
    if abs(d[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, d) * d
    u = u / max(np.linalg.norm(u), 1e-9)
    w = np.cross(d, u)

    n = int(params.n_sides)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)
    r_top = params.tip_diameter / 2.0
    r_bot = params.body_diameter / 2.0

    top_pt = np.asarray(top_pt, dtype=float)
    tip_pen = float(getattr(params, "tip_penetration", 0.0))
    use_ball = bool(getattr(params, "use_ball_tip", False))

    # ring0 (gdzie cone się "kończy" na górze):
    #   ball-tip: w top_pt level, radius = r_ball (= r_top) → SHARED z hemisphere
    #   cone-only: 0.12mm INTO part (penetration guarantee)
    if use_ball:
        ring0_anchor = top_pt
    else:
        ring0_anchor = top_pt - tip_pen * d

    p1 = top_pt + params.tip_height * d
    p2 = top_pt + total_h * d

    ring0 = ring0_anchor + r_top * (ca[:, None] * u + sa[:, None] * w)
    ring1 = p1 + r_bot * (ca[:, None] * u + sa[:, None] * w)
    ring2 = p2 + r_bot * (ca[:, None] * u + sa[:, None] * w)

    # === Wersja A: ball-tip (manifold hemisphere SHARES ring0 with cone) ===
    if use_ball:
        # Hemisphere top: ring_mid (theta=45°) + pole (theta=90°) → smooth dome
        r_ball = r_top
        cos45, sin45 = np.cos(np.radians(45.0)), np.sin(np.radians(45.0))
        ring_mid_center = ring0_anchor - r_ball * sin45 * d   # góra w print Z
        ring_mid_radius = r_ball * cos45
        ring_mid = (
            ring_mid_center + ring_mid_radius * (ca[:, None] * u + sa[:, None] * w)
        )
        pole_pt = ring0_anchor - r_ball * d                    # samo top sphere

        # verts layout: [ring0(n), ring1(n), ring2(n), ring_mid(n), pole(1), bot_center(1)]
        verts = np.vstack([
            ring0, ring1, ring2, ring_mid,
            pole_pt[None, :], p2[None, :],
        ])
        ring_mid_off = 3 * n
        pole_idx = 4 * n
        bot_center_idx = 4 * n + 1
    else:
        # verts: [ring0(n), ring1(n), ring2(n), top_center(1), bot_center(1)]
        verts = np.vstack([ring0, ring1, ring2, ring0_anchor[None, :], p2[None, :]])
        top_center_idx = 3 * n
        bot_center_idx = 3 * n + 1

    faces: list[list[int]] = []
    # side cone (ring0 → ring1) — wspólne dla obu wariantów
    for i in range(n):
        j = (i + 1) % n
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    # side cylinder (ring1 → ring2)
    for i in range(n):
        j = (i + 1) % n
        faces.append([n + i, n + j, 2 * n + j])
        faces.append([n + i, 2 * n + j, 2 * n + i])

    if use_ball:
        # Hemisphere quads: ring0 → ring_mid (lower latitude)
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, ring_mid_off + j])
            faces.append([i, ring_mid_off + j, ring_mid_off + i])
        # Hemisphere triangle fan: ring_mid → pole
        for i in range(n):
            j = (i + 1) % n
            faces.append([ring_mid_off + i, ring_mid_off + j, pole_idx])
    else:
        # Flat top cap (krążek z tip_diameter)
        for i in range(n):
            j = (i + 1) % n
            faces.append([top_center_idx, i, j])

    # bottom cap (większy krążek, facing dir_down)
    for i in range(n):
        j = (i + 1) % n
        faces.append([bot_center_idx, 2 * n + j, 2 * n + i])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    try:
        trimesh.repair.fix_normals(mesh)
    except Exception:
        pass
    return mesh


def generate_pillars(
    anchors: np.ndarray,
    direction: np.ndarray,
    params: PillarParams | None = None,
) -> trimesh.Trimesh | None:
    """Build pillar mesh per kotwica i konkatenuj w jeden Trimesh.

    Konkatenacja (NIE boolean union):
      - drukarki łyka STL z wieloma rozłącznymi bryłami,
      - każdy pillar = osobny manifold solid w tym samym pliku,
      - bezpieczniejsze i szybsze niż boolean (który potrafi się wywalić).

    Zwraca trimesh.Trimesh albo None gdy brak anchors.
    """
    if anchors is None or len(anchors) == 0:
        return None
    if params is None:
        params = PillarParams()
    meshes = [_make_pillar(a, direction, params) for a in anchors]
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


# ===========================================================================
# S5: front/back detection + print orientation + pillars-to-plane
# ===========================================================================

def detect_anterior_posterior(
    aligner_mesh,
    rim_mask: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Wykryj ramę łuku: oś lewo-prawo, przód-tył (→anterior), occlusal normal.

    Łuk to podkowa: **otwarcie (przerwa między molarami) jest z TYŁU**
    (posterior), a przód (siekacze 11/21) to zamknięty zakrzywiony koniec.
    Detekcja AP przez midline-gap: na linii środkowej (lr≈0) jest materiał z
    przodu (siekacze), ale NIE z tyłu (przerwa).

    **Orientacja occlusal (up/down):** PCA daje normalną z dowolnym znakiem.
    Używamy rim_mask: rim jest po stronie GINGIVAL (otwarcie shell-a), więc
    e_occ ma wskazywać w PRZECIWNĄ stronę (ku kopule okluzyjnej). Flip robimy
    jako 180° obrót wokół osi AP (e_occ ORAZ e_lr), żeby NIE odbić lustrzanie
    mesha (to byłaby zła ręka nakładki — proper rotation det=+1 zachowany).

    Zwraca dict: e_lr, e_ap (→anterior), e_occ (→occlusal dome), centroid.
    """
    verts = np.asarray(aligner_mesh.vertices, dtype=float)
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    cov = (centered.T @ centered) / max(len(centered), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    cand = [eigvecs[:, 2], eigvecs[:, 1]]       # 2 osie w płaszczyźnie okluzyjnej

    def _evaluate(ap, lr):
        ap_c = centered @ ap
        lr_c = centered @ lr
        midline = np.abs(lr_c) < (lr_c.std() * 0.35)
        if int(midline.sum()) < 20:
            return -1.0, ap
        ap_mid = ap_c[midline]
        gap_hi = ap_c.max() - ap_mid.max()
        gap_lo = ap_mid.min() - ap_c.min()
        score = abs(gap_hi - gap_lo)
        anterior = (-ap) if gap_hi > gap_lo else ap
        return score, anterior

    best_score, best_ant, best_lr = -1.0, cand[0], cand[1]
    for i, ap in enumerate(cand):
        lr = cand[1 - i]
        s, ant = _evaluate(ap, lr)
        if s > best_score:
            best_score, best_ant, best_lr = s, ant, lr

    e_ap = best_ant / max(np.linalg.norm(best_ant), 1e-9)
    e_lr = best_lr - np.dot(best_lr, e_ap) * e_ap
    e_lr = e_lr / max(np.linalg.norm(e_lr), 1e-9)
    e_occ = np.cross(e_lr, e_ap)                # RH: (e_lr, e_ap, e_occ), det(S)=+1
    e_occ = e_occ / max(np.linalg.norm(e_occ), 1e-9)

    # Orientacja occlusal up/down z rim (rim = strona gingival/otwarcie)
    flipped = False
    if rim_mask is not None and np.any(rim_mask):
        rim_centroid = verts[rim_mask].mean(axis=0)
        gingival = rim_centroid - centroid       # ku otwarciu (gingival)
        if float(np.dot(e_occ, gingival)) > 0:   # e_occ ku gingival = źle → flip
            e_occ = -e_occ
            e_lr = -e_lr                          # 180° wokół e_ap (zachowuje RH, NIE mirror)
            flipped = True

    if verbose:
        print(
            f"[supports] anterior≈[{e_ap[0]:.2f},{e_ap[1]:.2f},{e_ap[2]:.2f}], "
            f"gap_score={best_score:.2f}, occlusal_flip={flipped}"
        )
    return {"e_lr": e_lr, "e_ap": e_ap, "e_occ": e_occ, "centroid": centroid}


def compute_print_transform(
    aligner_mesh,
    frame: dict,
    tilt_deg: float,
    z_gap: float = 2.0,
) -> np.ndarray:
    """4×4 transform scan→print orientation.

    Mapuje ramę skanu na świat:
      e_lr  → X (poziomo, oś obrotu nachylenia)
      e_ap  → [0, cosθ, sinθ]   (θ>0: anterior do GÓRY, posterior nisko)
      e_occ → [0, -sinθ, cosθ]
    Slider θ ∈ [-45, +45] wybiera która strona nisko (θ<0: anterior nisko).

    Potem translacja: najniższy punkt nakładki → Z=z_gap (zostaje miejsce na
    nóżki podpór schodzące do raftu na Z=0).
    """
    e_lr, e_ap, e_occ = frame["e_lr"], frame["e_ap"], frame["e_occ"]
    centroid = frame["centroid"]
    theta = np.radians(float(tilt_deg))
    ct, st = np.cos(theta), np.sin(theta)

    S = np.column_stack([e_lr, e_ap, e_occ])           # scan basis (kolumny)
    U = np.column_stack([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, ct, st]),
        np.array([0.0, -st, ct]),
    ])
    R = U @ S.T                                         # rotacja scan→world

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ centroid

    # translacja Z: najniższy vert nakładki → z_gap
    v = np.asarray(aligner_mesh.vertices, dtype=float)
    vt = (R @ (v - centroid).T).T
    T[2, 3] += float(z_gap - vt[:, 2].min())
    return T


def transform_points(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Zastosuj 4×4 transform do (N,3) punktów."""
    pts = np.asarray(pts, dtype=float)
    if pts.size == 0:
        return pts
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def generate_pillars_to_plane(
    anchors_print: np.ndarray,
    target_z: float,
    params: PillarParams | None = None,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Pillary w print space: każda kotwica → PIONOWO -Z do płaszczyzny target_z.

    Nóżki różnej długości (kotwice wyżej = dłuższe). Pomija kotwice zbyt
    blisko płaszczyzny (h ≤ tip_height). Concat (multi-solid STL).
    """
    if anchors_print is None or len(anchors_print) == 0:
        return None
    if params is None:
        params = PillarParams()
    down = np.array([0.0, 0.0, -1.0])
    meshes = []
    skipped = 0
    for a in anchors_print:
        h = float(a[2] - target_z)
        if h <= params.tip_height + 0.5:
            skipped += 1
            continue
        meshes.append(_make_pillar(a, down, params, height=h))
    if verbose:
        print(
            f"[supports] pillars→plane(z={target_z:.1f}): {len(meshes)} szt., "
            f"pominięto {skipped} (za krótkie)"
        )
    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def make_raft(
    anchors_print: np.ndarray,
    z_top: float = 0.0,
    thickness: float = 1.5,
    band_width: float = 3.5,
    solid_disk: bool = False,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Raft U-kształtny = **wstęga** śledząca pętlę kotwic (gingival margin).

    `solid_disk=True`: zamiast wstęgi U-kształtnej, raft = **filled disk**
    (wypukła otoczka kotwic, wypełniony środek). Likwiduje PreForm "Cup
    Detected" (U-band zamknięta pętla = closed cavity outline w warstwie
    raftu → PreForm flag). Trade-off: ~30% więcej resin, mocniejsza adhesion
    do build plate.

    Zamiast convex hull (który wypełniał środek U = przestrzeń języka i marnował
    resin), budujemy wstęgę o szerokości `band_width` wzdłuż pętli kotwic:
    dla każdego punktu normalna 2D → offset ±band_width/2 → rail wewnętrzny i
    zewnętrzny → ekstruzja w dół o `thickness`. Środek U pusty (nic nad nim).

    Pillary siedzą na środku wstęgi (na linii kotwic) → margines z obu stron.
    Wstęga tworzy zamkniętą pętlę (picture-frame) → stabilna baza. Bez
    zależności od shapely/earcut (offset band zamiast triangulacji niewypukłej).

    Uwaga: przy bardzo ostrych zakrętach (distal molarów) wewnętrzny rail może
    się lekko pinchować — nieszkodliwe dla raftu (sacrificial baza).
    """
    if anchors_print is None or len(anchors_print) < 6:
        return None
    loop = np.asarray(anchors_print, dtype=float)[:, :2]
    N = len(loop)
    z_bot = z_top - thickness

    if solid_disk:
        # SOLID DISK: convex hull XY anchors → wypełniony disk
        # Brak cavity → brak PreForm "Cup Detected".
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(loop)
            hull_pts = loop[hull.vertices]  # punkty hull w kolejności
        except Exception:
            hull_pts = loop
        # Rozszerz hull o band_width/2 (extra margines do build plate adhesion)
        cx, cy = float(hull_pts[:, 0].mean()), float(hull_pts[:, 1].mean())
        outward = hull_pts - np.array([cx, cy])
        norms = np.linalg.norm(outward, axis=1, keepdims=True)
        outward_unit = outward / np.maximum(norms, 1e-9)
        hull_pts = hull_pts + outward_unit * (band_width / 2.0)
        M = len(hull_pts)

        # extrude prism: top hull verts + bottom hull verts + top/bottom faces
        top = np.column_stack([hull_pts, np.full(M, z_top)])
        bot = np.column_stack([hull_pts, np.full(M, z_bot)])
        verts = np.vstack([top, bot])

        faces: list[list[int]] = []
        # Side walls — outward normal (hull CCW od +Z, więc reverse winding)
        for i in range(M):
            j = (i + 1) % M
            faces.append([j, i, M + j])
            faces.append([i, M + i, M + j])
        # Caps: top fan facing +Z, bottom fan facing -Z
        top_c_idx = 2 * M
        bot_c_idx = 2 * M + 1
        verts = np.vstack([verts, np.array([[cx, cy, z_top], [cx, cy, z_bot]])])
        for i in range(M):
            j = (i + 1) % M
            # Top cap (normal +Z): triangle CCW from +Z view
            faces.append([top_c_idx, i, j])
            # Bottom cap (normal -Z): triangle CW from +Z view
            faces.append([bot_c_idx, M + j, M + i])

        raft = trimesh.Trimesh(
            vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False
        )
        try:
            trimesh.repair.fix_normals(raft)
        except Exception:
            pass
        if verbose:
            print(
                f"[supports] raft SOLID DISK: hull {M} pkt, grubość {thickness}mm, "
                f"watertight={raft.is_watertight}"
            )
        return raft
    hw = band_width / 2.0

    # tangenty (central diff z zawinięciem) → normalne 2D
    nxt = np.roll(loop, -1, axis=0)
    prv = np.roll(loop, 1, axis=0)
    tang = nxt - prv
    tang = tang / np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tang[:, 1], tang[:, 0]])

    outer = loop + normal * hw
    inner = loop - normal * hw

    OT = np.column_stack([outer, np.full(N, z_top)])
    IT = np.column_stack([inner, np.full(N, z_top)])
    OB = np.column_stack([outer, np.full(N, z_bot)])
    IB = np.column_stack([inner, np.full(N, z_bot)])
    verts = np.vstack([OT, IT, OB, IB])
    oT, iT, oB, iB = 0, N, 2 * N, 3 * N

    faces: list[list[int]] = []
    for i in range(N):
        j = (i + 1) % N
        # górna powierzchnia wstęgi (OT-IT)
        faces.append([oT + i, oT + j, iT + j])
        faces.append([oT + i, iT + j, iT + i])
        # dolna powierzchnia (OB-IB)
        faces.append([oB + i, iB + j, oB + j])
        faces.append([oB + i, iB + i, iB + j])
        # ściana zewnętrzna (OT-OB)
        faces.append([oT + i, oB + j, oT + j])
        faces.append([oT + i, oB + i, oB + j])
        # ściana wewnętrzna (IT-IB)
        faces.append([iT + i, iT + j, iB + j])
        faces.append([iT + i, iB + j, iB + i])

    raft = trimesh.Trimesh(
        vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False
    )
    try:
        trimesh.repair.fix_normals(raft)
    except Exception:
        pass
    if verbose:
        print(
            f"[supports] raft U-band: {N} pkt pętli, szer {band_width}mm, "
            f"grubość {thickness}mm, watertight={raft.is_watertight}"
        )
    return raft


def overhang_face_mask(
    aligner_mesh,
    transform: np.ndarray,
    overhang_angle_deg: float = 45.0,
    restrict_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Per-face bool: True gdy ściana jest **overhangiem** w orientacji druku.

    Druk SLA/DLP buduje warstwy od dołu (raft) w górę (+Z). Ściana patrząca w
    dół bardziej niż kąt krytyczny od poziomu nie ma na czym się oprzeć →
    drukuje "w powietrzu". Liczymy to z normalnej ściany po obrocie do print
    space (szybki ekwiwalent sweepu warstwowego):

        overhang  ⟺  (R · normal).z  <  -cos(overhang_angle)

    overhang_angle = kąt od poziomu; ściany bardziej poziome (od‑dół) niż ten
    kąt wymagają podpory. 45° to typowy próg żywicy.

    `restrict_vertex_mask`: jeśli podane, liczymy overhang TYLKO dla ścian
    dotykających tych wierzchołków (np. rim_mask → analiza overhangów wyłącznie
    na KRAWĘDZI, nie na powierzchni okluzyjnej — bo podpory idą tylko na rim).
    """
    R = np.asarray(transform, dtype=float)[:3, :3]
    fn = np.asarray(aligner_mesh.face_normals, dtype=float)
    fn_print_z = fn @ R[2, :]                       # (R·n).z = R[2,:]·n
    thresh = -np.cos(np.radians(float(overhang_angle_deg)))
    mask = fn_print_z < thresh
    if restrict_vertex_mask is not None:
        touches = restrict_vertex_mask[np.asarray(aligner_mesh.faces)].any(axis=1)
        mask = mask & touches
    return mask


def detect_overhang_faces_slicer_check(
    aligner_mesh,
    transform: np.ndarray,
    pillars_mesh=None,
    raft_mesh=None,
    overhang_angle_deg: float = 45.0,
    voxel_size: float = 0.5,
    restrict_vertex_mask: np.ndarray | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """**Slicer-style overhang detection** (PrusaSlicer/Cura logic): hybryda
    geometric pre-filter + per-voxel "is layer below empty?" check.

    Per-face geometric `overhang_face_mask` ma duże false positives: każda
    ściana patrząca w dół jest flagowana, nawet jeśli pod nią (w print Z) jest
    materiał (np. inverted cusp wewnątrz cap-u — ściana w dół, ale ściana
    cap-u trwa w dół = supported).

    Ten algorytm:
      1. **Geometric pre-filter** — ściany kandydaci `(R·n).z < -cos(angle)`,
         żeby nie sprawdzać każdej ściany przeciwko gridowi.
      2. **Layer-below voxel check** — voxelizujemy aligner+pillars+raft w
         print space. Dla każdego kandydata sprawdzamy: czy w warstwie
         **poniżej** (k-1) w przestrzeni `cot(angle)` lateralnie jest materiał?
         - YES → face ma punkt zaczepienia w warstwie niżej → NIE overhang.
         - NO  → face drukuje w powietrzu → overhang.

    Voxel size: 0.5mm dla typowego aligner = ~1-2 mln voxeli = <1s voxelizacji.

    `pillars_mesh` + `raft_mesh`: opcjonalne, dorzuca pillary+raft do
    `material` mask — face rim z pillarem pod sobą NIE jest overhang.

    `restrict_vertex_mask`: ścieśnia do ścian dotykających tych wierzchołków
    (np. tylko rim, albo tylko non-rim).
    """
    from scipy import ndimage as ndi

    from .sdf import voxelize_solid

    # 1. Geometric pre-filter
    R = np.asarray(transform, dtype=float)[:3, :3]
    fn = np.asarray(aligner_mesh.face_normals, dtype=float)
    fn_print_z = fn @ R[2, :]
    thresh = -np.cos(np.radians(float(overhang_angle_deg)))
    geometric_mask = fn_print_z < thresh
    if restrict_vertex_mask is not None:
        touches = restrict_vertex_mask[np.asarray(aligner_mesh.faces)].any(axis=1)
        geometric_mask = geometric_mask & touches

    n_geom = int(geometric_mask.sum())
    if n_geom == 0:
        if verbose:
            print(f"[supports] slicer-check: 0 kandydatów geometric")
        return np.zeros(len(aligner_mesh.faces), dtype=bool)

    # 2. Voxelize material w print space (aligner + pillars + raft)
    al = aligner_mesh.copy()
    al.apply_transform(np.asarray(transform, dtype=float))
    parts = [al]
    if pillars_mesh is not None and len(pillars_mesh.vertices) > 0:
        parts.append(pillars_mesh)
    if raft_mesh is not None and len(raft_mesh.vertices) > 0:
        parts.append(raft_mesh)
    bbox_min = np.min([p.bounds[0] for p in parts], axis=0).astype(float)
    bbox_max = np.max([p.bounds[1] for p in parts], axis=0).astype(float)
    bbox_min[:2] -= voxel_size
    bbox_max[:2] += voxel_size
    bbox_max[2] += voxel_size

    A_a = voxelize_solid(al, bbox_min, bbox_max, voxel_size).data
    A_p = (voxelize_solid(pillars_mesh, bbox_min, bbox_max, voxel_size).data
           if pillars_mesh is not None and len(pillars_mesh.vertices) > 0
           else np.zeros_like(A_a))
    A_r = (voxelize_solid(raft_mesh, bbox_min, bbox_max, voxel_size).data
           if raft_mesh is not None and len(raft_mesh.vertices) > 0
           else np.zeros_like(A_a))
    material = A_a | A_p | A_r
    nx, ny, nz = material.shape

    # 3. has_below[i,j,k] = czy warstwa k-1 ma materiał w cot(angle) lateralnie
    cot_a = 1.0 / max(np.tan(np.radians(float(overhang_angle_deg))), 1e-9)
    dilation = max(1, int(round(cot_a)))
    has_below = np.zeros_like(material)
    has_below[:, :, 0] = True   # build plate / pierwsza warstwa zawsze "supported"
    for k in range(1, nz):
        prev = material[:, :, k - 1]
        if dilation > 0:
            prev = ndi.binary_dilation(prev, iterations=dilation)
        has_below[:, :, k] = prev

    # 4. Dla każdego kandydata: sprawdź centroid w voxel space
    fc = np.asarray(al.triangles_center, dtype=float)
    ijk = ((fc - bbox_min) / voxel_size).astype(np.int64)
    ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
    ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
    ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)

    face_has_support_below = has_below[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    overhang = geometric_mask & ~face_has_support_below

    if verbose:
        n_overhang = int(overhang.sum())
        n_filtered = n_geom - n_overhang
        print(
            f"[supports] slicer-check: voxel={voxel_size}mm, "
            f"geometric={n_geom} → filtered_by_layer_below={n_filtered} → "
            f"overhang={n_overhang} ({100*n_filtered/max(n_geom,1):.0f}% "
            f"odfiltrowane jako 'ma materiał poniżej')"
        )

    return overhang


@dataclass
class OverhangAnalysis:
    """Wynik analizy layer-sweep overhang.

    Trzyma siatkę voxeli + wszystko czego potrzeba żeby:
      - kolorować ściany aligner (face_mask),
      - generować podpory pod overhangami (overhang_voxels + bbox_min + voxel_size).
    """
    face_mask: np.ndarray         # (F,) bool — które ściany aligner są overhang
    overhang_voxels: np.ndarray   # (NX, NY, NZ) bool — aligner voxele bez podparcia
    bbox_min: np.ndarray          # (3,) — origin gridu w print space
    voxel_size: float
    aligner_min_z: float          # najniższy Z aligner-material w print space
    n_overhang: int
    n_total: int


def analyze_overhang_layer_sweep(
    aligner_mesh,
    transform: np.ndarray,
    pillars_mesh=None,
    raft_mesh=None,
    overhang_angle_deg: float = 45.0,
    voxel_size: float = 1.0,
    restrict_vertex_mask: np.ndarray | None = None,
    restrict_dilate_iters: int = 1,
    verbose: bool = True,
) -> OverhangAnalysis:
    """**Component-based layer-sweep overhang** (jak modern slicer: Cura, PrusaSlicer).

    Voxelizuje aligner+pillars+raft we wspólnym gridzie w **print space**. Potem
    sweepuje Z od dołu (build plate) używając **per-component logic**:

      1. W każdej warstwie znajdź spójne 2D wyspy materiału (`scipy.ndimage.label`).
      2. Dla każdej wyspy: max(supported_below_shadow) na voxelach wyspy.
         - >0 ⟹ wyspa ma punkt zaczepienia → CAŁA supported,
         - =0 ⟹ wyspa wisi w powietrzu → CAŁA = overhang.
      3. Shadow = supported below rozszerzone o `cot(overhang_angle)` voxeli
         (printable slope tolerance).

    Dlaczego per-component a nie per-voxel:
      - Fizyka: w jednej warstwie cały resin curyje **jednocześnie**, więc jeśli
        wyspa ma choć JEDEN punkt zaczepienia, cała się trzyma na nim.
      - Cleaner: nie produkuje rozproszonych pojedynczych voxeli na leading edges,
        tylko spójne overhang regiony — łatwiejsze do wizualizacji i support gen.
      - Wykrywa "floating islands" (wyspy które pojawiają się znikąd) jako
        idealne kandydatki na podpory.

    Dilation per warstwa = `cot(overhang_angle)` voxeli (voxel_xy = voxel_z):
      - 45° → 1 voxel lateralnie (typowy próg żywicy)
      - 30° → 2 voxele (bardziej liberalny, mniej overhangu)
      - 60° → 1 voxel (bardziej rygorystyczny)

    **Fix:** bbox NIE pada w Z na dół (bottom siatki = bottom materiału = build
    plate). Seed: pierwsza warstwa z jakimkolwiek materiałem (raft lub bottom
    pillarów) — propagacja startuje stamtąd niezależnie od dokładnego Z=0.

    `restrict_vertex_mask`: jeśli podane (np. `report.rim_mask`), **detekcja
    overhang ogranicza się do voxeli przy ścianach dotykających tych
    wierzchołków** (typowo rim band — gingival margin). Sweep propagacji nadal
    używa CAŁEGO aligner (fizyka: occlusal mass podpiera rim nad sobą), ale
    `overhang_voxels` i `face_mask` są maskowane → podpory generujemy WYŁĄCZNIE
    na krawędzi (wymóg kliniczny: snap-off bez szlifowania powierzchni fit).
    `restrict_dilate_iters`: pogrubienie maski rim w voxelach (default 1 = ~1
    voxel rim band; zwiększ jeśli rim jest „dziurawy" przy małej rozdzielczości).
    """
    from scipy import ndimage as ndi

    from .sdf import voxelize_solid

    al = aligner_mesh.copy()
    al.apply_transform(np.asarray(transform, dtype=float))

    parts = [al]
    if pillars_mesh is not None and len(pillars_mesh.vertices) > 0:
        parts.append(pillars_mesh)
    if raft_mesh is not None and len(raft_mesh.vertices) > 0:
        parts.append(raft_mesh)

    # bbox: padding TYLKO w XY (lateralne miejsce na cot-dilation),
    # NIE w Z na dół — bottom siatki = bottom materiału = build plate.
    bbox_min = np.min([p.bounds[0] for p in parts], axis=0).astype(float)
    bbox_max = np.max([p.bounds[1] for p in parts], axis=0).astype(float)
    bbox_min[0] -= voxel_size
    bbox_min[1] -= voxel_size
    bbox_max[0] += voxel_size
    bbox_max[1] += voxel_size
    bbox_max[2] += voxel_size  # zapas na górze (na wszelki wypadek)

    A = voxelize_solid(al, bbox_min, bbox_max, voxel_size).data
    P = (voxelize_solid(pillars_mesh, bbox_min, bbox_max, voxel_size).data
         if pillars_mesh is not None and len(pillars_mesh.vertices) > 0
         else np.zeros_like(A))
    R = (voxelize_solid(raft_mesh, bbox_min, bbox_max, voxel_size).data
         if raft_mesh is not None and len(raft_mesh.vertices) > 0
         else np.zeros_like(A))
    material = A | P | R

    cot_a = 1.0 / max(np.tan(np.radians(float(overhang_angle_deg))), 1e-9)
    dilation = max(1, int(round(cot_a)))

    Z = A.shape[2]
    supported = np.zeros_like(A)

    # *Seed:* znajdź globalnie pierwszą warstwę z JAKIMKOLWIEK materiałem —
    # to jest pierwszy layer dotykający build plate (raft lub bottom pillarów).
    layers_with_mat = material.any(axis=(0, 1))   # (Z,) bool
    if not layers_with_mat.any():
        # nic do analizy
        empty_mask = np.zeros(len(al.faces), dtype=bool)
        return OverhangAnalysis(
            face_mask=empty_mask, overhang_voxels=np.zeros_like(A),
            bbox_min=bbox_min, voxel_size=voxel_size,
            aligner_min_z=float(bbox_min[2]), n_overhang=0, n_total=0,
        )
    first_k = int(np.argmax(layers_with_mat))
    supported[:, :, first_k] = material[:, :, first_k]

    # === Component-based propagation (slicer-style overhang detection) ===
    # W każdej warstwie znajdź 2D-spójne wyspy materiału. Wyspa jest
    # supported jeśli JAKIKOLWIEK jej voxel ma punkt zaczepienia w supported
    # below (z cot(angle) dilation — bo nie pionowo idealnie, tylko w obrębie
    # printable angle). W przeciwnym razie CAŁA wyspa = overhang.
    #
    # Fizyka: w jednej warstwie resin curyje równocześnie, więc jeśli wyspa
    # ma choć JEDEN punkt zaczepienia o warstwę niżej — cała się trzyma.
    # 8-connectivity (przekątne) — wyspy z punktem dotyku w narożniku to OK.
    struct2d = np.ones((3, 3), dtype=bool)
    for k in range(first_k + 1, Z):
        layer_mat = material[:, :, k]
        if not layer_mat.any():
            continue
        # "shadow" supported-below rozszerzony o cot(angle) — obszar w którym
        # ten layer może mieć punkt zaczepienia (printable slope).
        below_shadow = supported[:, :, k - 1]
        if dilation > 0:
            below_shadow = ndi.binary_dilation(below_shadow, iterations=dilation)
        # Spójne wyspy w obecnej warstwie
        labels_2d, n_comp = ndi.label(layer_mat, structure=struct2d)
        if n_comp == 0:
            continue
        # Per wyspa: max of below_shadow nad voxelami wyspy.
        # >0 ⟹ ma punkt zaczepienia ⟹ CAŁA wyspa supported.
        # =0 ⟹ wisi w powietrzu ⟹ CAŁA wyspa overhang.
        max_per_comp = ndi.maximum(
            below_shadow.astype(np.uint8),
            labels_2d,
            index=np.arange(1, n_comp + 1),
        )
        sup_labels = np.where(np.asarray(max_per_comp) > 0)[0] + 1
        if len(sup_labels) > 0:
            sup_mask_2d = np.isin(labels_2d, sup_labels)
            supported[:, :, k] = layer_mat & sup_mask_2d
        # else: cała warstwa to overhang, supported[:,:,k] zostaje zerowa

    overhang_voxels = A & ~supported

    # — Restrykcja do rim band (klinicznie: podpory tylko na krawędzi trim) —
    if restrict_vertex_mask is not None:
        rvm = np.asarray(restrict_vertex_mask, dtype=bool)
        if rvm.shape[0] != len(al.vertices):
            raise ValueError(
                f"restrict_vertex_mask length ({len(rvm)}) ≠ aligner "
                f"vertices ({len(al.vertices)})"
            )
        # ściana = rim jeśli JAKIKOLWIEK wierzchołek jest w masce (band touch)
        rim_face_mask = rvm[np.asarray(al.faces)].any(axis=1)
        rim_face_idx = np.where(rim_face_mask)[0]
        if len(rim_face_idx) == 0:
            overhang_voxels = np.zeros_like(A)
        else:
            # stamp rim face centroids w voxel grid → rim_voxels mask
            rim_fc = np.asarray(al.triangles_center, dtype=float)[rim_face_idx]
            r_ijk = ((rim_fc - bbox_min) / voxel_size).astype(np.int64)
            nx, ny, nz = A.shape
            r_ijk[:, 0] = np.clip(r_ijk[:, 0], 0, nx - 1)
            r_ijk[:, 1] = np.clip(r_ijk[:, 1], 0, ny - 1)
            r_ijk[:, 2] = np.clip(r_ijk[:, 2], 0, nz - 1)
            rim_voxels = np.zeros_like(A)
            rim_voxels[r_ijk[:, 0], r_ijk[:, 1], r_ijk[:, 2]] = True
            if restrict_dilate_iters > 0:
                rim_voxels = ndi.binary_dilation(
                    rim_voxels, iterations=int(restrict_dilate_iters)
                )
            overhang_voxels = overhang_voxels & rim_voxels
            if verbose:
                print(
                    f"[supports] rim restriction: {int(rim_face_mask.sum())} "
                    f"rim faces / {len(al.faces)}, rim voxels {int(rim_voxels.sum())}"
                )

    n_over = int(overhang_voxels.sum())
    n_total = int(A.sum())
    if verbose:
        pct = 100.0 * n_over / max(n_total, 1)
        scope = "rim only" if restrict_vertex_mask is not None else "full"
        print(
            f"[supports] layer-sweep: voxel={voxel_size}mm, dil={dilation}/layer "
            f"@{overhang_angle_deg:.0f}°, seed_layer={first_k}, scope={scope}, "
            f"overhang {n_over}/{n_total} vox ({pct:.1f}%)"
        )

    # mapowanie voxeli → ściany aligner (przez centroidy)
    if n_over == 0:
        face_mask = np.zeros(len(al.faces), dtype=bool)
    else:
        fc = np.asarray(al.triangles_center, dtype=float)
        ijk = ((fc - bbox_min) / voxel_size).astype(np.int64)
        nx, ny, nz = overhang_voxels.shape
        ijk[:, 0] = np.clip(ijk[:, 0], 0, nx - 1)
        ijk[:, 1] = np.clip(ijk[:, 1], 0, ny - 1)
        ijk[:, 2] = np.clip(ijk[:, 2], 0, nz - 1)
        face_mask = overhang_voxels[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    aligner_min_z = float(bbox_min[2] + first_k * voxel_size)

    return OverhangAnalysis(
        face_mask=face_mask,
        overhang_voxels=overhang_voxels,
        bbox_min=bbox_min,
        voxel_size=voxel_size,
        aligner_min_z=aligner_min_z,
        n_overhang=n_over,
        n_total=n_total,
    )


def detect_overhang_faces_layer_sweep(
    aligner_mesh,
    transform: np.ndarray,
    pillars_mesh=None,
    raft_mesh=None,
    overhang_angle_deg: float = 45.0,
    voxel_size: float = 1.0,
    verbose: bool = True,
) -> np.ndarray:
    """Wrapper zwracający tylko face_mask (kompat ze starszym kodem)."""
    return analyze_overhang_layer_sweep(
        aligner_mesh, transform,
        pillars_mesh=pillars_mesh, raft_mesh=raft_mesh,
        overhang_angle_deg=overhang_angle_deg, voxel_size=voxel_size,
        verbose=verbose,
    ).face_mask


def generate_overhang_supports(
    analysis: OverhangAnalysis,
    target_z: float = 0.0,
    params: PillarParams | None = None,
    min_cluster_voxels: int = 8,
    merge_xy_dist: float = 2.5,
    verbose: bool = True,
) -> tuple[trimesh.Trimesh | None, np.ndarray]:
    """Generuje podpory **w środku każdego klastra** overhang voxeli.

    Algorytm:
      1. 3D connected components na `overhang_voxels` (`scipy.ndimage.label`).
      2. Dla każdego klastra dostatecznie dużego (`min_cluster_voxels`):
         - znajdź **najniższą** warstwę Z w klastrze (leading edge — tu się
           overhang zaczyna „w powietrzu"),
         - weź XY centroid tej warstwy → punkt styku z aligner,
         - to jest „tip" pillara (punkt gdzie pillar dotyka nakładki).
      3. Dedupe XY: kotwice bliżej niż `merge_xy_dist` mm → jedna.
      4. `generate_pillars_to_plane(anchors, target_z)` → pillary pionowo w dół.

    Zwraca:
      - mesh pillarów (concat) albo None gdy nic do podparcia,
      - kotwice w print space (N, 3) — XY = środek klastra, Z = górny punkt
        styku (gdzie pillar styka się z nakładką).
    """
    from scipy import ndimage as ndi

    if not analysis.overhang_voxels.any():
        return None, np.zeros((0, 3), dtype=float)

    if params is None:
        params = PillarParams()

    labels, n_clusters = ndi.label(analysis.overhang_voxels)
    if n_clusters == 0:
        return None, np.zeros((0, 3), dtype=float)

    sizes = ndi.sum(
        analysis.overhang_voxels, labels, index=np.arange(1, n_clusters + 1)
    )
    anchors_list: list[list[float]] = []
    bb = analysis.bbox_min
    v = analysis.voxel_size

    for ci in range(1, n_clusters + 1):
        if sizes[ci - 1] < min_cluster_voxels:
            continue
        coords = np.argwhere(labels == ci)
        # leading edge = najniższa warstwa Z w klastrze
        kmin = int(coords[:, 2].min())
        leading = coords[coords[:, 2] == kmin]
        cx = float(leading[:, 0].mean())
        cy = float(leading[:, 1].mean())
        x = bb[0] + (cx + 0.5) * v
        y = bb[1] + (cy + 0.5) * v
        z = bb[2] + (kmin + 0.5) * v
        anchors_list.append([x, y, z])

    if not anchors_list:
        if verbose:
            print(
                f"[supports] overhang clusters: {n_clusters} found, "
                f"żaden ≥ min_cluster_voxels={min_cluster_voxels}"
            )
        return None, np.zeros((0, 3), dtype=float)

    anchors = np.asarray(anchors_list, dtype=float)

    # dedupe XY (jeden support na ~merge_xy_dist²)
    if merge_xy_dist > 0 and len(anchors) > 1:
        keys = np.floor(anchors[:, :2] / max(merge_xy_dist, 0.5)).astype(np.int64)
        _, idx = np.unique(keys, axis=0, return_index=True)
        anchors = anchors[idx]

    if verbose:
        kept = int((sizes >= min_cluster_voxels).sum())
        print(
            f"[supports] overhang clusters: {n_clusters} found, "
            f"{kept} ≥ {min_cluster_voxels} vox, {len(anchors)} kotwic po dedupe "
            f"(merge {merge_xy_dist}mm)"
        )

    pillars = generate_pillars_to_plane(anchors, target_z, params, verbose=False)
    if verbose and pillars is not None:
        print(
            f"[supports] overhang pillars: {len(anchors)} → "
            f"{len(pillars.vertices)}v / {len(pillars.faces)}f"
        )
    return pillars, anchors


def detect_overhang_anchors(
    aligner_mesh,
    transform: np.ndarray,
    overhang_angle_deg: float = 45.0,
    spacing: float = 3.0,
    rim_anchors_print: np.ndarray | None = None,
    dedupe_dist: float = 2.5,
    min_z: float = 1.5,
    verbose: bool = True,
) -> np.ndarray:
    """Kotwice priorytetowe na overhangach (w print space) → podpory tam gdzie
    nakładka inaczej drukowałaby się w powietrzu.

    1. ściany overhang (`overhang_face_mask`),
    2. ich centroidy → print space,
    3. odfiltruj za nisko (min_z — i tak blisko raftu/rim),
    4. grid‑downsample do `spacing` (nie przesadzamy z gęstością),
    5. dedupe względem kotwic rim (te już mają podpory).
    """
    mask = overhang_face_mask(aligner_mesh, transform, overhang_angle_deg)
    n_over = int(mask.sum())
    if n_over == 0:
        if verbose:
            print(f"[supports] overhang: 0 ścian @{overhang_angle_deg:.0f}° (świetna orientacja!)")
        return np.zeros((0, 3), dtype=float)

    fc = np.asarray(aligner_mesh.triangles_center, dtype=float)[mask]
    fc_print = transform_points(fc, transform)
    fc_print = fc_print[fc_print[:, 2] > min_z]
    if len(fc_print) == 0:
        return np.zeros((0, 3), dtype=float)

    # grid downsample: jeden punkt na komórkę spacing³
    keys = np.floor(fc_print / max(spacing, 0.5)).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    pts = fc_print[idx]

    if rim_anchors_print is not None and len(rim_anchors_print) > 0:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(rim_anchors_print).query(pts)
        pts = pts[d > dedupe_dist]

    if verbose:
        print(
            f"[supports] overhang: {n_over} ścian → {len(pts)} kotwic "
            f"priorytetowych (@{overhang_angle_deg:.0f}°, spacing {spacing}mm)"
        )
    return pts


def generate_braces(
    anchors_print: np.ndarray,
    params: PillarParams,
    brace_angle_deg: float = 45.0,
    brace_diameter: float = 0.6,
    min_height: float = 5.0,
    z_start: float = 1.0,
    verbose: bool = True,
) -> trimesh.Trimesh | None:
    """Zigzag diagonal cross-bracing między sąsiednimi pillarami (closed loop).

    Anti-collapse: wysokie cienkie pillary (cantilevery) wyboczają się pod
    peel-force druku. Przekątne zygzakiem TRIANGULUJĄ las pillarów (Warren
    truss) → wielokrotnie sztywniej przy minimum materiału.

    Geometria: dla każdej "bay" (para sąsiednich pillarów i↔i+1) i każdego
    "story" (segment wysokości) dodaj przekątną, alternując kierunek
    (zygzak ▽△▽△). Kąt od poziomu = `brace_angle_deg`; rise = horiz·tan(θ).
    45° = sweet spot (triangulacja + self-supporting w SLA). Tylko bay gdzie
    oba pillary > `min_height` (krótkie posterior nie potrzebują).

    Struts = cienkie cylindry (`brace_diameter`), concat (multi-solid STL).
    """
    if anchors_print is None or len(anchors_print) < 3:
        return None
    n = len(anchors_print)
    theta = np.radians(float(np.clip(brace_angle_deg, 20.0, 80.0)))
    tan_a = np.tan(theta)
    r = brace_diameter / 2.0

    struts = []
    n_bays = 0
    for i in range(n):
        a0 = anchors_print[i]
        a1 = anchors_print[(i + 1) % n]
        z_lim = float(min(a0[2], a1[2]) - params.tip_height)
        if z_lim < min_height:
            continue
        horiz = float(np.linalg.norm(np.asarray(a1[:2]) - np.asarray(a0[:2])))
        if horiz < 1e-3:
            continue
        rise = horiz * tan_a
        if rise < 1e-3:
            continue
        z = z_start
        story = 0
        bay_struts = 0
        while z + rise <= z_lim:
            if story % 2 == 0:
                p0 = np.array([a0[0], a0[1], z])
                p1 = np.array([a1[0], a1[1], z + rise])
            else:
                p0 = np.array([a0[0], a0[1], z + rise])
                p1 = np.array([a1[0], a1[1], z])
            try:
                struts.append(
                    trimesh.creation.cylinder(radius=r, segment=(p0, p1), sections=6)
                )
                bay_struts += 1
            except Exception:
                pass
            z += rise
            story += 1
        if bay_struts > 0:
            n_bays += 1

    if not struts:
        if verbose:
            print(f"[supports] braces: 0 (pillary za krótkie / min_height={min_height})")
        return None
    if verbose:
        print(
            f"[supports] braces: {len(struts)} struts w {n_bays} bayach "
            f"@{brace_angle_deg:.0f}° (Ø{brace_diameter}mm)"
        )
    return trimesh.util.concatenate(struts)


def optimize_tilt_for_min_overhang(
    aligner_mesh,
    ap_frame: dict,
    tilt_range_deg: tuple[float, float] = (-50.0, 50.0),
    n_steps: int = 51,
    exclude_vertex_mask: np.ndarray | None = None,
    overhang_angle_deg: float = 45.0,
    weight_by_area: bool = True,
    z_gap: float = 2.0,
    verbose: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Sweep kątów nachylenia, znajdź ten z **minimalnym overhangiem na nie-rim
    ścianach** (powierzchnie fit/okluzyjne — tam gdzie nie stawiamy podpór).

    Idea: rim ma już podpory niezależnie od kąta. Krytyczne jest żeby
    powierzchnia okluzyjna/buccal/lingual sama nie drukowała w powietrzu.
    Optymalizujemy kąt tilta tak żeby zminimalizować *powierzchnię* (lub liczbę)
    nie-rim ścian, które są przerwą okluzyjną względem build plate.

    Metryka: **szybka, normal-based** — `(R · n).z < -cos(overhang_angle)`. To
    nie uwzględnia self-supporting (gdy aligner sam się podpiera w jakimś
    miejscu), ale jest świetnym proxy do *orientacji*: minimum tej funkcji
    odpowiada minimum potencjalnego ryzyka. Finalną weryfikację robi
    `analyze_overhang_layer_sweep`.

    Złożoność: O(F) per kąt, ~50ms na 350k faces. Cały sweep 51 punktów = ~1s.

    `exclude_vertex_mask`: typowo `rim_mask` — ściany dotykające tych
    wierzchołków są WYKLUCZONE z metryki (rim dostaje podpory niezależnie).

    Zwraca `(best_tilt_deg, all_angles, all_scores)` — best_tilt do ustawienia
    slidera, all_* do ewentualnego wykresu/loga.
    """
    angles = np.linspace(
        float(tilt_range_deg[0]), float(tilt_range_deg[1]), int(n_steps)
    )
    faces_arr = np.asarray(aligner_mesh.faces)
    fn = np.asarray(aligner_mesh.face_normals, dtype=float)
    fa = np.asarray(aligner_mesh.area_faces, dtype=float)

    # mask "non-rim faces" — ściany NIE dotykające rim_mask
    if exclude_vertex_mask is not None:
        evm = np.asarray(exclude_vertex_mask, dtype=bool)
        rim_touches = evm[faces_arr].any(axis=1)
        non_rim_face = ~rim_touches
    else:
        non_rim_face = np.ones(len(faces_arr), dtype=bool)

    thresh = -np.cos(np.radians(float(overhang_angle_deg)))
    scores = np.zeros_like(angles)

    for i, theta_deg in enumerate(angles):
        T = compute_print_transform(aligner_mesh, ap_frame, float(theta_deg), z_gap=z_gap)
        R = T[:3, :3]
        # (R · n).z  ==  R[2,:] · n  per face (wektorowo)
        fn_z = fn @ R[2, :]
        over = (fn_z < thresh) & non_rim_face
        scores[i] = float(fa[over].sum()) if weight_by_area else float(over.sum())

    best_i = int(np.argmin(scores))
    best_angle = float(angles[best_i])
    if verbose:
        worst_i = int(np.argmax(scores))
        unit = "mm²" if weight_by_area else "ścian"
        print(
            f"[supports] tilt opt: {n_steps} próbek "
            f"{tilt_range_deg[0]:+.0f}°..{tilt_range_deg[1]:+.0f}°, "
            f"non-rim overhang @{overhang_angle_deg:.0f}° → "
            f"best={best_angle:+.1f}° ({scores[best_i]:.1f} {unit}), "
            f"worst={angles[worst_i]:+.1f}° ({scores[worst_i]:.1f} {unit})"
        )
    return best_angle, angles, scores


def make_drainage_cylinder(
    aligner_mesh,
    transform: np.ndarray,
    hole_radius_mm: float = 1.5,
    height_above_mm: float = 2.0,
    height_below_mm: float = 50.0,
    sections: int = 24,
) -> "trimesh.Trimesh":
    """Buduje cylinder drenażowy w SCAN SPACE: oś = print Z, XY centroid top
    occlusal. Wysokość spanuje od `height_above` ponad top cap aż do
    `height_below` poniżej (default 50mm → ⊃ raft + build plate).

    Single source of truth dla obu otworów (aligner + raft) — ten sam cylinder
    użyty w obu boolean diff → jeden zunifikowany tunel.
    """
    import trimesh

    T = np.asarray(transform, dtype=float)
    al_print = aligner_mesh.copy()
    al_print.apply_transform(T)

    z_max = float(al_print.vertices[:, 2].max())
    threshold = z_max - 1.0
    top_verts = al_print.vertices[al_print.vertices[:, 2] > threshold]
    if len(top_verts) == 0:
        top_verts = al_print.vertices[al_print.vertices[:, 2] >= z_max - 0.01]
    top_xy = top_verts[:, :2].mean(axis=0)

    cyl_height = height_above_mm + height_below_mm
    cyl_center_z = z_max + height_above_mm - cyl_height / 2

    cyl = trimesh.creation.cylinder(
        radius=float(hole_radius_mm), height=cyl_height, sections=sections,
    )
    cyl.apply_translation([top_xy[0], top_xy[1], cyl_center_z])

    T_inv = np.linalg.inv(T)
    cyl.apply_transform(T_inv)
    return cyl


def add_drainage_hole(
    aligner_mesh,
    transform: np.ndarray,
    hole_radius_mm: float = 1.5,
    hole_depth_mm: float = 10.0,
    verbose: bool = True,
) -> "trimesh.Trimesh":
    """**Drainage hole** w najwyższym punkcie nakładki w print Z (wnętrze cap-u).

    Aby drenować TAKŻE U-raft cavity (osobna cavity między raft i cap), użyj
    osobno `cut_raft_drain_hole(raft_mesh, aligner_mesh, transform, radius)`.
    """
    if hole_radius_mm <= 0:
        return aligner_mesh
    import trimesh

    cyl = make_drainage_cylinder(
        aligner_mesh, transform, hole_radius_mm,
        height_above_mm=2.0, height_below_mm=float(hole_depth_mm),
    )

    try:
        result = trimesh.boolean.difference([aligner_mesh, cyl])
    except Exception as e:
        if verbose:
            print(f"[supports] drainage hole: boolean FAILED ({e}) — bez otworu")
        return aligner_mesh

    if verbose:
        print(
            f"[supports] drainage hole (aligner): Ø{2*hole_radius_mm:.1f}mm → "
            f"{len(result.vertices)}v / {len(result.faces)}f, "
            f"watertight={result.is_watertight}"
        )
    return result


def cut_raft_drain_hole(
    raft_mesh,
    aligner_mesh,
    transform: np.ndarray,
    hole_radius_mm: float = 1.5,
    verbose: bool = True,
) -> "trimesh.Trimesh":
    """Drainage cylinder od top occlusal w dół przez raft → likwiduje
    PreForm "Cup Detected" dla **wnętrza U-band raftu** (cavity wewnątrz
    podkowy raft-u, zamknięta od dołu przez build plate i od góry przez cap).

    Używa tej samej osi co `add_drainage_hole` (single source of truth via
    `make_drainage_cylinder`) → jeden zunifikowany tunel od top cap-u w dół
    przez raft. Resin swobodnie wypływa podczas peel.

    Raft jest w PRINT space (z=0 do z=-thickness). Cylinder z scan space
    transformujemy do print space przed boolean.
    """
    if hole_radius_mm <= 0 or raft_mesh is None or len(raft_mesh.vertices) == 0:
        return raft_mesh
    import trimesh

    cyl_scan = make_drainage_cylinder(
        aligner_mesh, transform, hole_radius_mm,
        height_above_mm=2.0, height_below_mm=50.0,
    )
    T = np.asarray(transform, dtype=float)
    cyl_print = cyl_scan.copy()
    cyl_print.apply_transform(T)

    try:
        result = trimesh.boolean.difference([raft_mesh, cyl_print])
    except Exception as e:
        if verbose:
            print(f"[supports] raft drain: boolean FAILED ({e}) — bez otworu")
        return raft_mesh

    if verbose:
        print(
            f"[supports] drainage hole (raft): Ø{2*hole_radius_mm:.1f}mm → "
            f"{len(result.vertices)}v / {len(result.faces)}f, "
            f"watertight={result.is_watertight}"
        )
    return result
