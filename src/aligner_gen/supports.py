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
      - długość pillara wyznacza odległość apex → top raftu (S4); na razie
        stała wartość do podglądu geometrii w S3.
    """
    tip_diameter: float = 0.4    # mm — średnica kontaktu (snap-off interface)
    tip_height: float = 1.0      # mm — wysokość zwężenia od body do tip
    body_diameter: float = 1.0   # mm — średnica trzonu
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
    p1 = top_pt + params.tip_height * d
    p2 = top_pt + total_h * d

    ring0 = top_pt + r_top * (ca[:, None] * u + sa[:, None] * w)
    ring1 = p1 + r_bot * (ca[:, None] * u + sa[:, None] * w)
    ring2 = p2 + r_bot * (ca[:, None] * u + sa[:, None] * w)

    verts = np.vstack([ring0, ring1, ring2, top_pt[None, :], p2[None, :]])
    top_center_idx = 3 * n
    bot_center_idx = 3 * n + 1

    faces: list[list[int]] = []
    # side cone (ring0 → ring1)
    for i in range(n):
        j = (i + 1) % n
        # winding: zewnętrzna normalna od osi pillara
        faces.append([i, j, n + j])
        faces.append([i, n + j, n + i])
    # side cylinder (ring1 → ring2)
    for i in range(n):
        j = (i + 1) % n
        faces.append([n + i, n + j, 2 * n + j])
        faces.append([n + i, 2 * n + j, 2 * n + i])
    # top cap (mały krążek, facing OPPOSITE dir_down — czyli "do góry")
    for i in range(n):
        j = (i + 1) % n
        faces.append([top_center_idx, i, j])
    # bottom cap (większy krążek, facing dir_down — czyli "w dół")
    for i in range(n):
        j = (i + 1) % n
        faces.append([bot_center_idx, 2 * n + j, 2 * n + i])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    # napraw orientacje normalnych (na wszelki wypadek)
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
