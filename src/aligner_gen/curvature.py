"""Curvature-based live-wire: snap konturu do gingival margin.

Idea: gingival margin (linia szyjki zęba) to **wklęsłe** przegięcie powierzchni
o dużym kącie dwuściennym. Liczymy per-edge ridge score (wysoki tam gdzie
wklęsły grzbiet), budujemy graf gdzie takie krawędzie są "tanie", i Dijkstra
między waypointami spływa do margin.

To path-based (1D ścieżka), nie region-based (2D flood) — dlatego NIE rozlewa
się jak naiwny curvature threshold.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def compute_edge_ridge_scores(
    mesh: trimesh.Trimesh,
    *,
    max_angle: float = 2.5,
    downward_thresh: float = -0.5,
    verbose: bool = True,
) -> np.ndarray:
    """Per-`edges_unique` ridge score (radiany kąta dwuściennego).

    Wysoki score = wklęsłe przegięcie o dużym kącie (kandydat na gingival
    margin / valley). Wypukłe krawędzie (cuspy, brzegi koron) dostają 0,
    żeby live-wire do nich NIE snapował. Płaskie powierzchnie ~0.

    Dwa filtry odsiewające szum NIEUSUWALNY przez denoise komponentowy
    (bo to wielkie spójne blob-y, nie kropki — głównie artefakt capping-u):
      A) `downward_thresh`: wyklucz krawędź jeśli którakolwiek sąsiadująca
         ściana patrzy mocno w dół (normal_z < thresh). Margin/policzkowa/
         okluzja nigdy nie patrzą w dół — to wycina zaszumioną podstawę/czapę.
      B) `max_angle`: odetnij zdegenerowane fałdy (kąt > ~2.5rad ≈ 145°).
         Gingival margin ma ~30–90°; π rad (180°) to złożona geometria
         (T-junction / fold z capping-u), nie anatomia.
    """
    n_verts = len(mesh.vertices)
    n_edges = len(mesh.edges_unique)
    ridge = np.zeros(n_edges, dtype=np.float64)

    fa_edges = mesh.face_adjacency_edges          # (E, 2) wewnętrzne krawędzie
    fa_angles = mesh.face_adjacency_angles        # (E,) kąt dwuścienny [rad]
    fa_convex = mesh.face_adjacency_convex        # (E,) True = wypukła
    fa_faces = mesh.face_adjacency                # (E, 2) indeksy sąsiadujących ścian

    if len(fa_edges) == 0:
        return ridge

    # A) base/czapa: którakolwiek ściana patrzy w dół
    nz = mesh.face_normals[fa_faces, 2]           # (E, 2) z-składowe normalnych
    is_base = np.min(nz, axis=1) < downward_thresh
    # B) zdegenerowane fałdy
    is_fold = fa_angles > max_angle

    # Tylko wklęsłe przegięcia liczą się jako ridge (margin to valley),
    # minus base i minus fałdy.
    exclude = fa_convex | is_base | is_fold
    scores = np.where(exclude, 0.0, fa_angles).astype(np.float64)

    if verbose:
        n_concave = int((~fa_convex).sum())
        n_kept = int((~exclude).sum())
        print(
            f"[curvature] filtr A/B: wklęsłych={n_concave}, "
            f"base(nz<{downward_thresh})={int((is_base & ~fa_convex).sum())}, "
            f"fold(>{max_angle:.1f}rad)={int((is_fold & ~fa_convex).sum())} "
            f"→ ridge zostaje {n_kept}"
        )

    # Wektoryzowane mapowanie face_adjacency_edges → edges_unique (po kluczu)
    eu_sorted = np.sort(mesh.edges_unique, axis=1)
    eu_keys = eu_sorted[:, 0].astype(np.int64) * n_verts + eu_sorted[:, 1]
    order = np.argsort(eu_keys)
    eu_keys_sorted = eu_keys[order]

    fa_sorted = np.sort(fa_edges, axis=1)
    fa_keys = fa_sorted[:, 0].astype(np.int64) * n_verts + fa_sorted[:, 1]

    pos = np.searchsorted(eu_keys_sorted, fa_keys)
    pos_clipped = np.clip(pos, 0, len(eu_keys_sorted) - 1)
    match = eu_keys_sorted[pos_clipped] == fa_keys
    ridge[order[pos_clipped[match]]] = scores[match]
    return ridge


def denoise_ridge(
    mesh: trimesh.Trimesh,
    edge_ridge: np.ndarray,
    *,
    angle_threshold: float = 0.14,
    min_component_edges: int = 25,
    verbose: bool = True,
) -> np.ndarray:
    """Usuń szum z ridge przez connected-component filtering.

    Gingival margin = długie spójne struktury krawędzi (linie ząb-dziąsło).
    Szum = małe izolowane kropki/paski. Próg jest **absolutny** (kąt dwuścienny
    w radianach, np. 0.14 rad ≈ 8°) — w przeciwieństwie do percentyla NIE
    fragmentuje linii o jednorodnej krzywiźnie. Potem connected components
    i zachowanie tylko dużych grup (≥ min_component_edges).
    """
    eu = mesh.edges_unique
    if not np.any(edge_ridge > 0):
        return edge_ridge.copy()

    strong_mask = edge_ridge >= angle_threshold
    strong_edges = eu[strong_mask]
    if len(strong_edges) == 0:
        if verbose:
            print(
                f"[curvature] denoise: 0 krawędzi >= {angle_threshold:.3f}rad "
                f"({np.degrees(angle_threshold):.1f}°) — obniż próg ([)"
            )
        return np.zeros_like(edge_ridge)

    n = len(mesh.vertices)
    row = np.concatenate([strong_edges[:, 0], strong_edges[:, 1]])
    col = np.concatenate([strong_edges[:, 1], strong_edges[:, 0]])
    g = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)

    edge_labels = labels[strong_edges[:, 0]]
    comp_edge_count = np.bincount(edge_labels, minlength=n_comp)
    keep_comp = comp_edge_count >= min_component_edges
    keep_edge = keep_comp[edge_labels]

    out = np.zeros_like(edge_ridge)
    strong_idx = np.where(strong_mask)[0]
    out[strong_idx[keep_edge]] = edge_ridge[strong_idx[keep_edge]]

    if verbose:
        n_strong = int(strong_mask.sum())
        n_kept = int(keep_edge.sum())
        n_big = int(keep_comp.sum())
        print(
            f"[curvature] denoise: prog={angle_threshold:.3f}rad "
            f"({np.degrees(angle_threshold):.1f}°), strong={n_strong}, "
            f"kept={n_kept} w {n_big} komponentach (min {min_component_edges} edges), "
            f"usunieto {n_strong - n_kept} szumu"
        )
    return out


def build_snap_graph(
    mesh: trimesh.Trimesh,
    ridge_scores: np.ndarray,
    *,
    strength: float = 25.0,
    eps: float = 0.05,
    ridge_percentile: float = 88.0,
    verbose: bool = True,
) -> csr_matrix:
    """Graf krawędziowy z wagami snap-ującymi do gingival margin.

    waga = edge_length * (eps + (1 - norm_ridge) * strength)

    - silny ridge (margin)  → norm_ridge≈1 → waga ≈ edge_length * eps (tania)
    - brak ridge (flat/cusp)→ norm_ridge≈0 → waga ≈ edge_length * strength (droga)

    Dijkstra na tym grafie woli iść wzdłuż margin niż na skróty.
    """
    edges = mesh.edges_unique
    v = mesh.vertices
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)

    positive = ridge_scores[ridge_scores > 0]
    if positive.size > 0:
        ridge_ref = np.percentile(positive, ridge_percentile)
    else:
        ridge_ref = 1.0
    ridge_ref = max(float(ridge_ref), 1e-6)

    norm_ridge = np.clip(ridge_scores / ridge_ref, 0.0, 1.0)
    edge_cost = lengths * (eps + (1.0 - norm_ridge) * strength)

    if verbose:
        print(
            f"[curvature] snap graph: strength={strength}, eps={eps}, "
            f"ridge_ref={ridge_ref:.3f}rad (pct={ridge_percentile}), "
            f"waga=[{edge_cost.min():.3f}, {edge_cost.max():.3f}]"
        )

    n = len(v)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_cost, edge_cost])
    return csr_matrix((data, (row, col)), shape=(n, n))


def vertex_ridge_scores(mesh: trimesh.Trimesh, edge_ridge: np.ndarray) -> np.ndarray:
    """Rozprowadź per-edge ridge score na wierzchołki (max sąsiednich krawędzi).

    Do wizualizacji heatmapy — pokazuje gdzie live-wire będzie przyciągać.
    """
    n_verts = len(mesh.vertices)
    out = np.zeros(n_verts, dtype=np.float64)
    edges = mesh.edges_unique
    np.maximum.at(out, edges[:, 0], edge_ridge)
    np.maximum.at(out, edges[:, 1], edge_ridge)
    return out
