"""Selekcja po powierzchni mesha:

- shortest_path / shortest_path_along_mesh — geodesic łańcuch wierzchołków
  pomiędzy dwoma punktami (Dijkstra po krawędziach).
- fill_interior — po zamknięciu pętli (zbioru wierzchołków boundary)
  zwraca maskę „wnętrza" (mniejsza spójna komponenta po wycięciu boundary).
- geodesic_ball — zachowane na wypadek gdybyśmy chcieli wrócić do brusha.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def build_edge_graph(mesh: trimesh.Trimesh) -> csr_matrix:
    """Sparse adjacency: edge length as weight, symmetric."""
    edges = mesh.edges_unique
    v = mesh.vertices
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    n = len(v)
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([lengths, lengths])
    return csr_matrix((data, (row, col)), shape=(n, n))


def nearest_vertex(mesh: trimesh.Trimesh, point: np.ndarray) -> int:
    """Indeks vertex najbliższy punktowi 3D (euklidesowo)."""
    diff = mesh.vertices - np.asarray(point, dtype=float)
    return int(np.argmin(np.einsum("ij,ij->i", diff, diff)))


def shortest_path(graph: csr_matrix, src: int, dst: int) -> np.ndarray:
    """Geodesic najkrótsza ścieżka po krawędziach mesha: ciąg vertex idx
    od `src` do `dst` włącznie. Pusta tablica jeśli brak ścieżki."""
    _, pred = dijkstra(
        graph, indices=src, return_predecessors=True, directed=False
    )
    if pred[dst] < 0 and src != dst:
        return np.empty(0, dtype=np.int64)
    path = [dst]
    while path[-1] != src:
        nxt = int(pred[path[-1]])
        if nxt < 0:
            return np.empty(0, dtype=np.int64)
        path.append(nxt)
    path.reverse()
    return np.asarray(path, dtype=np.int64)


def geodesic_ball(graph: csr_matrix, start: int, radius: float) -> np.ndarray:
    """Bool maska wierzchołków w geodesic radius od `start`."""
    dist = dijkstra(graph, indices=start, limit=radius, directed=False)
    return np.isfinite(dist) & (dist <= radius)


def fill_interior(
    graph: csr_matrix,
    boundary_mask: np.ndarray,
    *,
    prefer: str = "smaller",
    verbose: bool = True,
) -> np.ndarray:
    """Znajdź wnętrze zamkniętej pętli na meshu.

    Etapy:
      1. Wytnij wierzchołki boundary z grafu.
      2. Spójne komponenty w pozostałej części.
      3. **Filtr "touching boundary"**: zostaw tylko te komponenty,
         które mają co najmniej jednego sąsiada na boundary. To eliminuje
         oderwane artefakty STL (floating islands), które nie mają nic
         wspólnego z narysowanym konturem.
      4. Spośród pozostałych wybierz mniejszą / większą.

    Boundary nie jest częścią zwracanej maski — viewer łączy je osobno.
    """
    n = boundary_mask.shape[0]
    interior_candidates = ~boundary_mask
    if not interior_candidates.any() or not boundary_mask.any():
        return np.zeros(n, dtype=bool)

    idx_map = np.where(interior_candidates)[0]
    inv_map = -np.ones(n, dtype=np.int64)
    inv_map[idx_map] = np.arange(idx_map.size)

    sub_graph = graph[interior_candidates][:, interior_candidates]
    n_comp, labels = connected_components(sub_graph, directed=False)
    if n_comp == 0:
        return np.zeros(n, dtype=bool)

    comp_sizes = np.bincount(labels)

    # Znajdź komponenty stykające się z boundary (sąsiad w grafie ∈ boundary)
    graph_csr = graph.tocsr()
    boundary_indices = np.where(boundary_mask)[0]
    touching: set[int] = set()
    for bi in boundary_indices:
        start, end = graph_csr.indptr[bi], graph_csr.indptr[bi + 1]
        neighbors = graph_csr.indices[start:end]
        for nb in neighbors:
            if not boundary_mask[nb]:
                touching.add(int(labels[inv_map[nb]]))

    if verbose:
        n_touching = len(touching)
        sizes_touching = sorted([int(comp_sizes[c]) for c in touching])
        print(
            f"[fill_interior] komponentów: {n_comp} (z {n} verts po wycięciu boundary), "
            f"stykających z boundary: {n_touching}, rozmiary: {sizes_touching}"
        )

    if not touching:
        if verbose:
            print("[fill_interior] żadna komponenta nie styka się z boundary — pusta maska.")
        return np.zeros(n, dtype=bool)

    if len(touching) == 1:
        if verbose:
            print(
                "[fill_interior] tylko 1 komponenta styka się z boundary — "
                "pętla nie izoluje regionu. Sprawdź czy kontur jest zamknięty "
                "bez przerw (waypointy za rzadko?)."
            )
        # Zwracamy tę jedną — lepsze niż nic, ale ostrzegamy
        target = next(iter(touching))
    else:
        # Wybierz wśród stykających po preferencji
        ordered = sorted(touching, key=lambda c: int(comp_sizes[c]))
        target = ordered[0] if prefer == "smaller" else ordered[-1]
        if verbose:
            print(
                f"[fill_interior] wybrano komponentę {target} "
                f"({int(comp_sizes[target])} verts, prefer={prefer})"
            )

    mask = np.zeros(n, dtype=bool)
    mask[idx_map[labels == target]] = True
    return mask


def fill_from_seed(
    graph: csr_matrix,
    boundary_mask: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Flood fill po meshu od `seed`, blokowany przez `boundary_mask`.

    Zwraca maskę spójnej komponenty zawierającej seed, w subgraphie po
    usunięciu boundary. Jeśli seed sam jest na boundary, zwraca pustkę.
    """
    n = boundary_mask.shape[0]
    if boundary_mask[seed]:
        return np.zeros(n, dtype=bool)

    interior_candidates = ~boundary_mask
    idx_map = np.where(interior_candidates)[0]
    inv_map = -np.ones(n, dtype=np.int64)
    inv_map[idx_map] = np.arange(idx_map.size)

    sub_graph = graph[interior_candidates][:, interior_candidates]
    _, labels = connected_components(sub_graph, directed=False)
    seed_label = labels[inv_map[seed]]

    mask = np.zeros(n, dtype=bool)
    mask[idx_map[labels == seed_label]] = True
    return mask
