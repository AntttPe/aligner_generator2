"""Smoke test bez okna viewera — kontur (shortest_path) + fill (interior)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aligner_gen.io import LoadedMesh, save_selection, load_selection, load_stl  # noqa
from aligner_gen.selection import (  # noqa
    build_edge_graph,
    fill_interior,
    geodesic_ball,
    nearest_vertex,
    shortest_path,
)


def main() -> int:
    # Syntetyczny mesh — sfera
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
    print(f"[smoke] sfera: {len(sphere.vertices)} verts, {len(sphere.faces)} faces")

    graph = build_edge_graph(sphere)
    print(f"[smoke] graf: nnz={graph.nnz}")

    # ---- test 1: geodesic ball (regression) ----
    pole = np.array([0.0, 0.0, 5.0])
    pole_idx = nearest_vertex(sphere, pole)
    ball = geodesic_ball(graph, pole_idx, radius=1.5)
    assert ball.sum() > 0 and sphere.vertices[ball, 2].min() > 0
    print(f"[smoke] geodesic_ball OK ({ball.sum()} verts, wszystkie z>0)")

    # ---- test 2: shortest_path ----
    src = nearest_vertex(sphere, np.array([5.0, 0.0, 0.0]))
    dst = nearest_vertex(sphere, np.array([0.0, 5.0, 0.0]))
    path = shortest_path(graph, src, dst)
    assert path.size >= 2 and path[0] == src and path[-1] == dst
    print(f"[smoke] shortest_path OK: {path.size} vertices wzdłuż ćwiartki")

    # ---- test 3: contour + fill_interior ----
    # 4 waypointy tworzące kwadrat wokół bieguna +z, każda około 30 stopni
    angle = np.deg2rad(30.0)
    r = 5.0
    waypoints_xyz = [
        [r * np.sin(angle), 0.0, r * np.cos(angle)],
        [0.0, r * np.sin(angle), r * np.cos(angle)],
        [-r * np.sin(angle), 0.0, r * np.cos(angle)],
        [0.0, -r * np.sin(angle), r * np.cos(angle)],
    ]
    wp_idx = [nearest_vertex(sphere, np.asarray(p)) for p in waypoints_xyz]
    print(f"[smoke] waypointy: {wp_idx}")

    # Budujemy boundary jako geodesic paths między kolejnymi (z zamknięciem)
    boundary = np.zeros(len(sphere.vertices), dtype=bool)
    closed_wp = wp_idx + [wp_idx[0]]
    for a, b in zip(closed_wp[:-1], closed_wp[1:]):
        seg = shortest_path(graph, a, b)
        boundary[seg] = True
    print(f"[smoke] boundary: {boundary.sum()} verts")

    # Fill (mniejsza komponenta = czapa wokół bieguna)
    interior = fill_interior(graph, boundary, prefer="smaller")
    assert interior.any(), "fill_interior zwrócił pustkę"
    z_interior = sphere.vertices[interior, 2]
    print(
        f"[smoke] interior: {interior.sum()} verts, "
        f"z ∈ [{z_interior.min():.2f}, {z_interior.max():.2f}]"
    )
    assert z_interior.min() > 0, "Interior zawiera wierzchołki z drugiej półkuli!"
    # Sanity: interior < druga strona
    larger = fill_interior(graph, boundary, prefer="larger")
    assert larger.sum() > interior.sum()
    print(f"[smoke] interior < larger: {interior.sum()} < {larger.sum()} ✓")

    # ---- test 4: save/load round-trip ----
    tmp_stl = Path("/tmp/aligner_gen_smoke.stl")
    sphere.export(tmp_stl)
    loaded = load_stl(tmp_stl)
    combined = boundary | interior
    save_path = save_selection(loaded, combined)
    loaded_back = load_selection(loaded)
    assert loaded_back is not None and np.array_equal(loaded_back, combined)
    print(f"[smoke] save/load OK ({save_path.name}, {combined.sum()} verts)")

    print("\n[smoke] ✓ wszystkie testy zaliczone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
