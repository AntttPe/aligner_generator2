"""Smoke test pipelinu nakładki na syntetycznej sferze.

Sfera = "ząb". Wybieramy czapę wokół bieguna +z jako "powierzchnia
przylegania". Pipeline powinien wygenerować cienką skorupę grubości
~0.6 mm na zewnątrz czapy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aligner_gen.aligner import AlignerParams, generate_aligner  # noqa
from aligner_gen.selection import (  # noqa
    build_edge_graph,
    fill_interior,
    nearest_vertex,
    shortest_path,
)


def main() -> int:
    # Sfera o promieniu 10 mm = przybliżenie zęba
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=10.0)
    print(f"[smoke] sfera: {len(sphere.vertices)} verts, watertight={sphere.is_watertight}")

    # Selekcja: czapa wokół bieguna +z, kontur na koło ~30° od bieguna
    graph = build_edge_graph(sphere)
    waypoints_xyz = []
    n_wp = 8
    angle = np.deg2rad(30.0)
    r = 10.0
    for k in range(n_wp):
        phi = 2 * np.pi * k / n_wp
        waypoints_xyz.append(
            [
                r * np.sin(angle) * np.cos(phi),
                r * np.sin(angle) * np.sin(phi),
                r * np.cos(angle),
            ]
        )
    wp_idx = [nearest_vertex(sphere, np.asarray(p)) for p in waypoints_xyz]

    boundary = np.zeros(len(sphere.vertices), dtype=bool)
    closed_wp = wp_idx + [wp_idx[0]]
    for a, b in zip(closed_wp[:-1], closed_wp[1:]):
        seg = shortest_path(graph, a, b)
        boundary[seg] = True
    interior = fill_interior(graph, boundary, prefer="smaller", verbose=False)
    selection = boundary | interior
    print(
        f"[smoke] selekcja: {selection.sum()} verts (boundary {boundary.sum()} + interior {interior.sum()})"
    )

    # Pipeline
    params = AlignerParams(
        thickness=0.6,
        inner_clearance=0.05,
        close_radius=1.0,
        selection_radius=1.5,
        voxel_pitch=0.2,  # nieco grubsze na smoke test żeby było szybko
    )
    aligner_mesh, rep = generate_aligner(sphere, selection, params)

    if aligner_mesh is None:
        print(f"[smoke] FAIL: {rep.notes}")
        return 1

    # Sanity check: nakładka powinna leżeć po stronie +z, blisko powierzchni sfery
    aligner_verts = np.asarray(aligner_mesh.vertices)
    z_range = (aligner_verts[:, 2].min(), aligner_verts[:, 2].max())
    radii = np.linalg.norm(aligner_verts, axis=1)
    r_range = (radii.min(), radii.max())
    print(f"[smoke] aligner z range: [{z_range[0]:.2f}, {z_range[1]:.2f}] mm")
    print(f"[smoke] aligner radii: [{r_range[0]:.2f}, {r_range[1]:.2f}] mm")

    # Powinno być:
    # - większość po stronie +z (z > 0 dla czapy)
    # - promienie w okolicach 10 + clearance ... 10 + clearance + thickness = 10.05 ... 10.65
    assert z_range[0] > 0, f"Aligner przecieka pod równik: {z_range}"
    # Tolerancja: 1 voxel pitch (0.2 mm) — typowa precyzja voxelizacji+MC
    pitch = params.voxel_pitch
    expected_inner = 10.0 + params.inner_clearance
    expected_outer = 10.0 + params.inner_clearance + params.thickness
    assert r_range[0] >= expected_inner - pitch, (
        f"Aligner za blisko/wewnątrz sfery: r_min={r_range[0]:.3f}, expected ≥ {expected_inner - pitch:.3f}"
    )
    assert r_range[1] <= expected_outer + pitch + params.close_radius, (
        f"Aligner za daleko od sfery: r_max={r_range[1]:.3f}, expected ≤ {expected_outer + pitch + params.close_radius:.3f}"
    )

    # Eksport STL
    out = Path("/tmp/aligner_smoke.stl")
    aligner_mesh.export(out)
    print(f"[smoke] zapisano: {out} ({out.stat().st_size // 1024} KB)")

    print(f"\n[smoke] ✓ pipeline OK ({rep.total_seconds:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
