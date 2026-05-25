"""Interactive viewer — rysowanie konturu + fill wnętrza.

Workflow
--------
1. Klikasz punkty na meshu — każdy punkt to waypoint. Kolejne waypointy są
   automatycznie łączone najkrótszą ścieżką po krawędziach mesha (geodesic).
2. Kliknij blisko **pierwszego** waypointa (lub naciśnij C) aby zamknąć pętlę.
3. F lub przycisk [Fill] — wypełnia wnętrze pętli kolorem (oznacza powierzchnię
   przylegania nakładki).
4. S lub przycisk [Save] — zapisuje selekcję dla późniejszych iteracji
   algorytmu generowania (debug — żeby nie malować od nowa).

Hotkeye
-------
C  zamknij pętlę                  I  odwróć fill (mniejsza/większa strona)
F  wypełnij wnętrze               Z  usuń ostatni waypoint
X  wyczyść wszystko               S  zapisz selekcję
L  wczytaj selekcję               Q/Esc  wyjście
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from pathlib import Path

import trimesh

from .aligner import AlignerParams, generate_aligner
from .curvature import (
    build_snap_graph,
    compute_edge_ridge_scores,
    denoise_ridge,
    vertex_ridge_scores,
)
from .io import LoadedMesh, load_selection, save_selection
from .selection import (
    build_edge_graph,
    fill_from_seed,
    fill_interior,
    nearest_vertex,
    shortest_path,
)

ALIGNER_COLOR = "#06b6d4"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "output"


SCALAR_CMAP = ["#cfcfcf", "#dc2f5a"]
CONTOUR_COLOR = "#22d3ee"
WAYPOINT_COLOR = "#fbbf24"
FIRST_WAYPOINT_COLOR = "#22c55e"
CLOSE_LOOP_THRESHOLD_FACTOR = 0.02  # % bbox diagonal — jak blisko 1. waypoint = close


class SelectionViewer:
    def __init__(self, loaded: LoadedMesh):
        self.loaded = loaded
        self.mesh = loaded.mesh

        # state
        self.waypoints: list[int] = []
        self.path_segments: list[np.ndarray] = []
        self.boundary_mask = np.zeros(len(self.mesh.vertices), dtype=bool)
        self.fill_mask = np.zeros(len(self.mesh.vertices), dtype=bool)
        self.is_closed = False
        self._awaiting_seed = False  # po G — następny klik to ziarno fillu

        # live-wire / trim mode: "manual" (geodesic) | "scalloped" (snap do margin)
        self.trim_mode = "manual"
        self._snap_graph = None        # lazy — curvature-weighted graph
        self._edge_ridge = None        # per-edge ridge scores PO denoise (cache)
        self._curvature_view = False   # heatmapa krzywizny ON/OFF
        # parametry live-wire (tunable — [ ] zmieniają próg kąta, ; ' rozmiar komponentu)
        self._angle_threshold = 0.14   # próg kąta dwuściennego [rad] (~8°) dla silnego ridge
        self._min_comp_edges = 25      # min krawędzi w spójnym komponencie (denoise)
        self._snap_strength = 25.0     # jak mocno snap przyciąga do margin
        # auto-trace (Etap 1): snap klikniętego waypointa do najbliższego
        # wierzchołka margin — pozwala klikać z grubsza, punkt sam ląduje na linii.
        self._ridge_kdtree = None      # lazy cKDTree pozycji wierzchołków ridge
        self._ridge_vert_idx = None    # indeksy wierzchołków z ridge>0
        self._snap_radius = 3.0        # [mm] max odległość snapu waypointa do margin

        # aligner state
        self.aligner_mesh: trimesh.Trimesh | None = None
        self._aligner_actor = None
        self._aligner_opacity = 0.85
        self.params = AlignerParams()
        # quality mode: "preview" = pitch 0.20 (~12s), "final" = pitch 0.10 (~90s)
        self._quality_mode = "final"  # zaczynamy od final bo direct-print
        self.params.voxel_pitch = 0.10

        # geometry / threshold scale
        bbox = self.mesh.bounds
        self._bbox_diag = float(np.linalg.norm(bbox[1] - bbox[0]))
        self._close_thresh = CLOSE_LOOP_THRESHOLD_FACTOR * self._bbox_diag
        self._waypoint_radius = 0.004 * self._bbox_diag

        print("[viewer] Buduję graf krawędzi (Dijkstra)...")
        self.graph = build_edge_graph(self.mesh)

        # auto-load — jeśli istnieje saved selection, wczytaj jako fill_mask
        loaded_mask = load_selection(loaded)
        if loaded_mask is not None:
            self.fill_mask = loaded_mask
            print(
                f"[viewer] Auto-load: {self.fill_mask.sum()} verts "
                f"(zaznacz jako wypełnienie, bez konturu)"
            )

        # pyvista mesh
        faces_flat = np.hstack(
            [np.full((len(self.mesh.faces), 1), 3, dtype=np.int64), self.mesh.faces]
        ).ravel()
        self.pv_mesh = pv.PolyData(np.asarray(self.mesh.vertices), faces_flat)
        self.pv_mesh.point_data["selected"] = self._combined_mask().astype(np.uint8)

        self.plotter = pv.Plotter(window_size=(1400, 900))
        self.plotter.set_background("#1a1a1a")
        self.plotter.add_axes()

        self._mesh_actor = self.plotter.add_mesh(
            self.pv_mesh,
            scalars="selected",
            cmap=SCALAR_CMAP,
            clim=(0, 1),
            show_scalar_bar=False,
            smooth_shading=True,
            interpolate_before_map=True,
            specular=0.25,
            specular_power=12,
            name="mesh_actor",
        )

        self._contour_actor = None
        self._waypoint_actor = None
        self._first_waypoint_actor = None

        self._setup_picking()
        self._setup_keys()
        self._setup_buttons()
        self._setup_sliders()
        self._refresh_all()

    # ---------- maski ----------
    def _combined_mask(self) -> np.ndarray:
        return self.boundary_mask | self.fill_mask

    # ---------- input setup ----------
    def _setup_picking(self) -> None:
        try:
            self.plotter.enable_surface_point_picking(
                callback=self._on_pick,
                show_message=False,
                show_point=False,
                left_clicking=True,
            )
        except (AttributeError, TypeError):
            self.plotter.enable_point_picking(
                callback=self._on_pick,
                show_message=False,
                show_point=False,
                left_clicking=True,
                use_picker="cell",
                pickable_window=False,
            )

    def _setup_keys(self) -> None:
        self.plotter.add_key_event("c", self._close_loop)
        self.plotter.add_key_event("f", self._fill)
        self.plotter.add_key_event("g", self._arm_seed_fill)
        self.plotter.add_key_event("i", self._invert_fill)
        self.plotter.add_key_event("z", self._undo)
        self.plotter.add_key_event("x", self._clear)
        self.plotter.add_key_event("s", self._save)
        self.plotter.add_key_event("l", self._reload)
        self.plotter.add_key_event("n", self._generate_aligner)
        self.plotter.add_key_event("e", self._export_aligner)
        self.plotter.add_key_event("h", self._toggle_aligner_visibility)
        self.plotter.add_key_event("m", self._toggle_mesh_visibility)
        self.plotter.add_key_event("p", self._toggle_quality_mode)
        self.plotter.add_key_event("w", self._toggle_trim_mode)
        self.plotter.add_key_event("v", self._toggle_curvature_view)
        # live-tuning denoise (z włączoną heatmapą V widać efekt od razu)
        self.plotter.add_key_event("bracketleft", lambda: self._adjust_angle_threshold(-2))
        self.plotter.add_key_event("bracketright", lambda: self._adjust_angle_threshold(+2))
        self.plotter.add_key_event("semicolon", lambda: self._adjust_min_comp(-10))
        self.plotter.add_key_event("apostrophe", lambda: self._adjust_min_comp(+10))

    def _setup_buttons(self) -> None:
        """Klikalne buttony — duplikat hotkeyów dla wygody.

        Pozycje w pikselach od dolnego lewego rogu okna. Każdy button to
        checkbox_widget z ignorowanym stanem (action-on-click).
        """
        specs = [
            ("FILL [F]",      self._fill,                    "#22c55e",   20),
            ("SEED [G]",      self._arm_seed_fill,           "#84cc16",  130),
            ("CLOSE [C]",     self._close_loop,              "#22d3ee",  240),
            ("UNDO [Z]",      self._undo,                    "#fbbf24",  370),
            ("CLEAR [X]",     self._clear,                   "#94a3b8",  480),
            ("SAVE [S]",      self._save,                    "#dc2f5a",  590),
            ("LOAD [L]",      self._reload,                  "#a78bfa",  700),
            ("GENERATE [N]",  self._generate_aligner,        "#f97316",  810),
            ("EXPORT [E]",    self._export_aligner,          "#10b981",  960),
            ("ALIGNER [H]",   self._toggle_aligner_visibility, "#64748b", 1090),
            ("TEETH [M]",     self._toggle_mesh_visibility,    "#a3a3a3", 1230),
            ("QUALITY [P]",   self._toggle_quality_mode,       "#ec4899", 1340),
            ("TRIM [W]",      self._toggle_trim_mode,          "#14b8a6", 1450),
            ("CURV [V]",      self._toggle_curvature_view,     "#eab308", 1560),
        ]
        y = 16
        for label, cb, color, x in specs:
            self.plotter.add_checkbox_button_widget(
                self._wrap_action(cb),
                value=False,
                position=(x, y),
                size=28,
                border_size=2,
                color_on=color,
                color_off=color,
                background_color="#333333",
            )
            self.plotter.add_text(
                label,
                position=(x + 34, y + 6),
                font_size=9,
                color="#e5e5e5",
                name=f"btn_label_{label}",
            )

    @staticmethod
    def _wrap_action(cb):
        def _inner(_value):
            cb()
        return _inner

    def _setup_sliders(self) -> None:
        """Slidery dla najważniejszych parametrów. Wartość self.params jest
        aktualizowana na żywo, ale regeneracja wymaga naciśnięcia N (slider
        nie triggeruje 25s pipelineu sam z siebie)."""
        # Offset / inner_clearance: 0.05–0.30 mm (visible przy pitch 0.15mm)
        self.plotter.add_slider_widget(
            callback=self._on_clearance_change,
            rng=(0.05, 0.30),
            value=max(0.05, min(self.params.inner_clearance, 0.30)),
            title="Offset od zebow [mm]",
            pointa=(0.72, 0.94),
            pointb=(0.98, 0.94),
            style="modern",
            fmt="%.3f",
            title_height=0.018,
            slider_width=0.025,
            tube_width=0.005,
        )
        # Grubość / thickness: 0.5–1.5 mm
        self.plotter.add_slider_widget(
            callback=self._on_thickness_change,
            rng=(0.5, 1.5),
            value=self.params.thickness,
            title="Grubosc nakladki [mm]",
            pointa=(0.72, 0.82),
            pointb=(0.98, 0.82),
            style="modern",
            fmt="%.2f",
            title_height=0.018,
            slider_width=0.025,
            tube_width=0.005,
        )
        # Fillet radius — zaokrąglenie krawędzi: 0.0–1.5 mm
        self.plotter.add_slider_widget(
            callback=self._on_fillet_change,
            rng=(0.0, 1.5),
            value=self.params.fillet_radius,
            title="Zaokraglenie krawedzi [mm]",
            pointa=(0.72, 0.70),
            pointb=(0.98, 0.70),
            style="modern",
            fmt="%.2f",
            title_height=0.018,
            slider_width=0.025,
            tube_width=0.005,
        )
        # Trim smoothing — gładkość przebiegu krawędzi: 0–12 voxeli
        self.plotter.add_slider_widget(
            callback=self._on_trim_smooth_change,
            rng=(0.0, 12.0),
            value=self.params.trim_smooth_sigma,
            title="Gladkosc krawedzi (trim)",
            pointa=(0.72, 0.58),
            pointb=(0.98, 0.58),
            style="modern",
            fmt="%.1f",
            title_height=0.018,
            slider_width=0.025,
            tube_width=0.005,
        )

    def _on_clearance_change(self, value: float) -> None:
        self.params.inner_clearance = float(value)
        self._refresh_overlay()

    def _on_thickness_change(self, value: float) -> None:
        self.params.thickness = float(value)
        self._refresh_overlay()

    def _on_fillet_change(self, value: float) -> None:
        self.params.fillet_radius = float(value)
        self._refresh_overlay()

    def _on_trim_smooth_change(self, value: float) -> None:
        self.params.trim_smooth_sigma = float(value)
        self._refresh_overlay()

    # ---------- callbacks ----------
    def _on_pick(self, point, *_args, **_kwargs) -> None:
        if point is None:
            return
        pt = np.asarray(point, dtype=float).reshape(-1)
        if pt.size < 3:
            return
        new_idx = nearest_vertex(self.mesh, pt[:3])

        # seed mode — klik wybiera ziarno fillu
        if self._awaiting_seed:
            self._awaiting_seed = False
            if not self.is_closed:
                print("[viewer] SEED anulowany — najpierw zamknij pętlę (C).")
                self._refresh_overlay()
                return
            if self.boundary_mask[new_idx]:
                print("[viewer] SEED na boundary — kliknij dalej od konturu.")
                self._refresh_overlay()
                return
            interior = fill_from_seed(self.graph, self.boundary_mask, new_idx)
            self.fill_mask = interior
            print(f"[viewer] SEED fill: {interior.sum()} verts (od klikniętego ziarna).")
            self._refresh_all()
            return

        if self.is_closed:
            print("[viewer] Pętla zamknięta. F=fill, G=seed-fill, X=wyczyść, Z=cofnij.")
            return

        # auto-trace (Etap 1): w scalloped waypoint sam ląduje na margin
        new_idx = self._snap_to_ridge(new_idx)

        # Czy klik blisko pierwszego waypointa → zamknij pętlę
        if len(self.waypoints) >= 3:
            first = self.waypoints[0]
            d = float(
                np.linalg.norm(self.mesh.vertices[new_idx] - self.mesh.vertices[first])
            )
            if d < self._close_thresh:
                self._close_loop()
                return

        self.waypoints.append(new_idx)
        if len(self.waypoints) >= 2:
            seg = shortest_path(self._path_graph(), self.waypoints[-2], new_idx)
            if seg.size == 0:
                print("[viewer] Brak ścieżki między waypointami.")
                self.waypoints.pop()
                return
            self.path_segments.append(seg)
            self.boundary_mask[seg] = True
        else:
            self.boundary_mask[new_idx] = True
            # NIE czyścimy fill_mask — żeby przypadkowy klik nie zniszczył
            # auto-loadowanej selekcji. _fill (F) i tak zastąpi fill_mask
            # nowym wnętrzem konturu, X jawnie czyści wszystko.

        self._refresh_all()

    def _close_loop(self) -> None:
        if self.is_closed:
            return
        if len(self.waypoints) < 3:
            print("[viewer] Potrzeba ≥3 waypointów żeby zamknąć pętlę.")
            return
        seg = shortest_path(self._path_graph(), self.waypoints[-1], self.waypoints[0])
        if seg.size == 0:
            print("[viewer] Nie udało się znaleźć ścieżki zamykającej pętlę.")
            return
        self.path_segments.append(seg)
        self.boundary_mask[seg] = True
        self.is_closed = True
        print("[viewer] Pętla zamknięta — auto-fill wnętrza...")
        self._refresh_all()
        self._fill()  # auto-fill (Etap 1) — bez ręcznego F; jak pusto, info o G

    def _fill(self) -> None:
        if not self.is_closed:
            print("[viewer] Najpierw zamknij pętlę (C lub klik blisko 1. waypointa).")
            return
        interior = fill_interior(self.graph, self.boundary_mask, prefer="smaller")
        if not interior.any():
            print(
                "[viewer] Wynik pusty. Spróbuj G (seed-fill) — klik wewnątrz pętli."
            )
            return
        self.fill_mask = interior
        print(f"[viewer] Wypełniono wnętrze: {interior.sum()} verts (auto, smaller).")
        self._refresh_all()

    def _arm_seed_fill(self) -> None:
        if not self.is_closed:
            print("[viewer] SEED dostępny dopiero po zamknięciu pętli (C).")
            return
        self._awaiting_seed = True
        print("[viewer] SEED mode: kliknij punkt WEWNĄTRZ pętli aby wypełnić.")
        self._refresh_overlay()

    # ---------- live-wire (curvature snap) ----------
    def _ensure_ridge(self):
        """Lazy compute denoised ridge (raz). Używane przez heatmapę i snap."""
        if self._edge_ridge is None:
            import numpy as _np
            print(
                f"[viewer] Liczę krzywiznę: angle_threshold={self._angle_threshold:.3f}rad "
                f"({_np.degrees(self._angle_threshold):.1f}°), "
                f"min_comp_edges={self._min_comp_edges}, snap_strength={self._snap_strength}"
            )
            raw = compute_edge_ridge_scores(self.mesh)
            print(
                f"[curvature] raw ridge: {np.count_nonzero(raw)} concave edges "
                f"(z {len(raw)} total), max={raw.max():.3f}rad"
            )
            self._edge_ridge = denoise_ridge(
                self.mesh,
                raw,
                angle_threshold=self._angle_threshold,
                min_component_edges=self._min_comp_edges,
            )
        return self._edge_ridge

    def _ensure_snap_graph(self):
        """Lazy build snap graph z denoised ridge."""
        if self._snap_graph is None:
            ridge = self._ensure_ridge()
            self._snap_graph = build_snap_graph(
                self.mesh, ridge, strength=self._snap_strength
            )
        return self._snap_graph

    def _path_graph(self):
        """Graf do rysowania ścieżki konturu: snap (scalloped) lub geodesic (manual)."""
        if self.trim_mode == "scalloped":
            return self._ensure_snap_graph()
        return self.graph

    def _ensure_ridge_kdtree(self):
        """Lazy cKDTree wierzchołków margin (ridge>0) do snapu waypointów."""
        if self._ridge_kdtree is None:
            from scipy.spatial import cKDTree

            ridge = self._ensure_ridge()
            vr = vertex_ridge_scores(self.mesh, ridge)
            self._ridge_vert_idx = np.where(vr > 0)[0]
            if self._ridge_vert_idx.size > 0:
                self._ridge_kdtree = cKDTree(self.mesh.vertices[self._ridge_vert_idx])
        return self._ridge_kdtree

    def _snap_to_ridge(self, idx: int) -> int:
        """Scalloped: przyciągnij kliknięty wierzchołek do najbliższego punktu
        margin (ridge>0) w promieniu `_snap_radius`. Pozwala klikać z grubsza —
        waypoint sam ląduje na linii zębodołowej. W manual: bez zmian."""
        if self.trim_mode != "scalloped":
            return idx
        tree = self._ensure_ridge_kdtree()
        if tree is None:
            return idx
        d, j = tree.query(self.mesh.vertices[idx])
        if d <= self._snap_radius:
            return int(self._ridge_vert_idx[j])
        print(
            f"[viewer] snap: brak margin w {self._snap_radius:.1f}mm "
            f"(najbliższy {d:.1f}mm) — waypoint bez snapu, klik bliżej linii."
        )
        return idx

    def _invalidate_ridge(self) -> None:
        """Wyrzuć cache ridge/snap — wymusi przeliczenie z nowymi parametrami."""
        self._edge_ridge = None
        self._snap_graph = None
        self._ridge_kdtree = None
        self._ridge_vert_idx = None

    def _recompute_and_show_ridge(self) -> None:
        """Przelicz ridge i odśwież heatmapę jeśli włączona."""
        self._invalidate_ridge()
        ridge = self._ensure_ridge()
        if self._curvature_view:
            vr = vertex_ridge_scores(self.mesh, ridge)
            self.pv_mesh.point_data["ridge"] = vr.astype(np.float32)
            self._rebuild_mesh_actor(scalars="ridge")
            self.plotter.render()

    def _adjust_angle_threshold(self, delta_deg: float) -> None:
        delta_rad = np.radians(delta_deg)
        self._angle_threshold = float(np.clip(self._angle_threshold + delta_rad, 0.02, 0.7))
        print(
            f"[viewer] angle_threshold = {self._angle_threshold:.3f}rad "
            f"({np.degrees(self._angle_threshold):.1f}°) — niżej = więcej krawędzi"
        )
        self._recompute_and_show_ridge()

    def _adjust_min_comp(self, delta: int) -> None:
        self._min_comp_edges = int(max(1, self._min_comp_edges + delta))
        print(f"[viewer] min_comp_edges = {self._min_comp_edges} (wyżej = mniej szumu)")
        self._recompute_and_show_ridge()

    def _toggle_trim_mode(self) -> None:
        if self.trim_mode == "manual":
            self.trim_mode = "scalloped"
            self._ensure_snap_graph()
            print(
                "[viewer] TRIM: SCALLOPED (auto-trace) — waypoint I ścieżka "
                "snap-ują do gingival margin. Klikaj ~5 zgrubnych kotwic wokół "
                f"łuku → C zamyka → auto-fill. Snap waypointa w {self._snap_radius:.0f}mm."
            )
        else:
            self.trim_mode = "manual"
            print("[viewer] TRIM: MANUAL — geodesic prosto między waypointami.")
        self._refresh_overlay()

    def _toggle_curvature_view(self) -> None:
        """Heatmapa krzywizny (ridge score) na meshu — podgląd gdzie snap przyciąga."""
        self._curvature_view = not self._curvature_view
        if self._curvature_view:
            ridge = self._ensure_ridge()
            vr = vertex_ridge_scores(self.mesh, ridge)
            self.pv_mesh.point_data["ridge"] = vr.astype(np.float32)
            self._rebuild_mesh_actor(scalars="ridge")
            print("[viewer] Heatmapa krzywizny ON (żółte = gingival margin / valley).")
        else:
            self._rebuild_mesh_actor(scalars="selected")
            print("[viewer] Heatmapa krzywizny OFF.")
        self.plotter.render()

    def _rebuild_mesh_actor(self, scalars: str) -> None:
        """Przebuduj actor mesha z innym scalarem (selected ↔ ridge)."""
        if self._mesh_actor is not None:
            self.plotter.remove_actor(self._mesh_actor, render=False)
        if scalars == "ridge":
            # Heatmapa: tło neutralne (białawe) → granica (margin) dobija do
            # czerwonego. clim od 0 do wysokiego percentyla nonzero, żeby margin
            # realnie saturował na czerwono (a nie tylko pojedyncza najsilniejsza
            # krawędź). "Reds" startuje ~#fff5f0 (prawie biały) → ciemnoczerwony.
            ridge_vals = np.asarray(self.pv_mesh.point_data["ridge"])
            nz = ridge_vals[ridge_vals > 0]
            hi = float(np.percentile(nz, 90)) if nz.size else 1.0
            self._mesh_actor = self.plotter.add_mesh(
                self.pv_mesh,
                scalars="ridge",
                cmap="Reds",
                clim=(0.0, max(hi, 1e-3)),
                show_scalar_bar=False,
                smooth_shading=True,
                name="mesh_actor",
                reset_camera=False,
            )
        else:
            self._mesh_actor = self.plotter.add_mesh(
                self.pv_mesh,
                scalars="selected",
                cmap=SCALAR_CMAP,
                clim=(0, 1),
                show_scalar_bar=False,
                smooth_shading=True,
                interpolate_before_map=True,
                specular=0.25,
                specular_power=12,
                name="mesh_actor",
                reset_camera=False,
            )

    def _invert_fill(self) -> None:
        if not self.is_closed:
            return
        other = fill_interior(self.graph, self.boundary_mask, prefer="larger")
        self.fill_mask = other
        print(f"[viewer] Odwrócono: {other.sum()} verts (larger touching).")
        self._refresh_all()

    def _undo(self) -> None:
        if self.is_closed:
            last_seg = self.path_segments.pop()
            self.boundary_mask[:] = False
            for seg in self.path_segments:
                self.boundary_mask[seg] = True
            for wp in self.waypoints:
                self.boundary_mask[wp] = True
            self.is_closed = False
            self.fill_mask[:] = False
            print("[viewer] Otwarto pętlę.")
        elif self.waypoints:
            self.waypoints.pop()
            if self.path_segments:
                self.path_segments.pop()
            self.boundary_mask[:] = False
            for seg in self.path_segments:
                self.boundary_mask[seg] = True
            for wp in self.waypoints:
                self.boundary_mask[wp] = True
            print(f"[viewer] Cofnięto. Zostało {len(self.waypoints)} waypointów.")
        self._refresh_all()

    def _clear(self) -> None:
        self.waypoints.clear()
        self.path_segments.clear()
        self.boundary_mask[:] = False
        self.fill_mask[:] = False
        self.is_closed = False
        print("[viewer] Wyczyszczono.")
        self._refresh_all()

    def _save(self) -> None:
        mask = self._combined_mask()
        if not mask.any():
            print("[viewer] Selekcja pusta — nic do zapisu.")
            return
        out = save_selection(self.loaded, mask)
        print(f"[viewer] Zapisano [debug]: {out}  ({mask.sum()} verts)")

    def _reload(self) -> None:
        m = load_selection(self.loaded)
        if m is None:
            print("[viewer] Brak zapisanej selekcji.")
            return
        self._clear()
        self.fill_mask = m
        print(f"[viewer] Wczytano {m.sum()} verts jako fill (bez konturu).")
        self._refresh_all()

    # ---------- aligner: generate / export / hide ----------
    def _generate_aligner(self) -> None:
        sel = self._combined_mask()
        n_sel = int(sel.sum())
        if not sel.any():
            print("[viewer] Najpierw zaznacz powierzchnię (waypointy + F lub G).")
            return
        if n_sel < 5000:
            print(
                f"[viewer] UWAGA: selekcja tylko {n_sel} verts — to bardzo mało, "
                f"nakładka będzie malutka/niewidoczna. Czy na pewno? "
                f"(L=wczytaj zapisaną pełną selekcję, F=wypełnij narysowaną pętlę)"
            )
        print(f"[viewer] Generuję nakładkę ({n_sel} verts selekcji)...")
        mesh_out, rep = generate_aligner(self.mesh, sel, self.params)
        if mesh_out is None:
            print(f"[viewer] Generacja nie powiodła się: {rep.notes}")
            return
        self.aligner_mesh = mesh_out
        self._show_aligner()
        self._refresh_overlay()

    def _show_aligner(self) -> None:
        if self.aligner_mesh is None:
            return
        if self._aligner_actor is not None:
            self.plotter.remove_actor(self._aligner_actor, render=False)
            self._aligner_actor = None

        v = np.asarray(self.aligner_mesh.vertices, dtype=float)
        f = np.asarray(self.aligner_mesh.faces, dtype=np.int64)
        faces_flat = np.hstack(
            [np.full((len(f), 1), 3, dtype=np.int64), f]
        ).ravel()
        pv_aligner = pv.PolyData(v, faces_flat)
        self._aligner_actor = self.plotter.add_mesh(
            pv_aligner,
            color=ALIGNER_COLOR,
            opacity=self._aligner_opacity,
            smooth_shading=True,
            specular=0.4,
            specular_power=15,
            name="aligner_actor",
            pickable=False,
            reset_camera=False,
        )

    def _toggle_aligner_visibility(self) -> None:
        if self._aligner_actor is None:
            print("[viewer] Brak nakładki — najpierw N (generate).")
            return
        vis = self._aligner_actor.GetVisibility()
        self._aligner_actor.SetVisibility(not vis)
        print(f"[viewer] Nakładka {'widoczna' if not vis else 'ukryta'}.")
        self.plotter.render()

    def _toggle_mesh_visibility(self) -> None:
        """Chowa/pokazuje główny mesh zębów (do inspekcji wnętrza nakładki)."""
        if self._mesh_actor is None:
            return
        vis = self._mesh_actor.GetVisibility()
        self._mesh_actor.SetVisibility(not vis)
        print(f"[viewer] Mesh zębów {'widoczny' if not vis else 'ukryty'}.")
        self.plotter.render()

    def _toggle_quality_mode(self) -> None:
        """Toggle PREVIEW (pitch 0.20, ~12s) ↔ FINAL (pitch 0.10, ~90s).

        PREVIEW do szybkiej iteracji parametrów, FINAL przed exportem STL
        do druku. Direct-print Formlabs/Asiga wymaga FINAL dla pełnej
        precyzji ~50µm.
        """
        if self._quality_mode == "final":
            self._quality_mode = "preview"
            self.params.voxel_pitch = 0.20
            print("[viewer] Quality: PREVIEW (pitch=0.20mm, ~12s/N)")
        else:
            self._quality_mode = "final"
            self.params.voxel_pitch = 0.10
            print("[viewer] Quality: FINAL (pitch=0.10mm, ~90s/N) — do druku 3D")
        self._refresh_overlay()

    def _export_aligner(self) -> None:
        if self.aligner_mesh is None:
            print("[viewer] Brak nakładki do eksportu — najpierw N (generate).")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUTPUT_DIR / f"{self.loaded.stl_path.stem}_aligner.stl"
        self.aligner_mesh.export(out)
        print(
            f"[viewer] Wyeksportowano nakładkę: {out}  "
            f"({len(self.aligner_mesh.vertices)} verts, "
            f"watertight={self.aligner_mesh.is_watertight})"
        )

    # ---------- rendering ----------
    def _refresh_all(self) -> None:
        self._refresh_scalars()
        self._refresh_contour()
        self._refresh_waypoints()
        self._refresh_overlay()

    def _refresh_scalars(self) -> None:
        new_vals = self._combined_mask().astype(np.uint8)
        if "selected" in self.pv_mesh.point_data:
            arr = self.pv_mesh.point_data["selected"]
            if arr.shape == new_vals.shape:
                arr[:] = new_vals
                # poinformuj VTK że scalar array się zmienił
                vtk_arr = self.pv_mesh.GetPointData().GetArray("selected")
                if vtk_arr is not None:
                    vtk_arr.Modified()
            else:
                self.pv_mesh.point_data["selected"] = new_vals
        else:
            self.pv_mesh.point_data["selected"] = new_vals
        self.plotter.render()

    def _refresh_contour(self) -> None:
        if self._contour_actor is not None:
            self.plotter.remove_actor(self._contour_actor, render=False)
            self._contour_actor = None
        if not self.path_segments:
            return
        all_pts = []
        line_blocks = []
        offset = 0
        for seg in self.path_segments:
            pts = np.asarray(self.mesh.vertices[seg], dtype=float)
            n = len(pts)
            all_pts.append(pts)
            idx = np.arange(offset, offset + n)
            line = np.empty(2 * (n - 1) + (n - 1), dtype=np.int64)
            # pv format: [count, i0, i1, count, i1, i2, ...]
            line = np.column_stack(
                [np.full(n - 1, 2), idx[:-1], idx[1:]]
            ).ravel()
            line_blocks.append(line)
            offset += n
        verts = np.vstack(all_pts)
        lines = np.concatenate(line_blocks)
        poly = pv.PolyData(verts, lines=lines)
        # lift slightly above surface so kontur nie z-fighting
        normals_per_vertex = self._approx_vertex_normals(verts)
        verts_lifted = verts + 0.0015 * self._bbox_diag * normals_per_vertex
        poly.points = verts_lifted
        self._contour_actor = self.plotter.add_mesh(
            poly,
            color=CONTOUR_COLOR,
            line_width=4,
            render_lines_as_tubes=False,
            pickable=False,
            name="contour_actor",
            reset_camera=False,
        )

    def _approx_vertex_normals(self, points: np.ndarray) -> np.ndarray:
        """Wyciągnij normalne mesha dla zadanych punktów (po najbliższym vertexie)."""
        if not hasattr(self.mesh, "vertex_normals"):
            return np.zeros_like(points)
        # nearest mesh vertex dla każdego z 'points'
        from scipy.spatial import cKDTree

        if not hasattr(self, "_kdtree"):
            self._kdtree = cKDTree(self.mesh.vertices)
        _, idx = self._kdtree.query(points, k=1)
        return np.asarray(self.mesh.vertex_normals[idx], dtype=float)

    def _refresh_waypoints(self) -> None:
        for actor_attr in ("_waypoint_actor", "_first_waypoint_actor"):
            actor = getattr(self, actor_attr)
            if actor is not None:
                self.plotter.remove_actor(actor, render=False)
                setattr(self, actor_attr, None)
        if not self.waypoints:
            return
        wp_positions = np.asarray(self.mesh.vertices[self.waypoints], dtype=float)
        # lift waypointy ponad powierzchnię żeby były widoczne
        lift = 0.002 * self._bbox_diag
        if hasattr(self.mesh, "vertex_normals"):
            wp_positions = wp_positions + lift * np.asarray(
                self.mesh.vertex_normals[self.waypoints], dtype=float
            )

        # pierwszy waypoint — większy, zielony
        first_pd = pv.PolyData(wp_positions[:1])
        first_glyph = first_pd.glyph(
            geom=pv.Sphere(radius=self._waypoint_radius * 1.6),
            scale=False,
            orient=False,
        )
        self._first_waypoint_actor = self.plotter.add_mesh(
            first_glyph,
            color=FIRST_WAYPOINT_COLOR,
            pickable=False,
            name="first_waypoint_actor",
            reset_camera=False,
        )

        if len(wp_positions) > 1:
            rest_pd = pv.PolyData(wp_positions[1:])
            rest_glyph = rest_pd.glyph(
                geom=pv.Sphere(radius=self._waypoint_radius),
                scale=False,
                orient=False,
            )
            self._waypoint_actor = self.plotter.add_mesh(
                rest_glyph,
                color=WAYPOINT_COLOR,
                pickable=False,
                name="waypoint_actor",
                reset_camera=False,
            )

    def _refresh_overlay(self) -> None:
        n_wp = len(self.waypoints)
        n_sel = int(self._combined_mask().sum())
        n_tot = len(self.mesh.vertices)
        status = "CLOSED" if self.is_closed else f"OPEN ({n_wp} pkt)"
        status += f"   |   TRIM: {self.trim_mode.upper()}"
        if self._awaiting_seed:
            status += "   >>> KLIKNIJ ZIARNO WEWNĄTRZ PĘTLI <<<"
        aligner_info = ""
        if self.aligner_mesh is not None:
            aligner_info = (
                f"\nNakładka: {len(self.aligner_mesh.vertices)} verts, "
                f"watertight={self.aligner_mesh.is_watertight}"
            )
        params_info = (
            f"\nParams: offset={self.params.inner_clearance:.3f}mm  "
            f"grubosc={self.params.thickness:.2f}mm  "
            f"fillet={self.params.fillet_radius:.2f}mm  "
            f"trim_smooth={self.params.trim_smooth_sigma:.1f}\n"
            f"        quality={self._quality_mode.upper()} (pitch={self.params.voxel_pitch:.2f})"
        )
        text = (
            f"Plik: {self.loaded.stl_path.name}\n"
            f"Stan: {status}\n"
            f"Zaznaczone: {n_sel} / {n_tot}  ({100.0 * n_sel / max(n_tot,1):.1f}%)"
            f"{aligner_info}"
            f"{params_info}\n"
            f"\n"
            f"Klik = waypoint  |  klik blisko 1.zielonego = zamknij\n"
            f"[C] close  [F] fill (auto)  [G] seed-fill  [I] invert\n"
            f"[W] trim manual/scalloped  [V] heatmapa  [ ] prog ridge  ; ' rozmiar\n"
            f"[Z] undo  [X] clear  [S] save  [L] load\n"
            f"[N] generate  [E] export STL  [P] preview/final  [H] aligner  [M] teeth"
        )
        color = "#fde047" if self._awaiting_seed else "#e5e5e5"
        self.plotter.add_text(
            text,
            position="upper_left",
            font_size=10,
            color=color,
            name="status_text",
        )

    # ---------- entry ----------
    def run(self) -> None:
        print(
            "[viewer] Klikaj waypointy → C (zamknij) → F (fill) → "
            "N (generate aligner) → E (export STL). Q/Esc kończy."
        )
        self.plotter.show()
