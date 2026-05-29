"""PySide6 + pyvistaqt desktop app — **Faza 2: kontrolki + funkcje**.

Faza 1 (zrobione): scaffold QMainWindow, LMB pick / RMB rotate.
Faza 2 (TEN PLIK): dock panel z natywnymi QSlider+SpinBox dla wszystkich parametrów,
                   toolbar z akcjami, scalloped + korytarz, fill, generate, export.
Faza 3 (TODO): QThread dla generate (90s — żeby okno nie zamarzało), undo/redo stack,
               recent files, presets.

Logika algorytmów (selection / sdf / curvature / aligner) — bez zmian, reuse z modułów.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from pyvistaqt import QtInteractor
except ImportError as e:  # pragma: no cover
    print(
        f"[qt_viewer] Brak Qt deps: {e}\n"
        f"Zainstaluj: .venv/bin/pip install -e ."
    )
    raise

import trimesh

from .aligner import AlignerParams, generate_aligner
from .curvature import (
    build_snap_graph,
    compute_edge_ridge_scores,
    denoise_ridge,
    vertex_ridge_scores,
)
from .io import LoadedMesh, load_selection, load_stl, save_selection
from .selection import (
    build_edge_graph,
    fill_from_seed,
    fill_interior,
    nearest_vertex,
    shortest_path,
)


# ---------- kolory ----------
SCALAR_CMAP = ["#cfcfcf", "#dc2f5a"]
WAYPOINT_COLOR = "#fbbf24"
FIRST_WAYPOINT_COLOR = "#22c55e"
CONTOUR_COLOR = "#22d3ee"
ALIGNER_COLOR = "#06b6d4"

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "output"


# ===========================================================================
# Helper widget — slider + spinbox sharing value (slider dla feel, spinbox dla precyzji)
# ===========================================================================
class LabeledSlider(QtWidgets.QWidget):
    """Suwak + spinbox + label dla parametru float.

    Sygnały:
      valueChanged       — emituje na KAŻDĄ zmianę (live; do tanich callbacków),
      valueChangedFinal  — emituje TYLKO na sliderReleased / editingFinished
                           (do drogich callbacków typu recompute krzywizny).
    """

    valueChanged = QtCore.Signal(float)
    valueChangedFinal = QtCore.Signal(float)

    def __init__(
        self,
        label: str,
        vmin: float,
        vmax: float,
        default: float,
        step: float,
        decimals: int = 2,
        suffix: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._mult = 10 ** decimals

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(6)

        self.label = QtWidgets.QLabel(label)
        self.label.setMinimumWidth(150)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(int(vmin * self._mult), int(vmax * self._mult))
        self.slider.setValue(int(default * self._mult))

        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        self.spin.setValue(default)
        self.spin.setSuffix(suffix)
        self.spin.setMinimumWidth(90)
        self.spin.setAlignment(QtCore.Qt.AlignRight)

        lay.addWidget(self.label)
        lay.addWidget(self.slider, stretch=1)
        lay.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider)
        self.slider.sliderReleased.connect(
            lambda: self.valueChangedFinal.emit(self.value())
        )
        self.spin.valueChanged.connect(self._on_spin)
        self.spin.editingFinished.connect(
            lambda: self.valueChangedFinal.emit(self.value())
        )

    def _on_slider(self, v):
        f = v / self._mult
        self.spin.blockSignals(True)
        self.spin.setValue(f)
        self.spin.blockSignals(False)
        self.valueChanged.emit(f)

    def _on_spin(self, f):
        self.slider.blockSignals(True)
        self.slider.setValue(int(f * self._mult))
        self.slider.blockSignals(False)
        self.valueChanged.emit(f)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, v: float):
        self.spin.setValue(v)


# ===========================================================================
# Custom interactor style: LMB picking, RMB rotate
# ===========================================================================
class PickRotateStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, on_left_click):
        super().__init__()
        self._on_left_click = on_left_click
        self.AddObserver("LeftButtonPressEvent", self._lmb_press, 10.0)
        self.AddObserver("LeftButtonReleaseEvent", self._consume, 10.0)
        self.AddObserver("RightButtonPressEvent", self._rmb_press, 10.0)
        self.AddObserver("RightButtonReleaseEvent", self._rmb_release, 10.0)

    @staticmethod
    def _abort(obj):
        try:
            obj.AbortFlagOn()
        except Exception:
            pass

    def _lmb_press(self, obj, event):
        iren = self.GetInteractor()
        if iren is not None:
            x, y = iren.GetEventPosition()
            try:
                self._on_left_click(x, y)
            except Exception as e:  # pragma: no cover
                print(f"[qt_viewer] pick callback error: {e}")
        self._abort(obj)

    def _consume(self, obj, event):
        self._abort(obj)

    def _rmb_press(self, obj, event):
        self.StartRotate()
        self._abort(obj)

    def _rmb_release(self, obj, event):
        self.EndRotate()
        self._abort(obj)


# ===========================================================================
# Główne okno
# ===========================================================================
class AlignerMainWindow(QtWidgets.QMainWindow):
    def __init__(self, loaded: LoadedMesh):
        super().__init__()
        self.loaded = loaded
        self.mesh = loaded.mesh

        self.setWindowTitle(f"Aligner Generator — {loaded.stl_path.name}")
        self.resize(1600, 1000)

        # ----- state: selekcja / waypointy -----
        self.waypoints: list[int] = []
        self.path_segments: list[np.ndarray] = []
        self.boundary_mask = np.zeros(len(self.mesh.vertices), dtype=bool)
        self.fill_mask = np.zeros(len(self.mesh.vertices), dtype=bool)
        self.is_closed = False
        self._awaiting_seed = False

        # ----- state: live-wire / krzywizna -----
        self.trim_mode = "manual"      # "manual" | "scalloped"
        self._snap_graph = None
        self._edge_ridge = None
        self._curvature_view = False
        self._angle_threshold = 0.14   # rad (~8°)
        self._min_comp_edges = 25
        self._snap_strength = 25.0
        self._ridge_kdtree = None
        self._ridge_vert_idx = None
        self._snap_radius = 3.0        # mm
        self._corridor_tol = 4.0       # mm

        # ----- state: aligner -----
        self.aligner_mesh: trimesh.Trimesh | None = None
        self._aligner_actor = None
        self._aligner_opacity = 0.85
        self.params = AlignerParams()
        self._quality_mode = "final"
        self.params.voxel_pitch = 0.10
        # ----- state: rim + apex + pillars (S1/S2/S3 support generation) -----
        from .supports import PillarParams
        self.rim_mask: np.ndarray | None = None
        self._rim_actor = None
        self._apex_loop: np.ndarray | None = None   # cache apex polyline
        self.anchors: np.ndarray | None = None      # spróbkowane kotwice (N, 3)
        self._anchors_actor = None
        self._anchor_spacing = 3.0                  # mm między kotwicami
        # S3: geometria pillarów
        self.pillar_params = PillarParams()
        self._pillar_direction: np.ndarray | None = None
        self.pillars_mesh = None                    # trimesh.Trimesh | None
        self._pillars_actor = None
        # S5: orientacja druku (nachylenie 45°, front/back, pillary do raftu)
        self._tilt_angle = 45.0                     # ° (-45..+45; znak = która strona nisko)
        self._print_view = False                    # czy widok print-ready (tilted)
        self._ap_frame = None                       # cache detekcji przód/tył
        self._print_transform = np.eye(4)
        self._display_matrix = np.eye(4)            # user_matrix aktorów scan-space
        self._ground_actor = None                   # płaszczyzna Z=0 (platforma)

        # ----- geometria -----
        bbox = self.mesh.bounds
        self._bbox_diag = float(np.linalg.norm(bbox[1] - bbox[0]))
        self._close_thresh = 0.02 * self._bbox_diag

        print("[qt_viewer] Buduję graf krawędzi (Dijkstra)...")
        self.graph = build_edge_graph(self.mesh)

        m = load_selection(loaded)
        if m is not None:
            self.fill_mask = m
            print(f"[qt_viewer] Auto-load selekcji: {m.sum()} verts")

        # ----- centralny viewport -----
        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1a1a1a")
        self.plotter.add_axes()
        self.setCentralWidget(self.plotter)

        faces_flat = np.hstack(
            [np.full((len(self.mesh.faces), 1), 3, dtype=np.int64), self.mesh.faces]
        ).ravel()
        self.pv_mesh = pv.PolyData(np.asarray(self.mesh.vertices), faces_flat)
        self.pv_mesh.point_data["selected"] = self._combined_mask().astype(np.uint8)
        self._mesh_actor = None
        self._waypoint_actor = None
        self._first_waypoint_actor = None
        self._contour_actor = None
        self._add_main_mesh(scalars="selected")

        # ----- interactor style -----
        self._style = PickRotateStyle(self._on_pick)
        self._install_style()

        # ----- UI: dock, toolbar, menu, statusbar -----
        self._build_toolbar()
        self._build_param_dock()
        self._build_menu()
        self.status = self.statusBar()
        self._update_status()

        # ----- skróty klawiszowe (subset; reszta przez toolbar/menu) -----
        QtGui.QShortcut(QtGui.QKeySequence("X"), self, activated=self._clear)
        QtGui.QShortcut(QtGui.QKeySequence("Z"), self, activated=self._undo)
        QtGui.QShortcut(QtGui.QKeySequence("C"), self, activated=self._close_loop)
        QtGui.QShortcut(QtGui.QKeySequence("F"), self, activated=self._fill)
        QtGui.QShortcut(QtGui.QKeySequence("G"), self, activated=self._arm_seed_fill)
        QtGui.QShortcut(QtGui.QKeySequence("I"), self, activated=self._invert_fill)

        self.plotter.reset_camera()
        self._refresh()

    # =====================================================================
    # interactor style install
    # =====================================================================
    def _install_style(self):
        for accessor in ("iren.interactor", "iren", "interactor"):
            try:
                obj = self.plotter
                for part in accessor.split("."):
                    obj = getattr(obj, part)
                obj.SetInteractorStyle(self._style)
                return
            except Exception:
                continue
        print("[qt_viewer] UWAGA: nie udało się wpiąć custom interactor style.")

    # =====================================================================
    # UI building
    # =====================================================================
    def _build_toolbar(self):
        tb = self.addToolBar("Główne")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(20, 20))

        def add_action(text, slot, shortcut=None, tooltip=None, checkable=False):
            a = QtGui.QAction(text, self)
            if shortcut:
                a.setShortcut(shortcut)
            if tooltip:
                a.setToolTip(tooltip)
            a.setCheckable(checkable)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        add_action("Wczytaj", self._load_stl_dialog, tooltip="Wczytaj STL")
        add_action("Zapisz sel", self._save, "Ctrl+S", "Zapisz selekcję")
        add_action("Wczytaj sel", self._reload, "Ctrl+L", "Wczytaj zapisaną selekcję")
        tb.addSeparator()

        self._act_trim = add_action(
            "Trim: Manual", self._toggle_trim_mode, "W",
            "Przełącz manual ↔ scalloped (auto-trace)", checkable=True,
        )
        self._act_heatmap = add_action(
            "Heatmapa", self._toggle_curvature_view, "V",
            "Wł/Wył podgląd krzywizny", checkable=True,
        )
        tb.addSeparator()

        add_action("Wypełnij", self._fill, "F", "Wypełnij wnętrze pętli (auto smaller)")
        add_action("Seed-fill", self._arm_seed_fill, "G", "Następny klik = ziarno fillu")
        add_action("Odwróć", self._invert_fill, "I", "Odwróć fill (większa/mniejsza)")
        tb.addSeparator()

        add_action("Generuj", self._generate_aligner, "N", "Generuj nakładkę z selekcji")
        add_action("Eksport STL", self._export_aligner, "E", "Zapisz nakładkę jako STL")
        tb.addSeparator()

        self._act_show_aligner = add_action(
            "Nakładka", self._toggle_aligner_visibility, "H",
            "Pokaż/ukryj nakładkę", checkable=True,
        )
        self._act_show_aligner.setChecked(True)
        self._act_show_mesh = add_action(
            "Zęby", self._toggle_mesh_visibility, "M",
            "Pokaż/ukryj zęby", checkable=True,
        )
        self._act_show_mesh.setChecked(True)
        self._act_show_rim = add_action(
            "Rim (krawędź)", self._toggle_rim_visibility, "B",
            "Podświetl wykrytą krawędź trim nakładki (dla podpór druku)",
            checkable=True,
        )
        self._act_show_rim.setChecked(True)
        self._act_show_anchors = add_action(
            "Kotwice", self._toggle_anchors_visibility, "K",
            "Pokaż próbki na apex line — przyszłe punkty kotwiczenia podpór",
            checkable=True,
        )
        self._act_show_anchors.setChecked(True)
        self._act_show_pillars = add_action(
            "Podpory", self._toggle_pillars_visibility, "P",
            "Pokaż pillary druku (stożek tip + cylinder body, per kotwica)",
            checkable=True,
        )
        self._act_show_pillars.setChecked(True)
        tb.addSeparator()
        self._act_print_view = add_action(
            "Widok druku", self._toggle_print_view, "D",
            "Nachyl nakładkę do orientacji druku (pillary pionowo do raftu Z=0)",
            checkable=True,
        )
        self._act_print_view.setChecked(False)

    def _build_param_dock(self):
        dock = QtWidgets.QDockWidget("Parametry", self)
        dock.setAllowedAreas(QtCore.Qt.RightDockWidgetArea | QtCore.Qt.LeftDockWidgetArea)
        dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetFloatable
        )

        inner = QtWidgets.QWidget()
        outer_lay = QtWidgets.QVBoxLayout(inner)
        outer_lay.setContentsMargins(8, 8, 8, 8)

        # ----- Grupa: Detekcja krzywizny -----
        g1 = QtWidgets.QGroupBox("Detekcja krzywizny (live-wire)")
        f1 = QtWidgets.QVBoxLayout(g1)
        self.sl_angle = LabeledSlider(
            "Próg kąta", 1.0, 40.0, np.degrees(self._angle_threshold), 0.5,
            decimals=1, suffix=" °",
        )
        self.sl_min_comp = LabeledSlider(
            "Min komponent", 1, 500, self._min_comp_edges, 5, decimals=0, suffix=" e",
        )
        self.sl_snap_radius = LabeledSlider(
            "Snap radius", 1.0, 10.0, self._snap_radius, 0.5, decimals=1, suffix=" mm",
        )
        self.sl_snap_strength = LabeledSlider(
            "Snap strength", 1.0, 100.0, self._snap_strength, 5.0, decimals=0,
        )
        for w in (self.sl_angle, self.sl_min_comp, self.sl_snap_radius, self.sl_snap_strength):
            f1.addWidget(w)
        # ridge params są drogie (recompute krzywizny) — używamy valueChangedFinal
        self.sl_angle.valueChangedFinal.connect(self._on_angle_change)
        self.sl_min_comp.valueChangedFinal.connect(self._on_min_comp_change)
        self.sl_snap_radius.valueChangedFinal.connect(self._on_snap_radius_change)
        self.sl_snap_strength.valueChangedFinal.connect(self._on_snap_strength_change)
        outer_lay.addWidget(g1)

        # ----- Grupa: Ścieżka konturu -----
        g2 = QtWidgets.QGroupBox("Ścieżka konturu")
        f2 = QtWidgets.QVBoxLayout(g2)
        self.sl_corridor = LabeledSlider(
            "Korytarz tolerancji", 1.0, 15.0, self._corridor_tol, 0.5,
            decimals=1, suffix=" mm",
        )
        # korytarz tani — value zapisuje się przy każdej zmianie (używamy przy kolejnym klik)
        self.sl_corridor.valueChanged.connect(self._on_corridor_change)
        f2.addWidget(self.sl_corridor)
        outer_lay.addWidget(g2)

        # ----- Grupa: Generacja nakładki -----
        g3 = QtWidgets.QGroupBox("Generacja nakładki")
        f3 = QtWidgets.QVBoxLayout(g3)
        self.sl_offset = LabeledSlider(
            "Offset od zębów", 0.05, 0.30, self.params.inner_clearance, 0.01,
            decimals=3, suffix=" mm",
        )
        self.sl_thickness = LabeledSlider(
            "Grubość ścianki", 0.50, 1.50, self.params.thickness, 0.05,
            decimals=2, suffix=" mm",
        )
        self.sl_fillet = LabeledSlider(
            "Zaokrąglenie krawędzi", 0.0, 1.50, self.params.fillet_radius, 0.05,
            decimals=2, suffix=" mm",
        )
        self.sl_trim_smooth = LabeledSlider(
            "Gładkość krawędzi (trim)", 0.0, 12.0, self.params.trim_smooth_sigma, 0.5,
            decimals=1, suffix=" vx",
        )
        for w in (self.sl_offset, self.sl_thickness, self.sl_fillet, self.sl_trim_smooth):
            f3.addWidget(w)
        self.sl_offset.valueChanged.connect(self._on_offset_change)
        self.sl_thickness.valueChanged.connect(self._on_thickness_change)
        self.sl_fillet.valueChanged.connect(self._on_fillet_change)
        self.sl_trim_smooth.valueChanged.connect(self._on_trim_smooth_change)
        outer_lay.addWidget(g3)

        # ----- Grupa: Jakość -----
        g4 = QtWidgets.QGroupBox("Jakość (voxel pitch)")
        f4 = QtWidgets.QHBoxLayout(g4)
        self.rb_preview = QtWidgets.QRadioButton("PREVIEW (~12s)")
        self.rb_final = QtWidgets.QRadioButton("FINAL — druk (~90s)")
        if self._quality_mode == "final":
            self.rb_final.setChecked(True)
        else:
            self.rb_preview.setChecked(True)
        self.rb_preview.toggled.connect(self._on_quality_change)
        f4.addWidget(self.rb_preview)
        f4.addWidget(self.rb_final)
        outer_lay.addWidget(g4)

        # ----- Przycisk Generate (duży, na dole panelu) -----
        self.btn_generate = QtWidgets.QPushButton("⚙  Generuj nakładkę")
        self.btn_generate.setMinimumHeight(40)
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #06b6d4; color: white; font-weight: bold; }"
        )
        self.btn_generate.clicked.connect(self._generate_aligner)
        outer_lay.addWidget(self.btn_generate)

        # ----- Grupa: Podpory druku — kotwice (S2) -----
        g5 = QtWidgets.QGroupBox("Kotwice podpór (S2)")
        f5 = QtWidgets.QVBoxLayout(g5)
        self.sl_anchor_spacing = LabeledSlider(
            "Rozstaw kotwic", 1.0, 6.0, self._anchor_spacing, 0.5,
            decimals=1, suffix=" mm",
        )
        self.sl_anchor_spacing.valueChanged.connect(self._on_anchor_spacing_change)
        f5.addWidget(self.sl_anchor_spacing)
        outer_lay.addWidget(g5)

        # ----- Grupa: Geometria pillarów (S3) -----
        g6 = QtWidgets.QGroupBox("Geometria pillarów (S3)")
        f6 = QtWidgets.QVBoxLayout(g6)
        self.sl_tip_dia = LabeledSlider(
            "Tip Ø (kontakt)", 0.2, 0.8, self.pillar_params.tip_diameter, 0.05,
            decimals=2, suffix=" mm",
        )
        self.sl_body_dia = LabeledSlider(
            "Body Ø (trzon)", 0.5, 1.5, self.pillar_params.body_diameter, 0.05,
            decimals=2, suffix=" mm",
        )
        self.sl_pillar_h = LabeledSlider(
            "Wysokość pillara", 3.0, 15.0, self.pillar_params.pillar_height, 0.5,
            decimals=1, suffix=" mm",
        )
        for w in (self.sl_tip_dia, self.sl_body_dia, self.sl_pillar_h):
            f6.addWidget(w)
        self.sl_tip_dia.valueChanged.connect(self._on_tip_dia_change)
        self.sl_body_dia.valueChanged.connect(self._on_body_dia_change)
        self.sl_pillar_h.valueChanged.connect(self._on_pillar_h_change)
        outer_lay.addWidget(g6)

        # ----- Grupa: Orientacja druku (S5) -----
        g7 = QtWidgets.QGroupBox("Orientacja druku (S5)")
        f7 = QtWidgets.QVBoxLayout(g7)
        self.sl_tilt = LabeledSlider(
            "Nachylenie", -45.0, 45.0, self._tilt_angle, 1.0,
            decimals=0, suffix=" °",
        )
        self.sl_tilt.valueChanged.connect(self._on_tilt_change)
        f7.addWidget(self.sl_tilt)
        hint = QtWidgets.QLabel(
            "+ = przód (siekacze) do góry · − = przód nisko.\n"
            "Włącz „Widok druku\" [D] żeby zobaczyć."
        )
        hint.setStyleSheet("color: #999; font-size: 10px;")
        hint.setWordWrap(True)
        f7.addWidget(hint)
        outer_lay.addWidget(g7)

        outer_lay.addStretch(1)

        # scroll wrapper — gdy okno wąskie
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        dock.setWidget(scroll)
        dock.setMinimumWidth(380)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

    def _build_menu(self):
        mb = self.menuBar()

        m_file = mb.addMenu("&Plik")
        a_open = QtGui.QAction("Wczytaj STL…", self)
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self._load_stl_dialog)
        m_file.addAction(a_open)
        m_file.addSeparator()
        a_quit = QtGui.QAction("Wyjście", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_view = mb.addMenu("&Widok")
        a_reset_cam = QtGui.QAction("Reset kamery", self)
        a_reset_cam.setShortcut("R")
        a_reset_cam.triggered.connect(lambda: (self.plotter.reset_camera(), self.plotter.render()))
        m_view.addAction(a_reset_cam)

    # =====================================================================
    # maski / refresh
    # =====================================================================
    def _combined_mask(self) -> np.ndarray:
        return self.boundary_mask | self.fill_mask

    def _set_matrix(self, actor):
        """Ustaw user_matrix aktora = display matrix (nachylenie print view)."""
        if actor is None:
            return
        try:
            actor.user_matrix = self._display_matrix
        except Exception:
            pass

    def _add_main_mesh(self, scalars: str):
        if self._mesh_actor is not None:
            self.plotter.remove_actor(self._mesh_actor, render=False)
            self._mesh_actor = None
        if scalars == "ridge":
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
                specular=0.25,
                specular_power=12,
                name="mesh_actor",
                reset_camera=False,
            )
        self._set_matrix(self._mesh_actor)

    def _update_status(self):
        n_sel = int(self._combined_mask().sum())
        n_tot = len(self.mesh.vertices)
        pct = 100.0 * n_sel / max(n_tot, 1)
        state = "CLOSED" if self.is_closed else f"OPEN ({len(self.waypoints)} pkt)"
        seed = " | KLIKNIJ ZIARNO" if self._awaiting_seed else ""
        aligner = ""
        if self.aligner_mesh is not None:
            aligner = (
                f" | nakładka {len(self.aligner_mesh.vertices)}v"
                f" watertight={self.aligner_mesh.is_watertight}"
            )
        self.status.showMessage(
            f"  {state}  |  trim {self.trim_mode.upper()}  "
            f"|  selekcja {n_sel}/{n_tot} ({pct:.1f}%){aligner}{seed}"
            f"    LMB waypoint · RMB obrót · MMB pan · wheel zoom"
        )

    def _refresh(self):
        # scalar update
        if not self._curvature_view:
            self.pv_mesh.point_data["selected"] = self._combined_mask().astype(np.uint8)
        # przebuduj overlay actorów (waypointy + linia)
        for attr in ("_waypoint_actor", "_first_waypoint_actor", "_contour_actor"):
            actor = getattr(self, attr, None)
            if actor is not None:
                self.plotter.remove_actor(actor, render=False)
                setattr(self, attr, None)

        if self.waypoints:
            pts = self.mesh.vertices[self.waypoints]
            self._first_waypoint_actor = self.plotter.add_mesh(
                pv.PolyData(pts[:1]),
                color=FIRST_WAYPOINT_COLOR,
                point_size=14,
                render_points_as_spheres=True,
                name="first_wp",
                reset_camera=False,
            )
            if len(pts) > 1:
                self._waypoint_actor = self.plotter.add_mesh(
                    pv.PolyData(pts[1:]),
                    color=WAYPOINT_COLOR,
                    point_size=11,
                    render_points_as_spheres=True,
                    name="rest_wp",
                    reset_camera=False,
                )

        if self.path_segments:
            verts_seq, lines, offset = [], [], 0
            for seg in self.path_segments:
                if seg.size < 2:
                    continue
                pts = self.mesh.vertices[seg]
                verts_seq.append(pts)
                lines.append(np.concatenate(([len(pts)], np.arange(len(pts)) + offset)))
                offset += len(pts)
            if verts_seq:
                poly = pv.PolyData(np.vstack(verts_seq))
                poly.lines = np.concatenate(lines)
                self._contour_actor = self.plotter.add_mesh(
                    poly,
                    color=CONTOUR_COLOR,
                    line_width=3,
                    name="contour",
                    reset_camera=False,
                )

        self._update_status()
        self.plotter.render()

    # =====================================================================
    # ridge / snap (live-wire)
    # =====================================================================
    def _ensure_ridge(self):
        if self._edge_ridge is None:
            print(
                f"[qt_viewer] Liczę krzywiznę: angle={self._angle_threshold:.3f}rad "
                f"({np.degrees(self._angle_threshold):.1f}°), "
                f"min_comp={self._min_comp_edges}, snap_strength={self._snap_strength}"
            )
            raw = compute_edge_ridge_scores(self.mesh)
            print(
                f"[qt_viewer] raw ridge: {np.count_nonzero(raw)} edges, max={raw.max():.3f}rad"
            )
            self._edge_ridge = denoise_ridge(
                self.mesh, raw,
                angle_threshold=self._angle_threshold,
                min_component_edges=self._min_comp_edges,
            )
        return self._edge_ridge

    def _ensure_snap_graph(self):
        if self._snap_graph is None:
            self._snap_graph = build_snap_graph(
                self.mesh, self._ensure_ridge(), strength=self._snap_strength
            )
        return self._snap_graph

    def _ensure_ridge_kdtree(self):
        if self._ridge_kdtree is None:
            from scipy.spatial import cKDTree

            vr = vertex_ridge_scores(self.mesh, self._ensure_ridge())
            self._ridge_vert_idx = np.where(vr > 0)[0]
            if self._ridge_vert_idx.size > 0:
                self._ridge_kdtree = cKDTree(self.mesh.vertices[self._ridge_vert_idx])
        return self._ridge_kdtree

    def _invalidate_ridge(self):
        self._edge_ridge = None
        self._snap_graph = None
        self._ridge_kdtree = None
        self._ridge_vert_idx = None

    def _snap_to_ridge(self, idx: int) -> int:
        if self.trim_mode != "scalloped":
            return idx
        tree = self._ensure_ridge_kdtree()
        if tree is None:
            return idx
        d, j = tree.query(self.mesh.vertices[idx])
        if d <= self._snap_radius:
            return int(self._ridge_vert_idx[j])
        print(
            f"[qt_viewer] snap: brak margin w {self._snap_radius:.1f}mm "
            f"(najbliższy {d:.1f}mm)"
        )
        return idx

    def _scalloped_path(self, a: int, b: int):
        from scipy.sparse.csgraph import dijkstra

        G = self._ensure_snap_graph()
        V = self.mesh.vertices
        va, vb = V[a], V[b]
        seg = vb - va
        L2 = float(seg @ seg)
        if L2 < 1e-12:
            dist_seg = np.linalg.norm(V - va, axis=1)
        else:
            t = np.clip(((V - va) @ seg) / L2, 0.0, 1.0)
            proj = va + t[:, None] * seg
            dist_seg = np.linalg.norm(V - proj, axis=1)
        mask = dist_seg <= self._corridor_tol
        mask[a] = mask[b] = True
        idx_map = np.where(mask)[0]
        sub = G[mask][:, mask]
        ra = int(np.searchsorted(idx_map, a))
        rb = int(np.searchsorted(idx_map, b))
        d, pred = dijkstra(sub, indices=ra, return_predecessors=True)
        if not np.isfinite(d[rb]):
            return None
        path = [rb]
        j = rb
        while j != ra:
            j = int(pred[j])
            if j < 0:
                return None
            path.append(j)
        path.reverse()
        return idx_map[np.asarray(path, dtype=np.int64)]

    def _compute_path(self, a: int, b: int) -> np.ndarray:
        if self.trim_mode == "scalloped":
            p = self._scalloped_path(a, b)
            if p is not None and p.size:
                return p
            print("[qt_viewer] korytarz: brak ścieżki — fallback pełny snap.")
            return shortest_path(self._ensure_snap_graph(), a, b)
        return shortest_path(self.graph, a, b)

    # =====================================================================
    # parametry — callbacki
    # =====================================================================
    def _on_angle_change(self, v_deg):
        self._angle_threshold = float(np.radians(v_deg))
        self._invalidate_ridge()
        if self._curvature_view:
            self._show_curvature()

    def _on_min_comp_change(self, v):
        self._min_comp_edges = int(v)
        self._invalidate_ridge()
        if self._curvature_view:
            self._show_curvature()

    def _on_snap_radius_change(self, v):
        self._snap_radius = float(v)

    def _on_snap_strength_change(self, v):
        self._snap_strength = float(v)
        # tylko snap_graph zależy od strength, ridge cache zostaje
        self._snap_graph = None

    def _on_corridor_change(self, v):
        self._corridor_tol = float(v)

    def _on_offset_change(self, v):
        self.params.inner_clearance = float(v)

    def _on_thickness_change(self, v):
        self.params.thickness = float(v)

    def _on_fillet_change(self, v):
        self.params.fillet_radius = float(v)

    def _on_trim_smooth_change(self, v):
        self.params.trim_smooth_sigma = float(v)

    def _on_quality_change(self):
        if self.rb_preview.isChecked():
            self._quality_mode = "preview"
            self.params.voxel_pitch = 0.20
        else:
            self._quality_mode = "final"
            self.params.voxel_pitch = 0.10
        print(f"[qt_viewer] Jakość: {self._quality_mode.upper()} (pitch={self.params.voxel_pitch})")

    # =====================================================================
    # pick / waypoints / close / fill
    # =====================================================================
    def _on_pick(self, screen_x: int, screen_y: int):
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        picker.Pick(screen_x, screen_y, 0, self.plotter.renderer)
        if picker.GetCellId() < 0:
            return
        wp = picker.GetPickPosition()
        new_idx = int(nearest_vertex(self.mesh, np.asarray(wp, dtype=float)))

        if self._awaiting_seed:
            self._awaiting_seed = False
            if not self.is_closed:
                print("[qt_viewer] SEED anulowany — pętla nie zamknięta.")
                self._refresh()
                return
            if self.boundary_mask[new_idx]:
                print("[qt_viewer] SEED na boundary — kliknij dalej od konturu.")
                self._refresh()
                return
            interior = fill_from_seed(self.graph, self.boundary_mask, new_idx)
            self.fill_mask = interior
            print(f"[qt_viewer] SEED fill: {interior.sum()} verts.")
            self._refresh()
            return

        if self.is_closed:
            print("[qt_viewer] Pętla zamknięta — X clear / Z otwiera.")
            return

        # scalloped: snap waypointa do margin (auto-trace)
        new_idx = self._snap_to_ridge(new_idx)

        # zamykanie pętli klikiem blisko zielonego
        if len(self.waypoints) >= 3:
            d = float(
                np.linalg.norm(self.mesh.vertices[new_idx] - self.mesh.vertices[self.waypoints[0]])
            )
            if d < self._close_thresh:
                self._close_loop()
                return

        self.waypoints.append(new_idx)
        if len(self.waypoints) >= 2:
            seg = self._compute_path(self.waypoints[-2], new_idx)
            if seg.size == 0:
                print("[qt_viewer] Brak ścieżki — cofam waypoint.")
                self.waypoints.pop()
            else:
                self.path_segments.append(seg)
                self.boundary_mask[seg] = True
        else:
            self.boundary_mask[new_idx] = True
        self._refresh()

    def _close_loop(self):
        if self.is_closed or len(self.waypoints) < 3:
            return
        seg = self._compute_path(self.waypoints[-1], self.waypoints[0])
        if seg.size == 0:
            print("[qt_viewer] Nie udało się zamknąć pętli.")
            return
        self.path_segments.append(seg)
        self.boundary_mask[seg] = True
        self.is_closed = True
        print("[qt_viewer] Pętla zamknięta — auto-fill...")
        self._refresh()
        self._fill()  # auto-fill

    def _fill(self):
        if not self.is_closed:
            print("[qt_viewer] Najpierw zamknij pętlę.")
            return
        interior = fill_interior(self.graph, self.boundary_mask, prefer="smaller")
        if not interior.any():
            print("[qt_viewer] Wnętrze puste — użyj G (seed-fill).")
            return
        self.fill_mask = interior
        print(f"[qt_viewer] Wypełniono: {interior.sum()} verts.")
        self._refresh()

    def _arm_seed_fill(self):
        if not self.is_closed:
            print("[qt_viewer] SEED dostępny po zamknięciu pętli.")
            return
        self._awaiting_seed = True
        print("[qt_viewer] SEED: kliknij ZIARNO wewnątrz pętli.")
        self._update_status()

    def _invert_fill(self):
        if not self.is_closed:
            print("[qt_viewer] INVERT dostępny po zamknięciu pętli.")
            return
        interior = fill_interior(self.graph, self.boundary_mask, prefer="larger")
        if not interior.any():
            print("[qt_viewer] Nie udało się odwrócić.")
            return
        self.fill_mask = interior
        print(f"[qt_viewer] Odwrócono: {interior.sum()} verts.")
        self._refresh()

    def _undo(self):
        if self.is_closed:
            self.is_closed = False
            if self.path_segments:
                self.path_segments.pop()
            self.boundary_mask[:] = False
            for s in self.path_segments:
                self.boundary_mask[s] = True
            for wp in self.waypoints:
                self.boundary_mask[wp] = True
            self._refresh()
            return
        if not self.waypoints:
            return
        self.waypoints.pop()
        if self.path_segments:
            self.path_segments.pop()
        self.boundary_mask[:] = False
        for s in self.path_segments:
            self.boundary_mask[s] = True
        for wp in self.waypoints:
            self.boundary_mask[wp] = True
        self._refresh()

    def _clear(self):
        self.waypoints.clear()
        self.path_segments.clear()
        self.boundary_mask[:] = False
        self.fill_mask[:] = False
        self.is_closed = False
        self._awaiting_seed = False
        print("[qt_viewer] Wyczyszczono.")
        self._refresh()

    # =====================================================================
    # trim mode / heatmapa
    # =====================================================================
    def _toggle_trim_mode(self):
        if self.trim_mode == "manual":
            self.trim_mode = "scalloped"
            self._ensure_snap_graph()
            self._act_trim.setText("Trim: Scalloped")
            print(
                f"[qt_viewer] TRIM: SCALLOPED — waypoint i ścieżka snap do margin "
                f"(korytarz {self._corridor_tol:.1f}mm)."
            )
        else:
            self.trim_mode = "manual"
            self._act_trim.setText("Trim: Manual")
            print("[qt_viewer] TRIM: MANUAL — geodesic.")
        self._update_status()

    def _toggle_curvature_view(self):
        self._curvature_view = not self._curvature_view
        if self._curvature_view:
            self._show_curvature()
        else:
            self._add_main_mesh(scalars="selected")
            self.plotter.render()

    def _show_curvature(self):
        ridge = self._ensure_ridge()
        vr = vertex_ridge_scores(self.mesh, ridge)
        self.pv_mesh.point_data["ridge"] = vr.astype(np.float32)
        self._add_main_mesh(scalars="ridge")
        self.plotter.render()

    # =====================================================================
    # generate / export / show-hide
    # =====================================================================
    def _generate_aligner(self):
        """UI feedback teraz, ciężka praca w następnym ticku event loopu.

        NIE używamy `processEvents()` — to powoduje re-entrancy do Qt event
        loopu w trakcie slotu i przy aktywnym override cursor + QtInteractor
        VTK potrafi assertować (SIGTRAP). `QTimer.singleShot(0)` deferruje
        wykonanie do następnego ticka — UI zdąży się zaktualizować, bez
        re-entrancy.
        """
        print("[qt_viewer] _generate_aligner: start", flush=True)
        try:
            if not self._combined_mask().any():
                print("[qt_viewer] Brak selekcji — najpierw zaznacz powierzchnię.")
                return
            if self._combined_mask().sum() < 5000:
                print("[qt_viewer] UWAGA: selekcja < 5000 verts — nakładka będzie mała.")
            eta = "12" if self._quality_mode == "preview" else "90"
            self.btn_generate.setEnabled(False)
            self.status.showMessage(
                f"Generuję nakładkę ({self._quality_mode.upper()})... "
                f"— okno zamarza na ~{eta}s (Faza 3 doda QThread + progress)"
            )
            print("[qt_viewer] _generate_aligner: scheduling _do_generate", flush=True)
            # defer ciężkiej pracy do następnego event-loop tick → UI flush, brak re-entrancy
            QtCore.QTimer.singleShot(0, self._do_generate)
        except Exception as e:
            import traceback
            print(f"[qt_viewer] _generate_aligner ERROR: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()

    def _do_generate(self):
        """Faktyczna ciężka generacja — odpalana po ticku event loopu."""
        import sys
        try:
            print(
                f"[qt_viewer] generate: offset={self.params.inner_clearance:.3f} "
                f"grub={self.params.thickness:.2f} fillet={self.params.fillet_radius:.2f} "
                f"trim_smooth={self.params.trim_smooth_sigma:.1f} pitch={self.params.voxel_pitch:.2f}",
                flush=True,
            )
            mesh, report = generate_aligner(self.mesh, self._combined_mask(), self.params)
            self.aligner_mesh = mesh
            self.rim_mask = report.rim_mask  # S1: krawędź dla podpór
            # S2: apex polyline cache invalidate (nowa nakładka = nowy apex)
            self._apex_loop = None
            print(
                f"[qt_viewer] OK: {len(mesh.vertices)} verts, "
                f"watertight={mesh.is_watertight}, t={report.total_seconds:.1f}s"
                + (
                    f", rim={int(self.rim_mask.sum())}v"
                    if self.rim_mask is not None
                    else ""
                ),
                flush=True,
            )
            for n in report.notes:
                print(f"  ! {n}", flush=True)
            # S5: nowa nakładka → invaliduj cache detekcji przód/tył
            self._ap_frame = None
            self._refresh_aligner()
            self._refresh_rim()
            # S2: wyciągnij apex + sampluj kotwice
            self._recompute_anchors()
            self._refresh_anchors()
            # S5: auto-włącz widok druku (pokaż print-ready: tilt + podpory)
            if not self._print_view:
                self._print_view = True
                self._act_print_view.setChecked(True)
            self._display_matrix = self._ensure_print_transform()
            self._apply_display_matrix()
            self._refresh_ground()
            # S3/S5: pillary w print space (pionowo do raftu)
            self._recompute_pillars()
            self._refresh_pillars()
            self.plotter.reset_camera()
        except Exception as e:
            print(f"[qt_viewer] Generate ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            QtWidgets.QMessageBox.critical(self, "Błąd generacji", str(e))
        finally:
            self.btn_generate.setEnabled(True)
            self._update_status()

    def _refresh_aligner(self):
        if self._aligner_actor is not None:
            self.plotter.remove_actor(self._aligner_actor, render=False)
            self._aligner_actor = None
        if self.aligner_mesh is None or not self._act_show_aligner.isChecked():
            self.plotter.render()
            return
        faces_flat = np.hstack(
            [np.full((len(self.aligner_mesh.faces), 1), 3, dtype=np.int64), self.aligner_mesh.faces]
        ).ravel()
        poly = pv.PolyData(np.asarray(self.aligner_mesh.vertices), faces_flat)
        self._aligner_actor = self.plotter.add_mesh(
            poly,
            color=ALIGNER_COLOR,
            opacity=self._aligner_opacity,
            smooth_shading=True,
            specular=0.3,
            specular_power=15,
            name="aligner_actor",
            reset_camera=False,
        )
        self._set_matrix(self._aligner_actor)
        self.plotter.render()

    def _toggle_aligner_visibility(self):
        self._refresh_aligner()

    def _refresh_rim(self):
        """S1: render zielone kulki na vertices rim (krawędź trim nakładki)."""
        if self._rim_actor is not None:
            self.plotter.remove_actor(self._rim_actor, render=False)
            self._rim_actor = None
        if (
            self.aligner_mesh is None
            or self.rim_mask is None
            or not self._act_show_rim.isChecked()
        ):
            self.plotter.render()
            return
        rim_pts = np.asarray(self.aligner_mesh.vertices)[self.rim_mask]
        if rim_pts.size == 0:
            self.plotter.render()
            return
        self._rim_actor = self.plotter.add_mesh(
            pv.PolyData(rim_pts),
            color="#22c55e",  # zielony — kontrast do cyjanowej nakładki
            point_size=6,
            render_points_as_spheres=True,
            name="rim_actor",
            reset_camera=False,
        )
        self._set_matrix(self._rim_actor)
        self.plotter.render()

    def _toggle_rim_visibility(self):
        self._refresh_rim()

    # ---------- S2: apex / kotwice ----------
    def _recompute_anchors(self):
        """Wyciągnij apex polyline i sampluj kotwice. Wywoływane raz po
        generacji + przy każdej zmianie slidera rozstawu (resampling jest tani)."""
        from .supports import extract_apex_loop, sample_apex_loop

        if self.aligner_mesh is None or self.rim_mask is None:
            self._apex_loop = None
            self.anchors = None
            return
        if self._apex_loop is None:
            self._apex_loop = extract_apex_loop(self.aligner_mesh, self.rim_mask)
            print(f"[qt_viewer] apex polyline: {len(self._apex_loop)} pts")
        self.anchors = sample_apex_loop(self._apex_loop, self._anchor_spacing)
        print(
            f"[qt_viewer] kotwice ({self._anchor_spacing:.1f}mm): "
            f"{len(self.anchors)} pkt"
        )

    def _refresh_anchors(self):
        """Render kotwic jako żółte kulki (większe niż rim dots)."""
        if self._anchors_actor is not None:
            self.plotter.remove_actor(self._anchors_actor, render=False)
            self._anchors_actor = None
        if (
            self.anchors is None
            or len(self.anchors) == 0
            or not self._act_show_anchors.isChecked()
        ):
            self.plotter.render()
            return
        self._anchors_actor = self.plotter.add_mesh(
            pv.PolyData(self.anchors),
            color="#fbbf24",  # żółty — kontrast do zielonego rim
            point_size=14,
            render_points_as_spheres=True,
            name="anchors_actor",
            reset_camera=False,
        )
        self._set_matrix(self._anchors_actor)
        self.plotter.render()

    def _toggle_anchors_visibility(self):
        self._refresh_anchors()

    def _on_anchor_spacing_change(self, v):
        """Slider live — resampling jest tani (~ms), nie wymaga regeneracji."""
        self._anchor_spacing = float(v)
        if self._apex_loop is not None:
            from .supports import sample_apex_loop
            self.anchors = sample_apex_loop(self._apex_loop, self._anchor_spacing)
            self._refresh_anchors()
            # zmiana liczby kotwic → rebuild pillars
            self._recompute_pillars()
            self._refresh_pillars()

    # ---------- S3: pillar geometry ----------
    def _recompute_pillars(self):
        """Pillary w print space: kotwice → orientacja druku → pionowo -Z do
        raftu Z=0. Budowane TYLKO w widoku druku (poza nim "dół" niezdefiniowany).
        """
        from .supports import generate_pillars_to_plane, transform_points

        if self.anchors is None or not self._print_view:
            self.pillars_mesh = None
            return
        anchors_print = transform_points(self.anchors, self._display_matrix)
        self.pillars_mesh = generate_pillars_to_plane(
            anchors_print, 0.0, self.pillar_params
        )
        if self.pillars_mesh is not None:
            print(
                f"[qt_viewer] pillars→raft: {len(self.anchors)} kotwic → "
                f"{len(self.pillars_mesh.vertices)}v / {len(self.pillars_mesh.faces)}f"
            )

    def _refresh_pillars(self):
        """Render pillars jako bryła pomarańczowa (~kontrast do cyjana aligner)."""
        if self._pillars_actor is not None:
            self.plotter.remove_actor(self._pillars_actor, render=False)
            self._pillars_actor = None
        if (
            self.pillars_mesh is None
            or not self._act_show_pillars.isChecked()
        ):
            self.plotter.render()
            return
        # konwersja trimesh → pv.PolyData
        faces_flat = np.hstack(
            [np.full((len(self.pillars_mesh.faces), 1), 3, dtype=np.int64),
             self.pillars_mesh.faces]
        ).ravel()
        poly = pv.PolyData(np.asarray(self.pillars_mesh.vertices), faces_flat)
        self._pillars_actor = self.plotter.add_mesh(
            poly,
            color="#f97316",   # pomarańczowy — kontrast do cyjana
            smooth_shading=True,
            specular=0.4,
            specular_power=20,
            name="pillars_actor",
            reset_camera=False,
        )
        self.plotter.render()

    def _toggle_pillars_visibility(self):
        self._refresh_pillars()

    def _on_tip_dia_change(self, v):
        self.pillar_params.tip_diameter = float(v)
        self._recompute_pillars()
        self._refresh_pillars()

    def _on_body_dia_change(self, v):
        self.pillar_params.body_diameter = float(v)
        self._recompute_pillars()
        self._refresh_pillars()

    def _on_pillar_h_change(self, v):
        self.pillar_params.pillar_height = float(v)
        self._recompute_pillars()
        self._refresh_pillars()

    # ---------- S5: orientacja druku (nachylenie + front/back) ----------
    def _ensure_print_transform(self):
        from .supports import compute_print_transform, detect_anterior_posterior

        if self.aligner_mesh is None:
            return np.eye(4)
        if self._ap_frame is None:
            self._ap_frame = detect_anterior_posterior(
                self.aligner_mesh, rim_mask=self.rim_mask
            )
        self._print_transform = compute_print_transform(
            self.aligner_mesh, self._ap_frame, self._tilt_angle, z_gap=2.0
        )
        return self._print_transform

    def _apply_display_matrix(self):
        """Ustaw user_matrix na wszystkich aktorach scan-space (nachyl je)."""
        for actor in (
            self._mesh_actor, self._aligner_actor, self._rim_actor,
            self._anchors_actor, self._waypoint_actor,
            self._first_waypoint_actor, self._contour_actor,
        ):
            self._set_matrix(actor)

    def _refresh_ground(self):
        """Płaszczyzna Z=0 (platforma druku) — tylko w widoku druku."""
        from .supports import transform_points

        if self._ground_actor is not None:
            self.plotter.remove_actor(self._ground_actor, render=False)
            self._ground_actor = None
        if not self._print_view or self.aligner_mesh is None:
            return
        at = transform_points(
            np.asarray(self.aligner_mesh.vertices), self._display_matrix
        )
        cx, cy = float(at[:, 0].mean()), float(at[:, 1].mean())
        sx = float(at[:, 0].max() - at[:, 0].min()) * 1.4 + 5.0
        sy = float(at[:, 1].max() - at[:, 1].min()) * 1.4 + 5.0
        plane = pv.Plane(
            center=(cx, cy, 0.0), direction=(0, 0, 1), i_size=sx, j_size=sy
        )
        self._ground_actor = self.plotter.add_mesh(
            plane, color="#2a2a2a", opacity=0.55, name="ground_actor",
            reset_camera=False,
        )

    def _toggle_print_view(self):
        self._print_view = self._act_print_view.isChecked()
        if self._print_view:
            if self.aligner_mesh is None:
                print("[qt_viewer] Widok druku: najpierw wygeneruj nakładkę.")
                self._act_print_view.setChecked(False)
                self._print_view = False
                return
            self._display_matrix = self._ensure_print_transform()
        else:
            self._display_matrix = np.eye(4)
        self._apply_display_matrix()
        self._recompute_pillars()
        self._refresh_pillars()
        self._refresh_ground()
        self.plotter.reset_camera()
        self.plotter.render()
        print(
            f"[qt_viewer] Widok druku: {'ON (tilt %.0f°)' % self._tilt_angle if self._print_view else 'OFF'}"
        )

    def _on_tilt_change(self, v):
        self._tilt_angle = float(v)
        if not self._print_view or self.aligner_mesh is None:
            return
        self._display_matrix = self._ensure_print_transform()
        self._apply_display_matrix()
        self._recompute_pillars()
        self._refresh_pillars()
        self._refresh_ground()
        self.plotter.render()

    def _toggle_mesh_visibility(self):
        if self._mesh_actor is None:
            return
        try:
            visible = self._act_show_mesh.isChecked()
            self._mesh_actor.SetVisibility(1 if visible else 0)
        except Exception:
            pass
        self.plotter.render()

    def _export_aligner(self):
        if self.aligner_mesh is None:
            print("[qt_viewer] Brak nakładki do eksportu — wygeneruj najpierw (N).")
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        default_name = OUTPUT_DIR / f"{self.loaded.stl_path.stem}_aligner.stl"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Eksport STL", str(default_name), "STL (*.stl)"
        )
        if not path:
            return
        self.aligner_mesh.export(path)
        print(f"[qt_viewer] Zapisano: {path}")

    # =====================================================================
    # save / load selekcji / wczytaj STL
    # =====================================================================
    def _save(self):
        mask = self._combined_mask()
        if not mask.any():
            print("[qt_viewer] Brak selekcji do zapisu.")
            return
        path = save_selection(self.loaded, mask)
        print(f"[qt_viewer] Zapisano selekcję: {path}")

    def _reload(self):
        m = load_selection(self.loaded)
        if m is None:
            print("[qt_viewer] Brak zapisanej selekcji.")
            return
        self.fill_mask = m
        self.waypoints.clear()
        self.path_segments.clear()
        self.boundary_mask[:] = False
        self.is_closed = False
        print(f"[qt_viewer] Wczytano selekcję: {m.sum()} verts.")
        self._refresh()

    def _load_stl_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Wczytaj STL", str(self.loaded.stl_path.parent), "STL (*.stl)"
        )
        if not path:
            return
        # Najprostsze: nowa instancja okna z nowym STL, zamknij obecne.
        try:
            new_loaded = load_stl(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Błąd ładowania", str(e))
            return
        global _CURRENT_WIN  # noqa: PLW0603
        _CURRENT_WIN = AlignerMainWindow(new_loaded)
        _CURRENT_WIN.show()
        self.close()


# ===========================================================================
# Entry helpers
# ===========================================================================
_CURRENT_WIN: AlignerMainWindow | None = None


def _install_crash_diagnostics():
    """faulthandler → natywny stack na SIGTRAP/SIGSEGV. excepthook → wyjątki
    Python (PySide6 6.11 abortuje na nieobsłużonym wyjątku w slocie, często
    bez widocznego tracebacku — to go wyłapuje i drukuje przed abortem)."""
    import faulthandler
    faulthandler.enable()

    def _hook(exc_type, exc_value, exc_tb):
        import traceback
        print("=" * 60, flush=True)
        print("[qt_viewer] NIEOBSŁUŻONY WYJĄTEK:", flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.stdout.flush()
        sys.stderr.flush()

    sys.excepthook = _hook


def run(stl_path: str | Path) -> int:
    _install_crash_diagnostics()
    p = Path(stl_path)
    if not p.exists():
        print(f"[qt_viewer] Plik nie istnieje: {p}", file=sys.stderr)
        return 2
    loaded = load_stl(p)
    print(
        f"[qt_viewer] Mesh: {len(loaded.mesh.vertices)} verts, "
        f"{len(loaded.mesh.faces)} faces, watertight={loaded.mesh.is_watertight}"
    )
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    global _CURRENT_WIN  # noqa: PLW0603
    _CURRENT_WIN = AlignerMainWindow(loaded)
    _CURRENT_WIN.show()
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) < 2:
        print("Usage: python -m aligner_gen.qt_viewer <stl_path>")
        sys.exit(1)
    sys.exit(run(sys.argv[1]))
