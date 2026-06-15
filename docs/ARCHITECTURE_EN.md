# Dental Aligner Generator — Architecture & Design Rationale

A tool that turns a **scanned STL of a dental arch** into a **3D‑printable clear‑aligner
shell** with print supports — driven by a manual surface selection, no ML, no auto‑segmentation.
Target workflow: **direct‑printed aligners** on Formlabs / Asiga resin printers (the STL maps 1:1
to the printed part).

---

## 1. What problem are we solving?

Given a tooth scan, produce a thin shell (0.5–2.0 mm) that snaps over the teeth, with:
- **no self‑intersections**,
- **no surface passing through teeth**,
- **embrasures (interdental gaps) bridged smoothly from outside, not filled**,
- **constant wall thickness**, **watertight & manifold** output,
- **print supports placed on the trim edge** (easy removal, no grinding of fitting surfaces).

A previous C++/meshlib attempt failed because of the *algorithm*, not the language/library —
so the central design choice here is **how** we offset the surface.

---

## 2. Tech stack (and why each)

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.13 | fast iteration on the algorithm |
| Mesh I/O & repair | **trimesh** | robust STL load, `outline()`, boolean‑free utilities, Taubin smoothing |
| Volumetric core | **NumPy + SciPy** (`ndimage`) | voxel grids, SDF via Euclidean Distance Transform, morphology, sparse graphs |
| Iso‑surfacing | **scikit‑image** `marching_cubes` | field → triangle mesh |
| 3D viewport | **VTK** via **pyvistaqt** | render + picking |
| Desktop UI | **PySide6** (LGPL) | dockable panels, sliders, toolbars, threads |

> **OpenVDB was the original plan** for the SDF core, but has no Python 3.13 wheels. `sdf.py`
> is an abstraction layer over a SciPy backend, so OpenVDB can be slotted in later without
> touching the pipeline.

---

## 3. End‑to‑end pipeline

```
STL ─► load + repair ─► manual selection ─► VOXEL/SDF GENERATION ─► aligner shell ─► SUPPORTS ─► print STL
        (watertight)     (contour + fill)    (the core)             (watertight)     (orient+raft)
```

### 3.1 Load & repair (`io.py`)
STL triangles are merged into a shared‑vertex mesh. If not watertight, we **cap open boundary
loops with small triangles** (index‑preserving — original vertices keep indices `0..N‑1` so a
saved selection stays valid). Success criterion is *"does it fill as a solid"* (coarse voxel
fill‑ratio), **not** `is_watertight`, because the cap introduces harmless T‑junctions.

### 3.2 Selection (`selection.py`, `curvature.py`)
The user paints the adhesion surface:
- **Waypoints** are joined by the **shortest geodesic path over mesh edges** (Dijkstra). This
  prevents the selection from "leaking" to the opposite side of the arch.
- **Fill** uses connected components on the sub‑graph cut by the contour, filtered to components
  that *touch* the contour (ignores floating scan artefacts).
- **Live‑wire (intelligent scissors):** edges are weighted cheap where the **concave dihedral
  angle** is high (the gingival margin is a valley), so Dijkstra "snaps" the contour to the
  gum line. A **corridor constraint** keeps the path inside a tube around the click‑to‑click
  segment, so it can't wander into noise.

### 3.3 Voxel / SDF generation (`sdf.py`, `aligner.py`) — the core
1. **Voxelize** the whole mesh into a binary grid (inside/outside).
2. **Morphological closing** (dilate→erode) bridges embrasures *without filling the gap* —
   the closed surface "sees" the interdental space as outside.
3. **Signed Distance Field** via `distance_transform_edt` (negative inside, positive outside).
4. **Owner‑based proximity:** voxelize the *selected* faces and the *full* surface separately;
   `distance_transform_edt(return_indices=True)` tells each voxel which surface it is closest
   to. A voxel "belongs" to the selection only if its nearest surface voxel is selected → this
   **trims the aligner naturally at the gingival margin**, independent of a radius.
5. **Field assembly** — the aligner solid is `field > 0` where:
   ```
   wall        = min(sdf − inner_iso, outer_iso − sdf)   # constant thickness (HARD min)
   sel_field   = selection_radius − dist_to_selected     # stay within selection
   owner_field = signed distance to "owned" region       # trim at margin
   field = smin( wall, sel_field, owner_field )          # smooth‑min only on the cut edges
   ```
6. **Marching cubes** → mesh; keep the largest component; **Taubin smoothing**
   (volume‑preserving — removes voxel stair‑stepping without shrinking the shell).

### 3.4 Support generation (`supports.py`) — staged S1…S5
1. **Rim detection (S1):** at each output vertex we sample the component fields; a vertex is
   "rim" (the trim edge) when `sel/owner` is the binding constraint rather than `wall`. We can
   do this because **we generated the mesh** — a generic slicer can only guess.
2. **Apex line (S2):** the rim band is the shell's side‑wall (a cylinder topologically). We take
   `trimesh.outline()` of the rim sub‑mesh → its two boundary loops, keep the longest, and
   resample it at even arc‑length → evenly spaced **anchor points**.
3. **Pillars (S3):** each anchor gets a cone‑tip + cylinder‑body strut (tiny contact for clean
   snap‑off; the part geometry is never modified, supports are *added*).
4. **Print orientation (S5):** PCA gives the occlusal plane; a **midline‑gap test** finds the
   arch opening (posterior) vs. the closed front (anterior); the **rim centroid** orients the
   occlusal dome *up* and the trim edge *down*. A tilt slider (−45°…+45°) rotates the assembly;
   pillars drop **vertically to a Z=0 raft plane** with per‑pillar length.
5. **Raft + bracing (S4):** a **U‑shaped band** follows the anchor loop (the tongue space stays
   empty — saves resin); a **zig‑zag diagonal truss** (Warren pattern, ~45°) ties tall pillars
   together so they don't buckle under print peel forces.

All parts are exported as a **multi‑solid STL (concatenated, not boolean‑unioned)** — the slicer
accepts multiple disjoint manifold bodies, and concatenation can't fail the way a boolean can.

---

## 4. Key design decisions — *defend these in an interview*

| Decision | Alternative | Why ours |
|---|---|---|
| **SDF / voxel offset** | direct vertex‑normal offset | direct offset self‑intersects in concave regions and fills embrasures; an SDF *cannot* self‑intersect, bridges gaps naturally, and tolerates topology change |
| **EDT for distance fields** | `cKDTree.query` per voxel | O(N) flood vs. millions of nearest‑neighbour queries → **116 s → ~2 s** |
| **Owner‑based trim** | distance‑to‑points radius | dense coverage in grooves + a *natural* gingival cut that doesn't depend on tuning a radius |
| **Hard‑min wall, smooth‑min only on edges** | smooth‑min everywhere | smooth‑min on the inner/outer pair pinches the wall to zero; separating them keeps thickness exact while still rounding the trim edge |
| **Trim‑path smoothing ≠ SDF smoothing** | smooth the whole field | we Gaussian‑smooth only the *trim* fields → a clean edge **without** blurring the inner fit surface |
| **Boundary capping for repair** | pymeshfix | pymeshfix tripled the mesh and OOM‑crashed (57 GB); capping is index‑preserving and ~0.01 s |
| **`outline()` for the apex** | morphological skeleton | skeletonising a triangulated 2‑D band fragments into hundreds of components; the band boundary is a clean topological loop trimesh already computes |
| **`actor.user_matrix` for tilt** | rebuild the mesh per slider tick | a 4×4 matrix on the VTK actor re‑orients 170 k verts for free |
| **multi‑solid STL** | boolean union | the printer slices disjoint bodies together; boolean union is slow and fails on edge cases |
| **`QTimer.singleShot` for heavy work** | `processEvents()` | `processEvents()` re‑enters the Qt loop and SIGTRAPs on macOS; deferring to the next tick is re‑entrancy‑free |

---

## 5. Module map

```
src/aligner_gen/
├── io.py          STL load, watertight repair (boundary capping), selection save/load
├── selection.py   geodesic Dijkstra, flood fill, seed fill
├── curvature.py   per‑edge concave‑ridge score, denoise, live‑wire snap graph
├── sdf.py         VoxelGrid, voxelize, EDT‑SDF, morphology, owner proximity, marching cubes
├── aligner.py     the SDF pipeline → aligner shell + rim_mask
├── supports.py    rim → apex → pillars → orientation → raft → zig‑zag braces
├── qt_viewer.py   PySide6 desktop app (current UI)
└── viewer.py      legacy PyVista‑only viewer (fallback)
```

## 6. Running it

```bash
.venv/bin/pip install -e .
.venv/bin/python -m aligner_gen data/stl/scan.stl --qt   # PySide6 desktop UI
```

Workflow in the app: paint the contour (geodesic / live‑wire) → fill → **Generate** → tweak
sliders (offset, thickness, fillet, trim‑smoothing) → toggle **Print view** to orient + add
supports → export the print‑ready STL.

---

## 7. One‑paragraph summary (for the interview)

> *"It converts a tooth scan into a printable aligner using a **signed‑distance‑field pipeline**
> rather than naïve surface offsetting, which structurally eliminates the self‑intersection and
> embrasure‑filling problems that sank the previous attempt. Distance fields are computed with
> the **Euclidean Distance Transform** for speed, the trim edge comes from an **owner‑based
> proximity** test, and the wall thickness is held exact by separating a hard‑min thickness term
> from a smooth‑min fillet term. Because **we** build the mesh, we can tag the trim edge exactly
> and grow application‑aware **print supports on the rim** — placed on a tilted, front/back‑aware
> print orientation with a material‑saving U‑shaped raft and a zig‑zag truss against buckling."*
