# Generator nakładek stomatologicznych — architektura i uzasadnienie decyzji

Narzędzie zamieniające **zeskanowany STL łuku zębowego** w **gotową do druku 3D powłokę
nakładki** (clear aligner) wraz z podporami druku — sterowane ręcznym zaznaczeniem powierzchni,
bez ML i bez auto‑segmentacji.
Docelowy workflow: **nakładki drukowane bezpośrednio** (direct‑print) na drukarkach żywicznych
Formlabs / Asiga (STL odwzorowuje się 1:1 w gotowym detalu).

---

## 1. Jaki problem rozwiązujemy?

Z skanu zębów wytworzyć cienką powłokę (0.5–2.0 mm) nasuwaną na zęby, z gwarancją:
- **brak self‑intersection**,
- **brak przechodzenia powierzchni przez zęby**,
- **embrasury (przestrzenie międzyzębowe) bridgowane gładko z zewnątrz, NIE wypełniane**,
- **stała grubość ścianki**, mesh **watertight i manifold**,
- **podpory druku na krawędzi trim** (łatwe usuwanie, bez szlifowania powierzchni dopasowania).

Poprzednia próba w C++/meshlib nie powiodła się przez *algorytm*, nie przez język/bibliotekę —
dlatego kluczową decyzją jest **jak** offsetujemy powierzchnię.

---

## 2. Stack technologiczny (i dlaczego)

| Warstwa | Narzędzie | Dlaczego |
|---|---|---|
| Język | Python 3.13 | szybka iteracja nad algorytmem |
| Mesh I/O + naprawa | **trimesh** | solidny load STL, `outline()`, narzędzia bez booleanów, Taubin smoothing |
| Rdzeń wolumetryczny | **NumPy + SciPy** (`ndimage`) | siatki voxeli, SDF przez Euclidean Distance Transform, morfologia, grafy rzadkie |
| Iso‑surface | **scikit‑image** `marching_cubes` | pole skalarne → mesh trójkątny |
| Viewport 3D | **VTK** przez **pyvistaqt** | renderowanie + picking |
| GUI desktop | **PySide6** (LGPL) | dokowalne panele, suwaki, toolbary, wątki |

> **OpenVDB był pierwotnym planem** na rdzeń SDF, ale nie ma wheeli na Pythona 3.13. `sdf.py` to
> warstwa abstrakcji nad backendem SciPy, więc OpenVDB można podmienić później bez ruszania
> pipeline'u.

---

## 3. Pipeline end‑to‑end

```
STL ─► load + naprawa ─► zaznaczenie ─► GENERACJA VOXEL/SDF ─► powłoka nakładki ─► PODPORY ─► STL druku
        (watertight)     (kontur+fill)    (rdzeń)              (watertight)       (orient+raft)
```

### 3.1 Load i naprawa (`io.py`)
Trójkąty STL scalane do mesha o wspólnych wierzchołkach. Jeśli niewatertight — **zaślepiamy
otwarte pętle brzegowe małymi trójkątami** (zachowując indeksy — oryginalne wierzchołki
zostają `0..N‑1`, więc zapisana selekcja pozostaje ważna). Kryterium sukcesu to *"czy wypełnia
się jako bryła"* (zgrubny fill‑ratio voxeli), **nie** `is_watertight` — czapa wprowadza
nieszkodliwe T‑junctiony.

### 3.2 Zaznaczenie (`selection.py`, `curvature.py`)
Użytkownik maluje powierzchnię przylegania:
- **Waypointy** łączone **najkrótszą geodesic ścieżką po krawędziach** (Dijkstra). To zapobiega
  „wyciekaniu" selekcji na przeciwną stronę łuku.
- **Fill** używa connected components na subgrafie przeciętym konturem, filtrując komponenty
  *stykające się* z konturem (pomija pływające artefakty skanu).
- **Live‑wire (intelligent scissors):** krawędzie tanie tam gdzie wysoki **wklęsły kąt
  dwuścienny** (gingival margin to dolina), więc Dijkstra „przyciąga" kontur do linii dziąsła.
  **Korytarz tolerancji** trzyma ścieżkę w tubie wokół odcinka klik‑klik, żeby nie zbaczała w szum.

### 3.3 Generacja voxel/SDF (`sdf.py`, `aligner.py`) — rdzeń
1. **Voxelizacja** całego mesha do siatki binarnej (wnętrze/zewnętrze).
2. **Closing morfologiczny** (dylatacja→erozja) bridguje embrasury *bez wypełniania gapu* —
   zamknięta powierzchnia „widzi" przestrzeń międzyzębową jako zewnątrz.
3. **Signed Distance Field** przez `distance_transform_edt` (ujemne wewnątrz, dodatnie zewnątrz).
4. **Owner‑based proximity:** voxelizujemy osobno *zaznaczone* twarze i *całą* powierzchnię;
   `distance_transform_edt(return_indices=True)` mówi każdemu voxelowi do której powierzchni mu
   najbliżej. Voxel „należy" do selekcji tylko gdy jego najbliższy voxel powierzchni jest
   zaznaczony → to **naturalnie ucina nakładkę na gingival margin**, niezależnie od promienia.
5. **Złożenie pola** — bryła nakładki to `field > 0` gdzie:
   ```
   wall        = min(sdf − inner_iso, outer_iso − sdf)   # stała grubość (TWARDY min)
   sel_field   = selection_radius − dist_to_selected     # zostań w selekcji
   owner_field = signed distance do "owned" regionu       # trim na margin
   field = smin( wall, sel_field, owner_field )          # smooth‑min TYLKO na krawędziach cięcia
   ```
6. **Marching cubes** → mesh; największa komponenta; **Taubin smoothing** (volume‑preserving —
   usuwa schodkowanie z voxelizacji bez kurczenia powłoki).

### 3.4 Generacja podpór (`supports.py`) — etapy S1…S5
1. **Detekcja rim (S1):** w każdym vertexie próbkujemy pola komponentowe; vertex jest „rim"
   (krawędź trim) gdy wiążący jest `sel/owner`, a nie `wall`. Możemy to zrobić, bo **to MY
   generujemy mesh** — generic slicer może tylko zgadywać.
2. **Linia apex (S2):** pasmo rim to boczna ścianka shell‑a (topologicznie cylinder). Bierzemy
   `trimesh.outline()` sub‑mesha rim → dwie pętle brzegowe, wybieramy najdłuższą i resamplujemy
   równomiernie po długości łuku → równo rozłożone **kotwice**.
3. **Pillary (S3):** każda kotwica → stożek tip + cylinder body (mały kontakt = czysty snap‑off;
   geometria nakładki nietknięta, podpory są *dodane*).
4. **Orientacja druku (S5):** PCA daje płaszczyznę okluzyjną; **test midline‑gap** znajduje
   otwarcie podkowy (tył) vs zamknięty przód; **centroid rim** orientuje kopułę okluzyjną *do
   góry* a krawędź trim *w dół*. Slider nachylenia (−45°…+45°) obraca zestaw; pillary spadają
   **pionowo do płaszczyzny raftu Z=0** z indywidualną długością.
5. **Raft + wzmocnienia (S4):** **wstęga U** śledzi pętlę kotwic (przestrzeń języka zostaje pusta
   — oszczędność resinu); **zigzag truss** (wzór Warrena, ~45°) wiąże wysokie pillary, żeby się
   nie wyboczyły pod siłami peel druku.

Wszystko eksportowane jako **multi‑solid STL (konkatenacja, NIE boolean union)** — slicer
przyjmuje wiele rozłącznych brył manifold, a konkatenacja nie może zawieść jak boolean.

---

## 4. Kluczowe decyzje — *te broń na interview*

| Decyzja | Alternatywa | Dlaczego nasza |
|---|---|---|
| **Offset SDF / voxel** | direct vertex‑normal offset | direct self‑intersectuje w obszarach wklęsłych i wypełnia embrasury; SDF z definicji *nie może* się przeciąć, bridguje gapy, toleruje zmianę topologii |
| **EDT na pola odległości** | `cKDTree.query` per voxel | O(N) flood vs miliony zapytań NN → **116 s → ~2 s** |
| **Owner‑based trim** | promień distance‑to‑points | gęste pokrycie w grooves + *naturalne* cięcie na dziąśle niezależne od strojenia promienia |
| **Twardy min ścianki, smooth‑min tylko na krawędziach** | smooth‑min wszędzie | smooth‑min na parze inner/outer ścieńcza ściankę do zera; rozdzielenie trzyma grubość dokładnie a zaokrągla tylko trim |
| **Wygładzanie ścieżki trim ≠ wygładzanie SDF** | gładzić całe pole | gauss tylko na polach *trim* → czysta krawędź **bez** rozmycia powierzchni dopasowania |
| **Capping brzegów do naprawy** | pymeshfix | pymeshfix potrajał mesh i OOM‑ował (57 GB); capping zachowuje indeksy i trwa ~0.01 s |
| **`outline()` na apex** | skeletyzacja morfologiczna | skeletyzacja trójkątnego pasma 2D fragmentuje się na setki komponentów; brzeg pasma to czysta pętla topologiczna którą trimesh już liczy |
| **`actor.user_matrix` na nachylenie** | przebudowa mesha per tick suwaka | macierz 4×4 na aktorze VTK obraca 170 k verts za darmo |
| **multi‑solid STL** | boolean union | drukarka tnie rozłączne bryły razem; boolean jest wolny i wywala się na edge case'ach |
| **`QTimer.singleShot` na ciężką pracę** | `processEvents()` | `processEvents()` re‑entruje pętlę Qt i SIGTRAP‑uje na macOS; defer do następnego ticka jest bez re‑entrancy |

---

## 5. Mapa modułów

```
src/aligner_gen/
├── io.py          load STL, naprawa watertight (capping), zapis/odczyt selekcji
├── selection.py   geodesic Dijkstra, flood fill, seed fill
├── curvature.py   per‑edge score wklęsłego grzbietu, denoise, graf snap live‑wire
├── sdf.py         VoxelGrid, voxelize, EDT‑SDF, morfologia, owner proximity, marching cubes
├── aligner.py     pipeline SDF → powłoka nakładki + rim_mask
├── supports.py    rim → apex → pillary → orientacja → raft → zigzag wzmocnienia
├── qt_viewer.py   aplikacja desktop PySide6 (aktualne UI)
└── viewer.py      legacy viewer PyVista (fallback)
```

## 6. Uruchomienie

```bash
.venv/bin/pip install -e .
.venv/bin/python -m aligner_gen data/stl/scan.stl --qt   # UI desktop PySide6
```

Workflow w aplikacji: maluj kontur (geodesic / live‑wire) → fill → **Generuj** → dostrój suwaki
(offset, grubość, fillet, gładkość trim) → **Widok druku** żeby zorientować + dodać podpory →
eksport gotowego STL.

---

## 7. Podsumowanie w jednym akapicie (na interview)

> *„Zamienia skan zębów w drukowalną nakładkę przez **pipeline signed‑distance‑field**, a nie
> naiwny offset powierzchni — co strukturalnie eliminuje self‑intersection i wypełnianie
> embrasur, które pogrążyły poprzednie podejście. Pola odległości liczymy **Euclidean Distance
> Transformem** dla szybkości, krawędź trim wynika z **owner‑based proximity**, a grubość
> ścianki trzymamy dokładnie przez rozdzielenie twardego‑min grubości od smooth‑min filletu.
> Ponieważ **to my** budujemy mesh, możemy dokładnie otagować krawędź trim i wygenerować
> **podpory świadome zastosowania — na krawędzi**, w nachylonej orientacji druku z detekcją
> przód/tył, z oszczędzającym resin raftem U i zigzag trussem przeciw wyboczeniu."*
