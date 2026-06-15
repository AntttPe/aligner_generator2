"""Presety drukarka + żywica — zapis/odczyt zestawu parametrów do JSON.

Każda kombinacja drukarka×żywica ma własne optymalne parametry (touchpoint,
grubość ścianki, raft, orientacja). Zamiast trzymać je w głowie, zapisujemy
nazwany preset do JSON i ładujemy jednym kliknięciem. Built-in presety oparte
na rekomendacjach Formlabs; własne presety użytkownik zapisuje do `data/presets/`.

Fundament pod next-gen UI / web: warstwa danych jest tu, GUI tylko ją woła.

Wartości Formlabs (źródła):
  - touchpoint default 0.40mm (zakres 0.10–1.00); tough/aligner resin → 0.4–0.5
  - penetracja kontaktu 0.10–0.15mm
  - direct-print aligner: grubość ścianki 0.5–1.0mm, layer 0.05–0.10mm
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PRESETS_DIR = Path("data/presets")
PRESET_VERSION = 1


@dataclass
class PrintPreset:
    """Komplet parametrów dla jednej kombinacji drukarka×żywica.

    Grupy: nakładka (aligner) / kontakt-pillary / raft / strategia podpór.
    Pola dobrane tak, by dało się je 1:1 zaaplikować na `AlignerParams`,
    `PillarParams` oraz pola raftu w viewerze.
    """
    name: str = "Custom"
    printer: str = ""
    resin: str = ""

    # --- nakładka (AlignerParams) ---
    thickness: float = 1.0            # grubość ścianki [mm]
    inner_clearance: float = 0.05     # offset od zębów [mm]
    voxel_pitch: float = 0.10         # rozdzielczość voxela [mm]

    # --- kontakt / pillary (PillarParams) ---
    tip_diameter: float = 0.4         # touchpoint Ø [mm] (Formlabs default 0.40)
    body_diameter: float = 1.0        # trzon Ø [mm]
    tip_penetration: float = 0.12     # wbicie w part [mm] (Formlabs 0.10–0.15)
    use_ball_tip: bool = True

    # --- raft ---
    raft_thickness: float = 0.8       # [mm] — cienki: footprint decyduje o adhezji
    raft_band_width: float = 4.0      # [mm]
    anti_peel_tabs: int = 4           # pady na ekstremach pętli

    # --- strategia podpór ---
    anchor_spacing: float = 3.0       # gęstość kotwic wzdłuż rim [mm]
    overhang_angle: float = 45.0      # próg nawisu [°]
    branch_supports: bool = False     # adaptacyjne podpory gałęziące z pillarów

    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["_version"] = PRESET_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PrintPreset":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def save(self, directory: Path | str = PRESETS_DIR) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)
        path = directory / f"{slug}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
        return path


def load_preset(path: Path | str) -> PrintPreset:
    data = json.loads(Path(path).read_text())
    return PrintPreset.from_dict(data)


def builtin_presets() -> dict[str, PrintPreset]:
    """Presety fabryczne (Formlabs-recommended baseline). Klucz = nazwa."""
    presets = [
        PrintPreset(
            name="Formlabs Form 4B — LT Comfort",
            printer="Formlabs Form 4B",
            resin="LT Comfort V1.1",
            thickness=1.0, inner_clearance=0.05, voxel_pitch=0.10,
            tip_diameter=0.5, body_diameter=1.0, tip_penetration=0.12,
            use_ball_tip=True,
            raft_thickness=0.8, raft_band_width=4.0, anti_peel_tabs=4,
            anchor_spacing=3.0, overhang_angle=45.0, branch_supports=True,
            notes="Tough/elastyczny aligner resin → touchpoint 0.5mm (pewny chwyt, "
                  "czysty snap na finiszowanej krawędzi trim). Form 4B = niskie peel, "
                  "thin-shell self-supporting (warning 'more support' na kopule = ignore).",
        ),
        PrintPreset(
            name="Formlabs Form 3B+ — Dental LT Clear",
            printer="Formlabs Form 3B+",
            resin="Dental LT Clear V2",
            thickness=0.8, inner_clearance=0.05, voxel_pitch=0.10,
            tip_diameter=0.4, body_diameter=1.0, tip_penetration=0.12,
            use_ball_tip=True,
            raft_thickness=1.0, raft_band_width=4.0, anti_peel_tabs=4,
            anchor_spacing=3.0, overhang_angle=45.0, branch_supports=False,
            notes="Wyższe peel forces niż Form 4B → grubszy raft (1.0mm). "
                  "Touchpoint Formlabs default 0.4mm.",
        ),
        PrintPreset(
            name="Generic — szybka iteracja (PREVIEW)",
            printer="dowolna",
            resin="dowolna",
            thickness=1.0, inner_clearance=0.05, voxel_pitch=0.20,
            tip_diameter=0.4, body_diameter=1.0, tip_penetration=0.12,
            use_ball_tip=True,
            raft_thickness=0.8, raft_band_width=4.0, anti_peel_tabs=4,
            anchor_spacing=3.0, overhang_angle=45.0, branch_supports=False,
            notes="Coarse pitch 0.20mm do szybkiej iteracji parametrów (~12s). "
                  "Przed drukiem przełącz na pitch 0.10mm (FINAL).",
        ),
    ]
    return {p.name: p for p in presets}


def list_presets(directory: Path | str = PRESETS_DIR) -> dict[str, PrintPreset]:
    """Wszystkie dostępne presety: built-in + zapisane w `directory` (JSON).

    Presety użytkownika o tej samej nazwie nadpisują built-in.
    """
    out = builtin_presets()
    directory = Path(directory)
    if directory.is_dir():
        for p in sorted(directory.glob("*.json")):
            try:
                preset = load_preset(p)
                out[preset.name] = preset
            except Exception as e:  # pragma: no cover
                print(f"[presets] pominięto {p.name}: {e}")
    return out
