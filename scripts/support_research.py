"""Support research sweep — baseline resin + Pareto (resin vs naprężenie peel).

KROK 1 z planu "tree supports": zanim zbudujemy branching, mierzymy ile resinu
zjadają OBECNE proste pillary i jak to się ma do wytrzymałości. Daje to liczby
uzasadniające (albo nie) refactor na drzewo + waliduje metryki.

Co liczymy — wszystko na REALNYM kodzie geometrii (`generate_pillars_to_plane`,
`make_raft`), więc objętości = dokładnie to co idzie na drukarkę:

  • resin [ml]            = (suma objętości pillarów + raft) / 1000
  • N pillarów            = f(spacing kotwic wzdłuż trim)
  • peel force / pillar   = f_line[N/mm] * spacing[mm]   (tributary length model)
  • σ_trunk [MPa]         = f_pillar / pole_przekroju body
  • σ_tip   [MPa]         = f_pillar / pole_przekroju tip   (kontakt z nakładką)

Model peel (udokumentowany, proxy): podczas druku każda warstwa odrywa się od
filmu FEP/PDMS — siła separacji działa wzdłuż obwodu skorupy. Modelujemy ją jako
obciążenie liniowe `f_line` [N/mm] wzdłuż linii trim. Pillar trzyma segment o
długości = spacing (tributary), więc:
    f_pillar = f_line * spacing
To fizycznie poprawna intuicja: rzadziej rozstawione pillary → każdy trzyma
dłuższy odcinek → większa siła → większe naprężenie. Bezwzględne MPa są
orientacyjne (f_line to proxy), ale TRENDY i Pareto są wiarygodne.

Pareto: szukamy małego resinu PRZY σ_trunk poniżej progu (żeby pillar nie pękł
zostawiając pieniek) i σ_tip kontrolowanego (clean snap-off bez rwania nakładki).

Uruchomienie:
    .venv/bin/python scripts/support_research.py
Wynik:
    data/output/support_research.csv   — pełna tabela sweepu
    data/output/support_research.png   — wykres Pareto
"""
from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from aligner_gen.supports import (
    PillarParams,
    generate_pillars_to_plane,
    make_raft,
)

OUT_DIR = Path("data/output")

# --- Model peel (proxy, udokumentowany powyżej) --------------------------------
F_LINE_N_PER_MM = 0.03      # obciążenie liniowe peel wzdłuż trim [N/mm]
# Progi klinika/materiał (orientacyjne dla sztywnego dental resin, np. LT/IBT):
SIGMA_TRUNK_LIMIT = 8.0     # MPa — powyżej: ryzyko pęknięcia trunka (pieniek na parcie)
SIGMA_TIP_SOFT = 25.0       # MPa — powyżej na tip: ryzyko rwania nakładki przy peel
# (te liczby służą do oznaczenia "czerwonej strefy" na wykresie, nie jako prawda absolutna)


def make_dental_arch_anchors(
    width_mm: float = 50.0,
    depth_mm: float = 40.0,
    arch_band_mm: float = 9.0,
    spacing_mm: float = 3.0,
    z_min: float = 6.0,
    z_var: float = 2.5,
    n_teeth: int = 14,
    scallop_mm: float = 1.2,
) -> tuple[np.ndarray, float]:
    """Realistyczny ZAMKNIĘTY trim łuku zębowego → kotwice co `spacing_mm`.

    Trim nakładki to zamknięta pętla biegnąca po stronie POLICZKOWEJ i JĘZYKOWEJ
    każdego zęba (nie pojedyncza krzywa) + scalloping girlandy wokół każdego zęba.
    Dlatego realny model daje ~100+ kotwic, nie ~25. Modelujemy to wiernie:

      • centerline = parabola U (łuk)
      • buccal rail = centerline + arch_band/2 (na zewnątrz)
      • lingual rail = centerline − arch_band/2 (do środka)
      • zamknięcie z tyłu (dystalnie) → pętla picture-frame
      • scalloping: pionowa girlanda (sinus n_teeth) ± scallop_mm — interproksymalne
        zagłębienia margin między zębami

    z = wysokość kotwicy nad raftem (z_top=0). Wariacja z_var = krzywa Spee
    (sieczne niżej, molary wyżej) → różne długości nóg jak w realnym pipeline.

    Zwraca (anchors[M,3], trim_length_mm).
    """
    def _rail(sign: float) -> np.ndarray:
        t = np.linspace(-1.0, 1.0, 240)
        cx = t * (width_mm / 2.0)
        cy = depth_mm * (t ** 2)
        # normalna 2D do centerline → offset na buccal/lingual rail
        dx = np.gradient(cx)
        dy = np.gradient(cy)
        nl = np.hypot(dx, dy)
        nx, ny = dy / nl, -dx / nl
        rx = cx + sign * (arch_band_mm / 2.0) * nx
        ry = cy + sign * (arch_band_mm / 2.0) * ny
        return np.column_stack([rx, ry])

    buccal = _rail(+1.0)
    lingual = _rail(-1.0)
    # pętla: buccal (lewo→prawo) + lingual (prawo→lewo) → zamknięta
    loop2d = np.vstack([buccal, lingual[::-1]])

    seg = np.linalg.norm(np.diff(loop2d, axis=0, append=loop2d[:1]), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])

    n = max(int(total / spacing_mm), 6)
    s_samples = np.linspace(0.0, total, n, endpoint=False)
    xs = np.interp(s_samples, arc[:-1], loop2d[:, 0])
    ys = np.interp(s_samples, arc[:-1], loop2d[:, 1])
    # wysokość: krzywa Spee (depth-driven) + girlanda scalloping (per-ząb sinus)
    z_spee = z_var * (ys / max(depth_mm, 1e-6))
    z_scallop = scallop_mm * np.abs(np.sin(s_samples / total * n_teeth * np.pi))
    zs = z_min + z_spee + z_scallop
    anchors = np.column_stack([xs, ys, zs])
    return anchors, total


def sweep():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = PillarParams()  # obecne defaulty = baseline

    # Osie sweepu
    spacings = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]          # gęstość kotwic
    body_diams = [0.6, 0.8, 1.0, 1.2]                        # grubość trzonu
    foot_modes = [("foot", base.foot_height, base.foot_extra_radius),
                  ("noFoot", 0.0, 0.0)]                       # fillet podstawy on/off

    rows = []
    for spacing in spacings:
        anchors, trim_len = make_dental_arch_anchors(spacing_mm=spacing)
        for body_d in body_diams:
            for foot_name, fh, fe in foot_modes:
                params = replace(
                    base, body_diameter=body_d,
                    foot_height=fh, foot_extra_radius=fe,
                )
                pillars = generate_pillars_to_plane(anchors, 0.0, params, verbose=False)
                raft = make_raft(anchors, z_top=0.0, thickness=1.5,
                                 band_width=3.5, solid_disk=False, verbose=False)
                if pillars is None:
                    continue
                v_pillars = float(pillars.volume)
                v_raft = float(raft.volume) if raft is not None else 0.0
                n_pillars = int(len(anchors))  # (część pomijana jeśli za krótka; tu wszystkie >tip)
                resin_ml = (v_pillars + v_raft) / 1000.0

                # peel proxy
                f_pillar = F_LINE_N_PER_MM * spacing          # N
                a_trunk = math.pi * (body_d / 2.0) ** 2       # mm² (=N/MPa)
                a_tip = math.pi * (base.tip_diameter / 2.0) ** 2
                sigma_trunk = f_pillar / a_trunk              # MPa
                sigma_tip = f_pillar / a_tip

                rows.append(dict(
                    spacing=spacing, body_d=body_d, foot=foot_name,
                    n_pillars=n_pillars, trim_len=round(trim_len, 1),
                    v_pillars_mm3=round(v_pillars, 1),
                    v_raft_mm3=round(v_raft, 1),
                    resin_ml=round(resin_ml, 3),
                    f_pillar_N=round(f_pillar, 4),
                    sigma_trunk_MPa=round(sigma_trunk, 2),
                    sigma_tip_MPa=round(sigma_tip, 2),
                    safe=bool(sigma_trunk <= SIGMA_TRUNK_LIMIT and sigma_tip <= SIGMA_TIP_SOFT),
                ))

    # CSV
    csv_path = OUT_DIR / "support_research.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Baseline (obecne defaulty, spacing 3.0)
    bl = next(r for r in rows
              if r["spacing"] == 3.0 and r["body_d"] == 1.0 and r["foot"] == "foot")

    # --- Konsola: podsumowanie -------------------------------------------------
    print("\n" + "=" * 74)
    print("SUPPORT RESEARCH — baseline + sweep (obecne PROSTE pillary)")
    print("=" * 74)
    print(f"Baseline (defaulty: body_d=1.0, spacing=3.0mm, foot ON):")
    print(f"   pillarów={bl['n_pillars']}  resin={bl['resin_ml']} ml  "
          f"(pillary {bl['v_pillars_mm3']} + raft {bl['v_raft_mm3']} mm³)")
    print(f"   σ_trunk={bl['sigma_trunk_MPa']} MPa  σ_tip={bl['sigma_tip_MPa']} MPa  "
          f"safe={bl['safe']}")

    # Najlżejszy SAFE wariant
    safe_rows = [r for r in rows if r["safe"]]
    if safe_rows:
        best = min(safe_rows, key=lambda r: r["resin_ml"])
        save_pct = 100.0 * (bl["resin_ml"] - best["resin_ml"]) / bl["resin_ml"]
        print(f"\nNajlżejszy SAFE wariant prostych pillarów:")
        print(f"   body_d={best['body_d']} spacing={best['spacing']}mm foot={best['foot']}")
        print(f"   resin={best['resin_ml']} ml  (−{save_pct:.0f}% vs baseline)  "
              f"σ_trunk={best['sigma_trunk_MPa']} σ_tip={best['sigma_tip_MPa']}")
        print(f"   → to jest SUFIT oszczędności bez branchingu. Tree pobije to "
              f"dając gęste tipy (mały σ_tip) + grube trunki (mały σ_trunk) PRZY mniejszej objętości.")

    # --- Wykres Pareto ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))
        cmap = plt.get_cmap("viridis")
        sp_vals = sorted({r["spacing"] for r in rows})
        for r in rows:
            color = cmap((sp_vals.index(r["spacing"])) / max(len(sp_vals) - 1, 1))
            marker = "o" if r["foot"] == "foot" else "^"
            edge = "none" if r["safe"] else "red"
            ax.scatter(r["resin_ml"], r["sigma_trunk_MPa"], c=[color], marker=marker,
                       s=60 + r["body_d"] * 40, edgecolors=edge, linewidths=1.4,
                       alpha=0.85, zorder=3)
        ax.axhline(SIGMA_TRUNK_LIMIT, color="red", ls="--", lw=1,
                   label=f"σ_trunk limit ({SIGMA_TRUNK_LIMIT} MPa)")
        ax.scatter([bl["resin_ml"]], [bl["sigma_trunk_MPa"]], s=320, marker="*",
                   c="gold", edgecolors="black", linewidths=1.5, zorder=5,
                   label="BASELINE (defaulty)")
        ax.set_xlabel("Resin [ml]  (pillary + raft, niżej = mniej zużycia)")
        ax.set_ylabel("σ_trunk [MPa]  (wyżej = ryzyko pęknięcia trunka)")
        ax.set_title("Support Pareto: resin vs wytrzymałość trunka\n"
                     "kolor=spacing (ciemny→gęsty), rozmiar=body_d, ○=foot ▲=noFoot, "
                     "czerwony rant=unsafe")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        # colorbar dla spacing
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(min(sp_vals), max(sp_vals)))
        plt.colorbar(sm, ax=ax, label="spacing kotwic [mm]")
        fig.tight_layout()
        png_path = OUT_DIR / "support_research.png"
        fig.savefig(png_path, dpi=130)
        print(f"\nZapisano:\n   {csv_path}\n   {png_path}")
    except Exception as e:  # pragma: no cover
        print(f"\n[warn] wykres pominięty: {e}\n   CSV: {csv_path}")


if __name__ == "__main__":
    sweep()
