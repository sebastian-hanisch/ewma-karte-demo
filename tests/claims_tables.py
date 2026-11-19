"""Tabellen der README, aus der Rechnung erzeugt: tests/test_claims.py prüft, dass genau diese Tabellen in der README stehen (Zahl für Zahl).

Aufruf zum Erzeugen des Texts: python tests/claims_tables.py"""

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ewk_constants as C  # noqa: E402
import ewk_evaluation as ev  # noqa: E402
import ewk_measures as M  # noqa: E402
import ewk_reference as R  # noqa: E402

DEL = C.DELTA_GRID


def _md(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def table_vergleich() -> str:
    """Verhältnis EWMA/CUSUM, ARL₀ = 370, einseitig, CUSUM k = 0,5, EWMA λ = 0,1 und 0,2 (feste Auslegung), vier Maße."""
    a = {lam: ev.exp_compare(lam, 0.5, 370, 1.0)["table_ratio"] for lam in (0.1, 0.2)}
    head = ["δ"] + [f"λ = {ev.de(lam, 1)}: {name}" for lam in (0.1, 0.2) for name in ("zero-state", "Pollak", "steady-state", "zyklisch")]
    rows = []
    for i, d in enumerate(DEL):
        rows.append([ev.de(d, 2)] + [ev.de(float(a[lam][m][i]), 4) for lam in (0.1, 0.2) for m in C.MEASURES])
    return _md(head, rows)


def table_kreuzung() -> str:
    head = ["EWMA λ", "zero-state", "Pollak", "steady-state", "zyklisch"]
    rows = []
    for lam in (0.1, 0.2):
        cr = ev.exp_compare(lam, 0.5, 370, 1.0)["crossing"]
        rows.append([ev.de(lam, 1)] + [("auf dem ganzen Gitter schneller" if cr[m]["art"] == "alle" else "ab δ = " + ev.de(cr[m]["delta"], 2)) for m in C.MEASURES])
    return _md(head, rows)


def table_best() -> str:
    head = ["δ", "zero-state", "Pollak", "steady-state", "zyklisch", "bestes λ / k (zero-state)"]
    rows = []
    for d in (0.5, 1.0, 2.0, 3.0):
        t = R.besttable(370, d)
        rows.append([ev.de(d, 1)] + [ev.de(t[m]["verhaeltnis"], 4) for m in C.MEASURES] + [f"{ev.de(t['zero']['lambda'], 2)} / {ev.de(t['zero']['k'], 3)}"])
    return _md(head, rows)


def table_arl0() -> str:
    head = ["ARL₀", "zero-state", "Pollak", "steady-state", "zyklisch"]
    rows = []
    for g in (100, 370, 1000):
        r = ev.exp_compare(0.1, 0.5, g, 1.0)["selected"]
        rows.append([str(g)] + [ev.de(r[m], 4) for m in C.MEASURES])
    return _md(head, rows)


def table_dnu() -> str:
    n = ev.exp_nu(0.1, 0.5, 370, 1.0)[False]
    head = ["ν", "CUSUM k = 0,5", "EWMA λ = 0,1"]
    rows = [[str(v), ev.de(float(n["cusum"]["D"][v]), 4), ev.de(float(n["ewma"]["D"][v]), 4)] for v in (0, 1, 2, 5, 7, 10, 25, 50, 100, 200)]
    rows.append(["Grenzwert ν → ∞", ev.de(n["cusum"]["steady"], 4), ev.de(n["ewma"]["steady"], 4)])
    return _md(head, rows)


def table_pollak_grenzwert() -> str:
    head = ["δ", "zero-state", "Pollak (= Grenzwert ν → ∞)", "steady-state"]
    c = M.build_card("ewma", 0.1, 370.0)
    rows = []
    for d in (1.5, 2.0, 3.0):
        x = c.delays(d)
        rows.append([ev.de(d, 1), ev.de(x["zero"], 4), ev.de(x["pollak"], 4), ev.de(x["steady"], 4)])
    return _md(head, rows)


def table_vorsprung() -> str:
    head = ["δ", "Maß", "CUSUM: mit ÷ ohne Vorsprung", "EWMA λ = 0,1: mit ÷ ohne Vorsprung"]
    rows = []
    for d in (0.5, 1.0, 2.0):
        n = ev.exp_nu(0.1, 0.5, 370, d)
        for m in (*C.MEASURES, "lorden"):
            def ratio(key):
                a0, a1 = n[False][key][m], n[True][key][m]
                return "unbegrenzt (einseitig)" if a0 is None else ev.de(a1 / a0, 4)
            rows.append([ev.de(d, 1), C.MEASURE_NAMES[m], ratio("cusum"), ratio("ewma")])
    return _md(head, rows)


def table_lorden_einseitig() -> str:
    head = ["Startwert", "Beobachtungen bis zum Alarm (λ = 0,1, δ = 1, ARL₀ = 370)"]
    r = M.lorden_einseitig(0.1, 370.0, 1.0)
    at = {int(b): float(v) for b, v in zip(r["b"], r["L"]) if float(b) in (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0)}
    return _md(head, [["0" if b == 0 else f"−{b} σ_Z", ev.de(v, 4)] for b, v in sorted(at.items())])


def table_lorden_zweiseitig() -> str:
    head = ["δ", "EWMA zero-state", "EWMA Lorden (Supremum)", "Ort des Supremums", "Randwert L(−H)", "CUSUM (zero-state = Lorden)", "Verhältnis zero-state", "Verhältnis Lorden"]
    rows = []
    for d in (0.5, 1.0, 2.0):
        a = ev.exp_lorden(0.1, True, 370, d, 0)
        c2 = a["cusum2"][1]
        rows.append([ev.de(d, 1), ev.de(a["zero"], 4), ev.de(a["lorden"], 4), ev.de(a["u_sup"] / a["H"], 3) + " H", ev.de(a["edge"], 4), ev.de(c2, 4),
                     ev.de(a["zero"] / c2, 4), ev.de(a["lorden"] / c2, 4)])
    return _md(head, rows)


def all_tables() -> dict:
    return {
        "vergleich": table_vergleich(), "kreuzung": table_kreuzung(), "best": table_best(), "arl0": table_arl0(), "dnu": table_dnu(),
        "pollak_grenzwert": table_pollak_grenzwert(), "vorsprung": table_vorsprung(), "lorden_einseitig": table_lorden_einseitig(),
        "lorden_zweiseitig": table_lorden_zweiseitig(),
    }


if __name__ == "__main__":
    for name, t in all_tables().items():
        print(f"### {name}\n{t}\n")
