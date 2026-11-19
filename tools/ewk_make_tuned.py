"""Erzeugt die gefrorenen Rechnungen der Demo (und prüft sie neu):

  ewk_tuned.json      Schwellen auf ARL_0 = 100 / 370 / 1000: EWMA (L, einseitig, zweiseitig, mit Vorsprung H/2), CUSUM (h, einseitig, mit Vorsprung h/2)
  ewk_cusum2.json     zweiseitiger CUSUM (k = 0,5): Schwelle h je ARL_0 und Verzögerung ab Zelle (0, 0) je Sprung (2D-Kette, Richardson aus m = 60 und 100)
  ewk_abgestimmt.json „bestes lambda gegen bestes k“ je Maß, Sprung und ARL_0 (beide Karten auf den wahren Sprung abgestimmt, nur im Rückblick möglich)

Aufruf (Demo-Ordner als Arbeitsverzeichnis):
  python tools/ewk_make_tuned.py            schreibt die drei Dateien neu (rund 70 Sekunden)
  python tools/ewk_make_tuned.py --check    rechnet neu und vergleicht mit den Dateien auf 6 Stellen (Exit-Code 1 bei Abweichung), schreibt nichts
  python tools/ewk_make_tuned.py --check --schnell   nur ewk_tuned.json und ewk_abgestimmt.json (ohne 2D-Kette)
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import ewk_chains as K  # noqa: E402
import ewk_constants as C  # noqa: E402
import ewk_measures as M  # noqa: E402
import ewk_tuning as T  # noqa: E402

STAND = "2026-11-18"


def _meta() -> dict:
    return {
        "erzeugt_von": "tools/ewk_make_tuned.py",
        "stand": STAND,
        "umgebung": f"Python {platform.python_version()}, numpy {np.__version__}, scipy {scipy.__version__}, {platform.system()}",
        "hinweis": "Modell: unabhängige Beobachtungen N(delta, 1), sigma = 1 bekannt, Ziel 0. ARL_0 ist das gesetzte Ziel (370, nicht 370,398).",
    }


def _k(x: float) -> str:
    return repr(round(float(x), 6))


def make_tuned() -> dict:
    gam = C.ARL0_CHOICES
    out = {"meta": _meta()}

    def section(quelle, rows):
        return {"quelle": quelle, "stand": STAND, "werte": rows}

    for name, two, hs in (("ewma_ein", False, False), ("ewma_zwei", True, False), ("ewma_ein_vorsprung", False, True)):
        rows = {_k(lam): {_k(g): T.ewma_L_live(lam, g, two, hs) for g in gam} for lam in C.LAMBDA_GRID}
        txt = ("Nullstellensuche des Faktors L (Grenze H = L sigma_Z) auf ARL_0 = Ziel; Integralgleichung (Nyström, ewk_chains.ewma_chain), "
               + ("zweiseitig" if two else "einseitig") + (", Startwert Z_0 = H/2" if hs else ", Startwert 0"))
        out[name] = section(txt, rows)
    for name, hs in (("cusum_ein", False), ("cusum_ein_vorsprung", True)):
        rows = {_k(k): {_k(g): T.cusum_h_live(k, g, hs) for g in gam} for k in C.K_GRID}
        out[name] = section("Nullstellensuche der Schwelle h auf ARL_0 = Ziel; Integralgleichung (Nyström, ewk_chains.cusum_chain), einseitig"
                            + (", Startwert S_0 = h/2" if hs else ", Startwert 0"), rows)
    return out


def _richardson(a1: float, a2: float, m1: int = 60, m2: int = 100) -> float:
    return (m2 ** 2 * a2 - m1 ** 2 * a1) / (m2 ** 2 - m1 ** 2)


def cusum2_arl(k: float, h: float, delta: float) -> tuple[float, float, float]:
    a1, _ = K.cusum2_grid(k, h, delta, 60)
    a2, _ = K.cusum2_grid(k, h, delta, 100)
    return _richardson(a1, a2), a1, a2


def make_cusum2() -> dict:
    k = 0.5
    rows = {}
    for g in C.ARL0_CHOICES:
        h = float(brentq(lambda x: cusum2_arl(k, x, 0.0)[0] - g, 2.0, 9.0, xtol=1e-9))
        per_delta = {}
        for d in C.DELTA_GRID:
            r, a1, a2 = cusum2_arl(k, h, d)
            per_delta[_k(d)] = {"verzoegerung": r, "m60": a1, "m100": a2}
        rows[_k(g)] = {"h": h, "delta": per_delta}
    return {
        "meta": _meta(),
        "quelle": "zweiseitiger CUSUM k = 0,5 als 2D-Kette (ewk_chains.cusum2_grid, Brook/Evans je Seite), Richardson-Extrapolation aus m = 60 und m = 100 (Fehler ~ 1/m^2); "
                  "h so, dass die extrapolierte ARL_0 gleich dem Ziel ist. Verzögerung ab Zelle (0, 0) = zero-state = Lorden (schlechtester Zustand ist (0, 0), "
                  "Kopplungsargument, in tests/test_oracle_lorden.py über alle Zellen geprüft)",
        "stand": STAND,
        "werte": rows,
    }


def make_abgestimmt() -> dict:
    rows = {}
    for g in C.ARL0_CHOICES:
        ec = {lam: M.build_card("ewma", lam, g) for lam in C.LAMBDA_BEST}
        cc = {k: M.build_card("cusum", k, g) for k in C.K_GRID}
        per_delta = {}
        for d in C.DELTA_GRID:
            ed = {lam: ec[lam].delays(d) for lam in C.LAMBDA_BEST}
            cd = {k: cc[k].delays(d) for k in C.K_GRID}
            per_m = {}
            for m in ("zero", "pollak", "steady", "cyc"):
                le = min(C.LAMBDA_BEST, key=lambda x: ed[x][m])
                lc = min(C.K_GRID, key=lambda x: cd[x][m])
                per_m[m] = {"lambda": le, "k": lc, "ewma": ed[le][m], "cusum": cd[lc][m], "verhaeltnis": ed[le][m] / cd[lc][m]}
            per_delta[_k(d)] = per_m
        rows[_k(g)] = per_delta
    return {
        "meta": _meta(),
        "quelle": "je Maß das beste lambda der EWMA (einseitig) gegen das beste k des CUSUM (einseitig), beide auf den wahren Sprung abgestimmt (im Betrieb unbekannt); "
                  "Gitter lambda in LAMBDA_BEST, k in K_GRID (ewk_constants.py); Maße aus ewk_measures.py",
        "stand": STAND,
        "werte": rows,
    }


def _flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("meta", "quelle", "stand"):
                continue
            yield from _flatten(v, f"{prefix}/{k}")
    else:
        yield prefix, x


def _compare(name: str, new: dict, old: dict, rtol: float = 1e-6) -> int:
    a, b = dict(_flatten(new)), dict(_flatten(old))
    bad = 0
    if set(a) != set(b):
        print(f"  {name}: Schlüssel verschieden ({len(set(a) ^ set(b))} Unterschiede)")
        bad += 1
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if abs(x - y) > rtol * max(abs(x), abs(y), 1e-12):
                print(f"  {name}{key}: neu {x!r} gegen Datei {y!r}")
                bad += 1
        elif x != y:
            print(f"  {name}{key}: neu {x!r} gegen Datei {y!r}")
            bad += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="neu rechnen und mit den Dateien vergleichen, nichts schreiben")
    ap.add_argument("--schnell", action="store_true", help="ohne die 2D-Kette (ewk_cusum2.json)")
    args = ap.parse_args()
    jobs = [("ewk_tuned.json", make_tuned), ("ewk_abgestimmt.json", None)]
    if not args.schnell:
        jobs.append(("ewk_cusum2.json", make_cusum2))
    bad = 0
    for name, fn in jobs:
        t0 = time.time()
        path = ROOT / name
        if args.check:
            old = json.loads(path.read_text(encoding="utf-8"))
        if name == "ewk_abgestimmt.json":
            # braucht die Schwellen: bei --check die Datei, beim Schreiben die eben geschriebene
            data = make_abgestimmt()
        else:
            data = fn()
        if args.check:
            n = _compare(name, data, old)
            print(f"{name}: {'GLEICH' if n == 0 else str(n) + ' Abweichungen'} ({time.time() - t0:.0f} s)")
            bad += n
        else:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"{name}: geschrieben ({time.time() - t0:.0f} s)")
            if name == "ewk_tuned.json":
                T.tuned.cache_clear()
                M.build_card.cache_clear()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
