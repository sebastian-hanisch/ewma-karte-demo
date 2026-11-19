"""Eingefrorene Daten (ewk_cusum2.json, ewk_abgestimmt.json) und das Spielzeug in Brüchen.

Die Dateien erzeugt tools/ewk_make_tuned.py (mit `--check` neu gerechnet und auf 6 Stellen verglichen); je Abschnitt stehen Quelle und Stand.
Das Spielzeug (zwei Zustände, alles in Brüchen) ist ein AUSGEDACHTES Modell: Richtung und Größe der Abstände gelten dort, nicht für die Normal-EWMA."""

import json
import math
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    try:
        return json.loads((HERE / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _key(x: float) -> str:
    return repr(round(float(x), 6))


def besttable(arl0: float, delta: float):
    """„Bestes lambda gegen bestes k“ je Maß für ARL_0 und Sprung (nur für die Sprünge des Rasters, sonst None): {Maß: {lambda, k, ewma, cusum, verhaeltnis}}."""
    try:
        return _load("ewk_abgestimmt.json")["werte"][_key(arl0)][_key(delta)]
    except KeyError:
        return None


def cusum2_frozen(arl0: float, delta: float):
    """Zweiseitiger CUSUM (k = 0,5): (Schwelle h, Verzögerung ab Zelle (0, 0) = zero-state = Lorden) aus der eingefrorenen 2D-Kette; None, wenn nicht vorgerechnet."""
    try:
        row = _load("ewk_cusum2.json")["werte"][_key(arl0)]
        return row["h"], row["delta"][_key(delta)]["verzoegerung"]
    except KeyError:
        return None


def cusum2_h(arl0: float):
    try:
        return _load("ewk_cusum2.json")["werte"][_key(arl0)]["h"]
    except KeyError:
        return None


def cusum2_live(h: float, delta: float, k: float = 0.5) -> float:
    """Verzögerung des zweiseitigen CUSUM ab Zelle (0, 0) live (Richardson aus m = 60 und 100, rund 1 s), für Sprünge außerhalb des eingefrorenen Rasters."""
    import ewk_chains as K

    a1, _ = K.cusum2_grid(k, h, delta, 60)
    a2, _ = K.cusum2_grid(k, h, delta, 100)
    return (100 ** 2 * a2 - 60 ** 2 * a1) / (100 ** 2 - 60 ** 2)


# ----------------------------------------------------------------- Spielzeug-EWMA (lambda = 1/2, Beobachtungen +1 mit Wahrscheinlichkeit q, sonst -1, Alarm bei Z >= 1/2)
def toy_arl(q: F) -> tuple[F, F]:
    """Zwei Zustände: P (Z >= 0: Start oder letzte Beobachtung +1) und N (Z < 0). Aus P alarmiert +1; aus N gibt +1 nur Z in (0, 1/2), also Zustand P.
    x_P = 1 + (1 - q) x_N, x_N = 1 + q x_P + (1 - q) x_N  =>  x_P = 1/q^2, x_N = (1 + q)/q^2."""
    return 1 / (q * q), (1 + q) / (q * q)


def toy_measures(q0: F = F(1, 2), q1: F = F(3, 4)) -> dict:
    """Die fünf Zahlen im Spielzeug: vor der Änderung q0, danach q1. Brüche, wo es welche gibt; steady-state quasistationär (goldener Schnitt bei q0 = 1/2) als Gleitkommazahl."""
    xp0, xn0 = toy_arl(q0)
    xp1, xn1 = toy_arl(q1)
    occ_p, occ_n = 1 / q0, (1 - q0) / q0 ** 2  # erwartete Besuche vor dem Alarm ab P
    cyc = (occ_p * xp1 + occ_n * xn1) / (occ_p + occ_n)
    a = float(1 - q0)
    lam1 = (a + math.sqrt(a * a + 4 * a * float(q0))) / 2  # dominanter Eigenwert von [[0, 1 - q0], [q0, 1 - q0]]
    ratio_np = lam1 / float(q0)  # Gewicht N : P der quasistationären Verteilung
    steady = (float(xp1) + ratio_np * float(xn1)) / (1 + ratio_np)
    vp, vn = F(1), F(0)
    d = []
    for _ in range(60):
        d.append(float((vp * xp1 + vn * xn1) / (vp + vn)))
        vp, vn = vn * q0, vp * (1 - q0) + vn * (1 - q0)
    return {
        "arl0": xp0, "arl0_schlechtester": xn0, "zero": xp1, "zyklisch": cyc, "steady": steady,
        "pollak": max(d), "nu_pollak": d.index(max(d)), "lorden": max(xp1, xn1), "gewicht_NP": ratio_np, "D": d,
    }
