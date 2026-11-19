"""Abstimmung der Schwellen auf das Ziel ARL_0: gefrorene Werte (ewk_tuned.json) für das Raster der Voreinstellungen, sonst live per Nullstellensuche.

Die Datei ewk_tuned.json erzeugt tools/ewk_make_tuned.py (`--check` rechnet neu und vergleicht auf 6 Stellen). Je Abschnitt stehen Quelle und Stand.
Live rechnet eine Nullstellensuche je Karte 10 bis 15 Lösungen einer Matrix mit höchstens 600 Knoten (Messung: README, Abschnitt Rechenzeiten).
ARL_0 ist hier immer das GESETZTE Ziel (370 gegen 370,398 der 3-Sigma-Karte: ein anderes Ziel gehört zu anderen Schwellen)."""

import json
from functools import lru_cache
from pathlib import Path

from scipy.optimize import brentq

import ewk_chains as K

TUNED_PATH = Path(__file__).with_name("ewk_tuned.json")
GAMMA_MIN = 2.0          # darunter ist die Karte keine Karte mehr (ARL_0 <= 1: jede Beobachtung alarmiert)
GAMMA_MAX = 1.0e6        # darüber ist die Nullstellensuche nicht mehr verlässlich (Gitter und Knotenzahl sind dafür nicht ausgelegt)


@lru_cache(maxsize=1)
def tuned() -> dict:
    try:
        return json.loads(TUNED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _key(x: float) -> str:
    return repr(round(float(x), 6))


def _lookup(section: str, par: float, gamma: float):
    try:
        return tuned()[section]["werte"][_key(par)][_key(gamma)]
    except KeyError:
        return None


def _check_gamma(gamma: float) -> None:
    if not (GAMMA_MIN <= gamma <= GAMMA_MAX):
        raise ValueError(f"Ziel-ARL_0 = {gamma} liegt außerhalb von {GAMMA_MIN} bis {GAMMA_MAX:.0e}: keine Schwelle abstimmbar")


def _root(f, lo: float, hi: float, tol: float) -> float:
    flo, fhi = f(lo), f(hi)
    if not (flo < 0.0 < fhi):
        raise ValueError(f"Ziel-ARL_0 im Bereich der Suche nicht erreichbar (ARL_0 {flo:+.4g} bis {fhi:+.4g} gegen das Ziel)")
    return float(brentq(f, lo, hi, xtol=tol))


def ewma_L_live(lam: float, gamma: float, two: bool, hs: bool = False) -> float:
    """Faktor L der Grenze H = L sigma_Z mit ARL_0 = gamma (Start 0 bzw. H/2), einseitig oder zweiseitig; Nullstellensuche, ohne Tabelle."""
    _check_gamma(gamma)
    sz = K.sigma_z(lam)

    def f(L):
        H = L * sz
        lo = -H if two else K.ewma_lower(lam)
        return K.arl_ewma(lam, H, 0.0, two, lo=lo, u0=H / 2 if hs else 0.0) - gamma

    return _root(f, 0.3, 6.5, 1e-10)


def ewma_L(lam: float, gamma: float, two: bool = False, hs: bool = False) -> float:
    section = "ewma_" + ("zwei" if two else "ein") + ("_vorsprung" if hs else "")
    v = _lookup(section, lam, gamma)
    return float(v) if v is not None else ewma_L_live(lam, gamma, two, hs)


def cusum_h_live(k: float, gamma: float, hs: bool = False) -> float:
    """Schwelle h des einseitigen CUSUM mit ARL_0 = gamma (Start 0 bzw. h/2); Nullstellensuche, ohne Tabelle."""
    _check_gamma(gamma)

    def f(h):
        return K.arl_cusum(k, h, 0.0, 60, s0=h / 2 if hs else 0.0) - gamma

    hi = 0.5
    while hi < 80.0 and f(hi) < 0.0:
        hi *= 1.5
    return _root(f, 0.05, hi, 1e-11)


def cusum_h(k: float, gamma: float, hs: bool = False) -> float:
    v = _lookup("cusum_ein" + ("_vorsprung" if hs else ""), k, gamma)
    return float(v) if v is not None else cusum_h_live(k, gamma, hs)
