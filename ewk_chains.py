"""Ketten der Karten: exakte Lauflängen aus Integralgleichungen (Nyström, Gauss-Legendre) und Gitter (Brook/Evans).

Modell überall: unabhängige Beobachtungen x_t ~ N(delta, 1), Ziel 0, sigma = 1 bekannt. Eine Karte ist eine Markov-Kette auf ihrem Zustand
(CUSUM: S, EWMA: Z); die Lauflänge ist die Zahl der Beobachtungen bis zum Alarm, die Alarmbeobachtung zählt mit. Alarm = fehlende Masse
der substochastischen Matrix M (Zeile = von, Spalte = nach).

Die Rechenwege sind aus den Vormessungen V1 (`ae_v1_lib.py`) und V12 (`ae_v12_ewma_vergleich.py`) der Linie abgeschrieben, nicht importiert:
  * Nyström-Verfahren für die Integralgleichung (Page 1954 für den CUSUM, Crowder 1987 für die EWMA; beide nicht gelesen, die Gleichungen
    sind aus der Kettenüberlegung selbst hergeleitet und gegen das Gitter und die geschlossene Form geprüft, siehe tests/test_oracle_*.py),
  * Gitter nach Brook/Evans (1972, nicht gelesen) als zweiter, unabhängiger Rechenweg und Zweitentscheider in der App.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # kleine dichte Löser: BLAS-Threads machen sie 100-mal langsamer (DEMO-PLAYBOOK, Gotchas)

from dataclasses import dataclass  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402

PHI = ndtr


def phi(x):
    """Dichte der Standardnormalverteilung."""
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / np.sqrt(2.0 * np.pi)


def gl(a: float, b: float, n: int):
    """Gauss-Legendre-Knoten und -Gewichte auf [a, b]."""
    x, w = leggauss(n)
    return 0.5 * (b - a) * (x + 1) + a, 0.5 * (b - a) * w


@dataclass
class Chain:
    """Substochastische Matrix M auf Knoten, Startknoten s; rowfun(u) liefert die Zeile für einen beliebigen Zustand u (Nyström-Auswertung)."""

    M: np.ndarray
    s: int
    rowfun: Callable
    nodes: np.ndarray


def _augment(M: np.ndarray, start_row: np.ndarray):
    """Hängt einen Startknoten an, der keine eingehende Masse hat (für Startwerte außerhalb der Gauss-Knoten)."""
    n = M.shape[0]
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = M
    A[n, :n] = start_row
    return A, n


def sigma_z(lam: float) -> float:
    """Grenzwert der Standardabweichung der EWMA-Statistik: sqrt(lambda / (2 - lambda))."""
    return float(np.sqrt(lam / (2.0 - lam)))


def cusum_chain(k: float, h: float, delta: float, n: int = 60, s0: float = 0.0) -> Chain:
    """Einseitiger CUSUM S_t = max(0, S_(t-1) + x_t - k), Alarm bei S_t >= h. Knoten: der Zustand 0 (mit Masse) und n Gauss-Knoten auf (0, h)."""
    y, wq = gl(0.0, h, n)
    u = np.concatenate([[0.0], y])

    def rowfun(uu):
        uu = np.atleast_1d(np.asarray(uu, dtype=float))
        R = np.zeros((len(uu), n + 1))
        R[:, 0] = PHI(k - uu - delta)
        R[:, 1:] = phi(y[None, :] - uu[:, None] + k - delta) * wq[None, :]
        return R if len(uu) > 1 else R[0]

    M = rowfun(u)
    if s0 == 0.0:
        return Chain(M, 0, rowfun, u)
    A, s = _augment(M, rowfun(s0))
    return Chain(A, s, rowfun, np.concatenate([u, [s0]]))


def ewma_nodes(lam: float, h: float, lo: float) -> int:
    """Knotenzahl: Der Kern ist eine Gauss-Glocke der Breite lambda; ab etwa fünf Knoten je Breite genügt es (Test: Verdopplung ändert nichts)."""
    return int(min(600, max(80, 5.0 * (h - lo) / lam)))


def ewma_chain(lam: float, h: float, delta: float, lo: float, n: int | None = None, u0: float = 0.0) -> Chain:
    """EWMA Z_t = lambda x_t + (1 - lambda) Z_(t-1) auf (lo, h): Alarm bei Z_t > h; die Masse unterhalb von lo geht verloren (zweiseitig: Alarm, einseitig: vernachlässigbar)."""
    n = n or ewma_nodes(lam, h, lo)
    y, wq = gl(lo, h, n)

    def rowfun(uu):
        uu = np.atleast_1d(np.asarray(uu, dtype=float))
        R = phi((y[None, :] - (1 - lam) * uu[:, None]) / lam - delta) / lam * wq[None, :]
        return R if len(uu) > 1 else R[0]

    M = rowfun(y)
    A, s = _augment(M, rowfun(u0))
    return Chain(A, s, rowfun, np.concatenate([y, [u0]]))


def ewma_lower(lam: float, b: float = 9.0) -> float:
    """Untergrenze des Zustandsraums der einseitigen EWMA: -b Standardabweichungen der Statistik (b = 9: die Masse darunter ist unter 1e-18)."""
    return -b * sigma_z(lam)


def solve_arl(ch: Chain) -> np.ndarray:
    """Erwartete Lauflänge ab jedem Knoten: (I - M) L = 1."""
    n = ch.M.shape[0]
    return np.linalg.solve(np.eye(n) - ch.M, np.ones(n))


def arl_cusum(k: float, h: float, delta: float, n: int = 60, s0: float = 0.0) -> float:
    ch = cusum_chain(k, h, delta, n, s0)
    return float(solve_arl(ch)[ch.s])


def arl_ewma(lam: float, h: float, delta: float, two: bool, n: int | None = None, u0: float = 0.0, lo: float | None = None) -> float:
    lo = lo if lo is not None else (-h if two else ewma_lower(lam))
    ch = ewma_chain(lam, h, delta, lo, n, u0)
    return float(solve_arl(ch)[ch.s])


# ----------------------------------------------------------------- Shewhart (lambda = 1), geschlossen
def shewhart_p(c: float, delta: float, two: bool = True) -> float:
    """Alarmwahrscheinlichkeit je Beobachtung der Shewhart-Karte mit Grenze c (in Standardabweichungen)."""
    p = float(PHI(delta - c))
    return p + float(PHI(-c - delta)) if two else p


def shewhart_c(gamma: float, two: bool = True) -> float:
    p = 1.0 / gamma
    return float(ndtri(1.0 - (p / 2.0 if two else p)))


# ----------------------------------------------------------------- Gitter (Brook/Evans), zweiter Rechenweg
def cusum_grid_arl(k: float, h: float, delta: float, m: int) -> float:
    """Brook/Evans: Zustände 0..m-1, Zellbreite w = h / (m - 1/2); Zelle 0 = [0, w/2), Zelle j = [(j - 1/2) w, (j + 1/2) w). Rückgabe: ARL ab Zelle 0."""
    w = h / (m - 0.5)
    v = w * np.arange(m)
    upper = (np.arange(m) + 0.5) * w
    C = PHI(upper[None, :] - v[:, None] + k - delta)
    R = np.diff(C, axis=1, prepend=0.0)
    return float(np.linalg.solve(np.eye(m) - R, np.ones(m))[0])


def ewma_grid(lam: float, h: float, delta: float, two: bool, m_up: int, b: float = 9.0):
    """Brook/Evans-Gitter der EWMA: Zellmitten j w, j = -n_dn..m_up, w = h / (m_up + 1/2); Start in der Zelle mit Mitte 0. Rückgabe: (R, L, Index der Startzelle, Mitten)."""
    w = h / (m_up + 0.5)
    n_dn = m_up if two else int(np.ceil(b * sigma_z(lam) / w))
    cent = w * np.arange(-n_dn, m_up + 1)
    edges = np.concatenate([[-np.inf], (cent[:-1] + cent[1:]) / 2, [np.inf]])
    edges[-1] = h
    edges[0] = -h if two else -np.inf
    cdf = PHI((edges[None, :] - (1 - lam) * cent[:, None]) / lam - delta)
    R = np.diff(cdf, axis=1)
    L = np.linalg.solve(np.eye(len(cent)) - R, np.ones(len(cent)))
    return R, L, n_dn, cent


def ewma_grid_arl(lam: float, h: float, delta: float, two: bool, m_up: int = 80) -> float:
    _, L, n_dn, _ = ewma_grid(lam, h, delta, two, m_up)
    return float(L[n_dn])


# ----------------------------------------------------------------- zweiseitiger CUSUM (2D-Kette)
def cusum2_grid(k: float, h: float, delta: float, m: int):
    """Zweiseitiger CUSUM (S+, S-) als Kette auf m x m Zellen (Brook/Evans je Seite). Beide Statistiken hängen vom selben x ab: je Zustand wird die
    x-Achse an allen Zellgrenzen zerlegt. Rückgabe: ARL ab Zelle (0, 0) und die ARL aller m*m Zellen (Zeile i * m + j = Zelle (i, j))."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl

    w = h / (m - 0.5)
    v = w * np.arange(m)
    q = np.arange(1, m + 1)
    rows, cols, data = [], [], []
    for i in range(m):
        tplus = (q - 0.5) * w - v[i] + k  # S+' >= (q - 1/2) w  <=>  x >= tplus_q (q = m: Alarm)
        for j in range(m):
            s_q = v[j] - k - (q - 0.5) * w  # S-' >= (q - 1/2) w  <=>  x <= s_q (fallend in q)
            B = np.sort(np.concatenate([tplus, s_q]))
            lo_e = np.concatenate([[-np.inf], B])
            hi_e = np.concatenate([B, [np.inf]])
            mid = np.where(np.isfinite(lo_e) & np.isfinite(hi_e), 0.5 * (lo_e + hi_e), np.where(np.isfinite(lo_e), lo_e + 1.0, hi_e - 1.0))
            pr = PHI(hi_e - delta) - PHI(lo_e - delta)
            ip = np.searchsorted(tplus, mid, side="right")
            jp = m - np.searchsorted(s_q[::-1], mid, side="left")
            ok = (ip < m) & (jp < m) & (pr > 0)
            rows.append(np.full(int(ok.sum()), i * m + j))
            cols.append(ip[ok] * m + jp[ok])
            data.append(pr[ok])
    R = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(m * m, m * m))
    L = spl.spsolve(sp.identity(m * m, format="csc") - R.tocsc(), np.ones(m * m))
    return float(L[0]), L
