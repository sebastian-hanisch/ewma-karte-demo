"""Die fünf Zahlen der Verzögerung aus EINER exakten Rechnung: zero-state, Pollak, steady-state (quasistationär), zyklisch (Shiryaev), Lorden.

Bezeichnungen (Konvention der Linie, Quelle: Quellenbericht der Linie Nr. 22 bis 24, gelesen; Moustakides u. a. 2009 und Bartroff/Tartakovsky 2024):
  nu = Zahl der Beobachtungen VOR der Änderung (letzte Beobachtung vor der Änderung); die Alarmbeobachtung zählt mit, T - nu >= 1.
  D_nu = E_nu[T - nu | T > nu]: erwartete Beobachtungen bis zum Alarm, wenn die Änderung nach nu Beobachtungen kommt und bis dahin kein Alarm war.
  zero-state      = D_0 (Änderung vor der ersten Beobachtung, Start im Startzustand der Karte).
  Pollak          = sup über nu von D_nu (Pollak 1985, nicht gelesen; hier nach Moustakides u. a. 2009 definiert).
  steady-state    = lim D_nu für nu -> unendlich (quasistationär, so rechnen die Vormessungen V1/V12); „steady-state“ ohne Zusatz heißt immer quasistationär.
  zyklisch        = Summe_k E_k[(T - k)+] / E_inf[T] (Shiryaev-Maß; die Karte startet nach jedem Alarm neu und läuft lange; Lucas/Saccucci 1990, Anhang A,
                    Lesestufe: Quellenbericht „OCR mit Zeichenverlusten“).
  Lorden          = Supremum über nu UND über alle erreichbaren Vorgeschichten (Zustände) der Karte.
Der Anfangsvorsprung ist eine Auslegung (Startwert S_0 bzw. Z_0 ungleich 0), weder Zähleinheit noch Maß.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize_scalar

import ewk_chains as K
import ewk_tuning as T

NU_MAX = 1500  # Horizont der Folge D_nu; (1 - lambda)^1500 ist für lambda >= 0,03 unter 1e-19


class Pre:
    """Alles, was nur von der Kette VOR der Änderung (delta = 0) abhängt: ARL_0, Besuchszahlen (zyklisch), Zustandsverteilungen je nu, quasistationäre Verteilung."""

    def __init__(self, ch0: K.Chain, nu_max: int = NU_MAX):
        M = ch0.M
        n = M.shape[0]
        self.n, self.s, self.ch0 = n, ch0.s, ch0
        e = np.zeros(n)
        e[ch0.s] = 1.0
        lu = lu_factor(np.eye(n) - M)
        self.arl0 = float(lu_solve(lu, np.ones(n))[ch0.s])
        self.occ = lu_solve(lu, e, trans=1)  # Zeile e_s (I - M)^-1: erwartete Besuche je Knoten bis zum Alarm
        V = np.empty((nu_max + 1, n))
        surv = np.empty(nu_max + 1)
        v = e.copy()
        for nu in range(nu_max + 1):
            sm = v.sum()
            surv[nu] = sm
            V[nu] = v / sm
            v = v @ M
        self.V, self.surv = V, surv  # surv[nu] = P(T > nu) vor der Änderung
        q = V[-1]  # nach nu_max Schritten ist die bedingte Verteilung die quasistationäre (Konvergenz geometrisch, siehe NU_MAX)
        self.q = q / q.sum()
        self.eig = float((self.q @ M).sum())  # dominanter Eigenwert = Überlebenswahrscheinlichkeit je Schritt im quasistationären Zustand

    def delays(self, L1: np.ndarray) -> dict:
        """Die vier Maße aus der Restlauflänge L1 je Knoten nach der Änderung. Pollak: Supremum über nu inklusive Grenzwert (Argmax None, wenn er der Grenzwert ist)."""
        D = self.V @ L1
        steady = float(self.q @ L1)
        top = float(D.max())
        pollak = max(top, steady)  # Supremum: größter Wert der Folge oder ihr Grenzwert
        tol = 1e-7 * pollak
        if D[0] >= pollak - tol:
            nu_pollak = 0  # das Maximum liegt bei nu = 0 (dann ist Pollak gleich zero-state)
        elif top > steady + tol:
            nu_pollak = int(np.argmax(D >= top - tol))
        else:
            nu_pollak = None  # das Supremum ist der Grenzwert nu -> unendlich, D_nu erreicht ihn von unten
        return {
            "zero": float(L1[self.s]),
            "pollak": pollak,
            "nu_pollak": nu_pollak,
            "steady": steady,
            "cyc": float(self.occ @ L1 / self.occ.sum()),
            "D": D,
        }


@dataclass
class Card:
    """Eine ausgelegte Karte: Art, Parameter, Schwelle für das Ziel ARL_0, und ihr Vorberechnetes."""

    kind: str  # "ewma" oder "cusum"
    par: float  # lambda bzw. k
    gamma: float  # Ziel-ARL_0
    two: bool
    hs: bool  # Anfangsvorsprung H/2 bzw. h/2 (die Schwelle ist dann neu auf ARL_0 geeicht)
    mult: float  # L (EWMA) bzw. h (CUSUM)
    thr: float  # H = L sigma_Z (EWMA) bzw. h (CUSUM)
    lo: float  # untere Grenze des Zustandsraums (EWMA); 0 beim CUSUM
    start: float
    pre: Pre

    def chain(self, delta: float, u0: float | None = None, n: int | None = None, lo: float | None = None) -> K.Chain:
        if self.kind == "cusum":
            return K.cusum_chain(self.par, self.thr, delta, 60, s0=self.start if u0 is None else u0)
        return K.ewma_chain(self.par, self.thr, delta, self.lo if lo is None else lo, n, u0=self.start if u0 is None else u0)

    def delays(self, delta: float) -> dict:
        ch1 = self.chain(delta)
        L1 = K.solve_arl(ch1)
        d = self.pre.delays(L1)
        if self.kind == "cusum":
            d["lorden"] = float(L1[0])  # einseitiger CUSUM: der Zustand 0 ist der schlechteste (L fällt in S), Lorden = zero-state ohne Vorsprung
        elif self.two:
            d["lorden"] = sup_restlauflaenge(ch1, L1, self.thr)[0]
        else:
            d["lorden"] = None  # einseitige EWMA: im gerechneten Bereich ohne Grenze (siehe lorden_einseitig)
        d["L1"] = L1
        return d


@lru_cache(maxsize=24)
def build_card(kind: str, par: float, gamma: float, two: bool = False, hs: bool = False) -> Card:
    """Legt die Karte auf ARL_0 = gamma aus (Schwelle aus ewk_tuned.json oder live) und rechnet das Vorberechnete. Einseitig: Anstieg."""
    if kind == "cusum":
        if two:
            raise ValueError("zweiseitiger CUSUM: nur als eingefrorene 2D-Kette (ewk_cusum2.json), keine Karte mit Zeitverlauf")
        h = T.cusum_h(par, gamma, hs)
        ch0 = K.cusum_chain(par, h, 0.0, 60, s0=h / 2 if hs else 0.0)
        return Card("cusum", par, gamma, False, hs, h, h, 0.0, h / 2 if hs else 0.0, Pre(ch0))
    mult = T.ewma_L(par, gamma, two, hs)
    thr = mult * K.sigma_z(par)
    lo = -thr if two else K.ewma_lower(par)
    start = thr / 2 if hs else 0.0
    return Card("ewma", par, gamma, two, hs, mult, thr, lo, start, Pre(K.ewma_chain(par, thr, 0.0, lo, u0=start)))


# ----------------------------------------------------------------- Lorden-Maß: Supremum der Restlauflänge über alle Startwerte
def sup_restlauflaenge(ch1: K.Chain, L1: np.ndarray, thr: float, grid: int = 4001) -> tuple[float, float]:
    """Supremum von L(u) = 1 + K(u, .) L über alle Startwerte u in [-thr, thr] der zweiseitigen EWMA (Nyström-Auswertung an beliebigem u):
    Raster mit 4001 Punkten, dann beschränkte Verfeinerung um das Raster-Maximum; die Randwerte werden mitgeprüft. Rückgabe: (Supremum, Ort u).
    Das Supremum liegt NICHT automatisch am Rand: bei kleinem Sprung liegt es im Inneren des Gebiets (Befund D der Linienplanung)."""
    n = len(L1) - 1  # ohne den Startknoten

    def f(u):
        return float(1.0 + ch1.rowfun(u) @ L1[:n])

    us = np.linspace(-thr, thr, grid)
    vals = 1.0 + ch1.rowfun(us) @ L1[:n]
    j = int(np.argmax(vals))
    lo, hi = us[max(j - 1, 0)], us[min(j + 1, len(us) - 1)]
    r = minimize_scalar(lambda u: -f(u), bounds=(lo, hi), method="bounded", options={"xatol": 1e-12})
    return max((f(-thr), -thr), (f(thr), thr), (-float(r.fun), float(r.x)))


def restlauflaenge_kurve(card: Card, delta: float, points: int = 241) -> tuple[np.ndarray, np.ndarray]:
    """L(u) über die Startwerte u in [-H, H] der zweiseitigen EWMA (für die Kurve in Experiment 4)."""
    ch1 = card.chain(delta)
    L1 = K.solve_arl(ch1)
    n = len(L1) - 1
    us = np.linspace(-card.thr, card.thr, points)
    return us, 1.0 + ch1.rowfun(us) @ L1[:n]


def lorden_einseitig(lam: float, gamma: float, delta: float, b_max: float = 30.0, points: int = 61) -> dict:
    """Einseitige EWMA: erwartete Beobachtungen bis zum Alarm ab Startwert u = -b sigma_Z (Restlauflänge L(u)) für b = 0..b_max.
    Das Gebiet wird mit der tiefsten Startposition nach unten verlegt (Untergrenze u - 8 lambda, sonst würde die Rechnung falsch)."""
    card = build_card("ewma", lam, gamma, False, False)
    sz = K.sigma_z(lam)
    lo = min(K.ewma_lower(lam), -b_max * sz - 8.0 * lam)
    ch1 = K.ewma_chain(lam, card.thr, delta, lo, u0=0.0)
    L1 = K.solve_arl(ch1)
    n = len(L1) - 1
    bs = np.linspace(0.0, b_max, points)
    vals = 1.0 + ch1.rowfun(-bs * sz) @ L1[:n]
    return {"b": bs, "L": vals, "H": card.thr, "sigma_z": sz, "n_knoten": n}


def restlauflaenge_ab(lam: float, gamma: float, delta: float, b: float, n: int | None = None) -> float:
    """Einseitige EWMA, Startwert -b sigma_Z, Gebiet bis unter den Startwert (u - 8 lambda); wahlweise mit vorgegebener Knotenzahl (Gitterfeinheit)."""
    card = build_card("ewma", lam, gamma, False, False)
    u = -b * K.sigma_z(lam)
    lo = min(K.ewma_lower(lam), u - 8.0 * lam)
    ch = K.ewma_chain(lam, card.thr, delta, lo, n, u0=u)
    return float(K.solve_arl(ch)[ch.s])


# ----------------------------------------------------------------- Lauflängenverteilung und Richtung des Alarms
def lauflaengen_verteilung(card: Card, delta: float, tmax: int = 1500) -> tuple[np.ndarray, np.ndarray]:
    """P(RL = t) für t = 1..tmax und P(RL > t) für t = 0..tmax (zero-state, Änderung vor der ersten Beobachtung), aus den Iterierten der Kette."""
    ch = card.chain(delta)
    M = ch.M
    v = np.zeros(M.shape[0])
    v[ch.s] = 1.0
    surv = np.empty(tmax + 1)
    for t in range(tmax + 1):
        surv[t] = v.sum()
        v = v @ M
    return -np.diff(surv), surv


def anteil_alarm_nach_oben(lam: float, thr: float, delta: float, u0: float, n: int | None = None) -> float:
    """Zweiseitige EWMA, Start u0: Wahrscheinlichkeit, dass der erste Alarm nach OBEN geht (Z > H); der Rest geht nach unten (Z < -H)."""
    ch = K.ewma_chain(lam, thr, delta, -thr, n, u0=u0)
    m = ch.M.shape[0] - 1
    nodes = ch.nodes[:m]
    b_up = 1.0 - K.PHI((thr - (1 - lam) * nodes) / lam - delta)
    pu = np.linalg.solve(np.eye(m) - ch.M[:m, :m], b_up)
    return float((1.0 - K.PHI((thr - (1 - lam) * u0) / lam - delta)) + ch.rowfun(u0)[:m] @ pu)


def zyklisch_summe(card: Card, L1: np.ndarray, kmax: int = 60000) -> float:
    """Zweiter Weg zum zyklischen Wert: die Reihe Summe_k E_k[(T - k)+] / E_inf[T] (Shiryaev), mit unnormierten Vorwärtsvektoren der Kette vor der Änderung."""
    pre = card.pre
    v = np.zeros(pre.n)
    v[pre.s] = 1.0
    tot = 0.0
    for _ in range(kmax):
        tot += float(v @ L1)
        v = v @ pre.ch0.M
        if v.sum() < 1e-14:
            break
    return tot / pre.arl0
