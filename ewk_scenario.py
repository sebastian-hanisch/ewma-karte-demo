"""Reihen und Simulation: ausgedachte Normalreihen mit bekanntem Sprung (Vehikel A), erzeugt mit dem eigenen Zufallsgenerator (SplitMix64).

x_t ~ N(0, 1) für t <= nu, danach N(delta, 1), sigma = 1 bekannt. Es gibt keine reale Datenquelle. Die gezeigte Reihe nimmt `normal()` (Box-Muller,
plattformfest, rund 100 µs je Zahl); die Wiederholungssimulation nimmt `random()` mit der Umkehrfunktion der Normalverteilung (rund 1 µs, nur als Band
plattformfest: `NormalDist.inv_cdf` benutzt die Rechenfunktionen des Systems). Die Simulation ist eine Gegenprobe der exakten Werte, nicht deren Quelle."""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from ewk_rng import SplitMix64

_INV = NormalDist().inv_cdf
_EPS = 2.0 ** -54  # random() liegt in [0, 1); die Umkehrfunktion braucht (0, 1)


def _draw(rng: SplitMix64) -> float:
    u = rng.random()
    return _INV(u if u > 0.0 else _EPS)


@dataclass(frozen=True)
class Series:
    x: tuple[float, ...]
    z: tuple[float, ...]       # EWMA-Statistik Z_t (Z_0 = 0)
    alarm: int | None          # erste Beobachtung t (1-basiert) mit Alarm, sonst None
    nu: int                    # Beobachtungen vor der Änderung


def make_series(lam: float, thr: float, two: bool, delta: float, nu: int, nobs: int, seed: int) -> Series:
    """Eine Reihe der Länge nobs und der EWMA-Verlauf; `alarm` ist die erste Beobachtung mit Z > H (einseitig) bzw. |Z| > H (zweiseitig)."""
    rng = SplitMix64(seed)
    x, z, zt, alarm = [], [], 0.0, None
    for t in range(1, nobs + 1):
        xt = rng.normal() + (delta if t > nu else 0.0)
        zt = lam * xt + (1.0 - lam) * zt
        x.append(xt)
        z.append(zt)
        if alarm is None and ((abs(zt) > thr) if two else (zt > thr)):
            alarm = t
    return Series(tuple(x), tuple(z), alarm, nu)


def ewma_stepper(lam: float, thr: float, two: bool, z0: float = 0.0):
    """(Startzustand, Schrittfunktion) der EWMA; Schritt: (Zustand, x) -> (neuer Zustand, Alarm)."""
    if two:
        def step(z, x):
            z = lam * x + (1.0 - lam) * z
            return z, (z > thr or z < -thr)
    else:
        def step(z, x):
            z = lam * x + (1.0 - lam) * z
            return z, z > thr
    return z0, step


def cusum_stepper(k: float, h: float, s0: float = 0.0):
    def step(s, x):
        s = s + x - k
        if s < 0.0:
            s = 0.0
        return s, s >= h
    return s0, step


def run_lengths(stepper, delta: float, runs: int, seed: int, cap: int = 100000) -> list[int]:
    """Lauflängen (Beobachtungen bis zum Alarm, die Alarmbeobachtung zählt mit) ab dem Startzustand; Änderung vor der ersten Beobachtung.
    Ein Lauf ohne Alarm nach `cap` Beobachtungen zählt mit `cap` (der Aufrufer entscheidet vorher, ob die Simulation bezahlbar ist)."""
    state0, step = stepper
    rng = SplitMix64(seed)
    out = []
    for _ in range(runs):
        s, t = state0, 0
        while t < cap:
            t += 1
            s, alarm = step(s, _draw(rng) + delta)
            if alarm:
                break
        out.append(t)
    return out


def alarms_within(stepper, delta: float, runs: int, seed: int, horizon: int) -> int:
    """Zahl der Läufe mit Alarm innerhalb der ersten `horizon` Beobachtungen (zero-state)."""
    state0, step = stepper
    rng = SplitMix64(seed)
    n = 0
    for _ in range(runs):
        s = state0
        for _t in range(horizon):
            s, alarm = step(s, _draw(rng) + delta)
            if alarm:
                n += 1
                break
    return n


def mean_se(values) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()) if len(a) else float("nan"), float("inf")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def proportion_se(k: int, n: int) -> tuple[float, float]:
    p = k / n
    return p, float(np.sqrt(max(p * (1.0 - p), 1e-12) / n))
