"""Handbeispiele der Planung (handbeispiele_03.py), als exakte Tests: Spielzeug-EWMA in Brüchen, die fünf Zahlen im Spielzeug, Normal-EWMA (Varianzformel, 5-Zellen-Gitter, λ = 1).

Das Orakel des Spielzeugs ist die AUFZÄHLUNG aller 2^t Folgen mit der echten EWMA-Rekursion in Brüchen (kein Code der Demo); die Kette und die Zahlen der Demo
(`ewk_reference.toy_*`) werden dagegen geprüft. Brüche und Ganzzahlen exakt, Gleitkomma mit Toleranz."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

import ewk_chains as K
import ewk_reference as R

LAM, HTHR = F(1, 2), F(1, 2)


def enum_rl_dist(q: F, tmax: int) -> dict:
    """P(Lauflänge = t) für t = 1..tmax durch Aufzählen aller 2^t Folgen (Beobachtung +1 mit q, sonst -1) mit der EWMA-Rekursion in Brüchen."""
    dist = {t: F(0) for t in range(1, tmax + 1)}
    for t in range(1, tmax + 1):
        for xs in itertools.product((1, -1), repeat=t):
            z, alarm_at = F(0), None
            for i, x in enumerate(xs, start=1):
                z = LAM * x + (1 - LAM) * z
                if z >= HTHR:
                    alarm_at = i
                    break
            if alarm_at == t:
                p = F(1)
                for x in xs:
                    p *= q if x == 1 else 1 - q
                dist[t] += p
    return dist


def chain_rl_dist(q: F, tmax: int) -> dict:
    """Dieselbe Verteilung aus der Zwei-Zustands-Kette P / N (Rekursion über die Wahrscheinlichkeitsvektoren)."""
    vp, vn, out = F(1), F(0), {}
    for t in range(1, tmax + 1):
        out[t] = vp * q
        vp, vn = vn * q, vp * (1 - q) + vn * (1 - q)
    return out


@pytest.mark.parametrize("q", [F(1, 2), F(3, 4)])
def test_aufzaehlung_gleich_kette(q):
    assert enum_rl_dist(q, 11) == chain_rl_dist(q, 11)


def test_erste_wahrscheinlichkeiten_und_fibonacci():
    c = chain_rl_dist(F(1, 2), 30)
    assert [c[t] for t in range(1, 9)] == [F(1, 2), F(0), F(1, 8), F(1, 16), F(1, 16), F(3, 64), F(5, 128), F(1, 32)]
    fib = [0, 1, 1]
    for _ in range(40):
        fib.append(fib[-1] + fib[-2])
    assert all(c[t] == F(fib[t - 2], 2 ** t) for t in range(3, 31))  # P(RL = t) = Fib(t - 2) / 2^t


def test_arl_von_hand():
    assert R.toy_arl(F(1, 2)) == (F(4), F(6))
    assert R.toy_arl(F(3, 4)) == (F(16, 9), F(28, 9))
    c = chain_rl_dist(F(1, 2), 400)
    assert abs(float(sum(c.values())) - 1.0) < 1e-12
    assert abs(float(sum(t * p for t, p in c.items())) - 4.0) < 1e-9  # Mittel der Verteilung gleich ARL0 = 4


def test_die_fuenf_zahlen_im_spielzeug():
    t = R.toy_measures()
    assert t["arl0"] == F(4) and t["arl0_schlechtester"] == F(6)
    assert t["zero"] == F(16, 9) and t["zyklisch"] == F(22, 9) and t["lorden"] == F(28, 9)
    assert t["nu_pollak"] == 1 and t["pollak"] == pytest.approx(28 / 9, rel=1e-15)  # Pollak = Lorden, Maximum bei nu = 1
    assert t["D"][0] == pytest.approx(16 / 9) and t["D"][1] == pytest.approx(28 / 9) and t["D"][2] == pytest.approx(22 / 9)
    assert t["gewicht_NP"] == pytest.approx((1 + math.sqrt(5)) / 2, rel=1e-12)  # goldener Schnitt
    assert t["steady"] == pytest.approx(2.6018, abs=5e-5)
    assert float(t["zero"]) < float(t["zyklisch"]) < t["steady"] < float(t["lorden"])


def test_zyklisch_gleich_shiryaev_summe_im_spielzeug():
    """Zweiter Weg: Summe_k E_k[(T - k)+] / E_inf[T] mit Gleitkomma-Vorwärtsvektoren der Kette vor der Änderung."""
    vp, vn, tot = 1.0, 0.0, 0.0
    for _ in range(300):
        tot += vp * 16 / 9 + vn * 28 / 9
        vp, vn = vn * 0.5, vp * 0.5 + vn * 0.5
    assert tot / 4.0 == pytest.approx(22 / 9, abs=1e-9)


def test_derselbe_detektor_als_cusum():
    """Reflektierter Irrweg (Schritte ±1, k = 0, Schwelle 2): CUSUM von 0 gleich schlechtester EWMA-Zustand, CUSUM von 1 gleich EWMA zero-state (nur Spielzeug)."""
    q = F(3, 4)
    L0 = (1 + q) / (q * q)
    L1 = 1 + (1 - q) * L0
    assert L0 == 1 + (1 - q) * L0 + q * L1
    assert (L0, L1) == (F(28, 9), F(16, 9))


# --- Normal-EWMA ----------------------------------------------------------------------------------------------------------------------------
def test_varianzformel_in_bruechen():
    lam = F(1, 2)
    for t in range(1, 7):
        w = [lam * (1 - lam) ** j for j in range(t)]
        assert sum(x * x for x in w) == lam / (2 - lam) * (1 - (1 - lam) ** (2 * t))
    assert F(1, 2) / (2 - F(1, 2)) == F(1, 3) and F(1) / (2 - F(1)) == 1


def _nystroem(lam, H, delta, lo, N=200, u0=0.0):
    """Eigene, kurze Integralgleichung (unabhängig von ewk_chains)."""
    x, w = leggauss(N)
    y = 0.5 * (H - lo) * (x + 1) + lo
    wq = 0.5 * (H - lo) * w
    ph = lambda z: np.exp(-0.5 * np.asarray(z) ** 2) / math.sqrt(2 * math.pi)  # noqa: E731
    A = ph((y[None, :] - (1 - lam) * y[:, None]) / lam - delta) / lam * wq[None, :]
    L = np.linalg.solve(np.eye(N) - A, np.ones(N))
    return float(1 + (ph((y - (1 - lam) * u0) / lam - delta) / lam * wq * L).sum())


@pytest.mark.parametrize("d", [0.0, 1.0, 2.0])
def test_lambda_eins_gleich_shewhart_geschlossen(d):
    p = ndtr(d - 3) + ndtr(-3 - d)
    assert _nystroem(1.0, 3.0, d, -3.0, 100) == pytest.approx(1 / p, rel=1e-6)
    assert K.arl_ewma(1.0, 3.0, d, True) == pytest.approx(1 / p, rel=1e-6)  # auch die Kette der Demo
    assert 1 / K.shewhart_p(3.0, d, True) == pytest.approx(1 / p, rel=1e-12)


def _grid(lam, H, m):
    w = H / (m + 0.5)
    cent = w * np.arange(-m, m + 1)
    edges = np.concatenate([[-H], (cent[:-1] + cent[1:]) / 2, [H]])
    R_ = np.diff(ndtr((edges[None, :] - (1 - lam) * cent[:, None]) / lam), axis=1)
    return R_, float(np.linalg.solve(np.eye(2 * m + 1) - R_, np.ones(2 * m + 1))[m])


def test_gitter_mit_fuenf_zellen_und_konvergenz():
    lam = 0.5
    H = 3.0 * math.sqrt(lam / (2 - lam))
    exact = _nystroem(lam, H, 0.0, -H)
    R5, arl5 = _grid(lam, H, 2)
    assert R5.shape == (5, 5) and np.all(R5 >= 0) and np.all(R5.sum(axis=1) < 1.0)  # Zeilensummen unter 1: der Rest ist der Alarm
    assert exact == pytest.approx(397.461, abs=0.005)
    assert arl5 == pytest.approx(338.58, abs=0.01) and abs(arl5 - exact) / exact < 0.15
    errs = [abs(_grid(lam, H, m)[1] - exact) for m in (2, 5, 10, 20, 40, 80)]
    assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
    assert errs[-2] / errs[-1] == pytest.approx(4.0, abs=0.3)  # Fehler ~ 1/m^2
    assert K.arl_ewma(lam, H, 0.0, True) == pytest.approx(exact, rel=1e-8)  # die Kette der Demo gleich der eigenen Gleichung


def test_symmetrie_und_literaturpaar_und_monotonie():
    s = math.sqrt(0.2 / 1.8)
    H = 2.86 * s
    assert _nystroem(0.2, H, 0.7, -H) == pytest.approx(_nystroem(0.2, H, -0.7, -H), abs=1e-7)  # ARL(delta) = ARL(-delta)
    s = math.sqrt(0.25 / 1.75)
    assert _nystroem(0.25, 2.998 * s, 0.0, -2.998 * s) == pytest.approx(499.84, abs=0.1)
    s = math.sqrt(0.1 / 1.9)
    vals = [_nystroem(0.1, 2.7 * s, d, -2.7 * s) for d in (0.0, 0.5, 1.0, 2.0)]
    assert all(vals[i] > vals[i + 1] for i in range(3))
