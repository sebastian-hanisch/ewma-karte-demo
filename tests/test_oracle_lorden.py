"""Orakel-Test für das Lorden-Maß: das Supremum der Restlauflänge, die Richtung der Alarme, der CUSUM-Fall und die einseitige Kurve.

Rechenweg der Demo: Nyström-Auswertung L(u) = 1 + K(u, ·) L an beliebigem u, Supremum aus 4001-Raster plus Verfeinerung (ewk_measures.sup_restlauflaenge).
Orakel: (a) eine eigene, feine Rastersuche über das Gitter nach Brook/Evans (161 Zellen) und über eine eigene Nyström-Gleichung, (b) die Simulation ab dem Startwert
(SplitMix64, Band |z| <= 4), (c) der Kopplungsüberblick des zweiseitigen CUSUM: der schlechteste Zustand der 1600 Zellen ist (0, 0), (d) Richardson-Konsistenz der 2D-Kette."""

import math

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

import ewk_chains as K
import ewk_measures as M
import ewk_reference as R
import ewk_scenario as S

LAM, GAMMA = 0.1, 370.0


def _card2():
    return M.build_card("ewma", LAM, GAMMA, True)


def _own_curve(H, delta, us, N=300):
    """Eigene Nyström-Gleichung (unabhängig von ewk_chains) für die zweiseitige EWMA; L(u) an den Startwerten us."""
    x, w = leggauss(N)
    y, wq = H * x, H * w
    ph = lambda z: np.exp(-0.5 * np.asarray(z) ** 2) / math.sqrt(2 * math.pi)  # noqa: E731
    A = ph((y[None, :] - (1 - LAM) * y[:, None]) / LAM - delta) / LAM * wq[None, :]
    L = np.linalg.solve(np.eye(N) - A, np.ones(N))
    return 1 + (ph((y[None, :] - (1 - LAM) * us[:, None]) / LAM - delta) / LAM * wq[None, :]) @ L


def _grid_cells(H, delta, m=80):
    wd = H / (m + 0.5)
    cent = wd * np.arange(-m, m + 1)
    edges = np.concatenate([[-H], (cent[:-1] + cent[1:]) / 2, [H]])
    Rm = np.diff(ndtr((edges[None, :] - (1 - LAM) * cent[:, None]) / LAM - delta), axis=1)
    return cent, np.linalg.solve(np.eye(2 * m + 1) - Rm, np.ones(2 * m + 1))


@pytest.mark.parametrize("delta,sup_ref,pos_ref", [(0.5, 33.3386, -0.749), (1.0, 13.5565, -0.909), (2.0, 6.6860, -1.0)])
def test_supremum_im_inneren_gegen_raster_und_gitter(delta, sup_ref, pos_ref):
    c = _card2()
    ch = c.chain(delta)
    L1 = K.solve_arl(ch)
    sup, u_sup = M.sup_restlauflaenge(ch, L1, c.thr)
    us = np.linspace(-c.thr, c.thr, 2001)
    Lc = _own_curve(c.thr, delta, us)  # eigener Rechenweg
    assert sup == pytest.approx(sup_ref, rel=6e-4) and u_sup / c.thr == pytest.approx(pos_ref, abs=0.01)
    assert sup == pytest.approx(float(Lc.max()), rel=1e-4)
    assert u_sup / c.thr == pytest.approx(float(us[int(np.argmax(Lc))] / c.thr), abs=0.01)
    cent, Lg = _grid_cells(c.thr, delta)  # Gitter mit 161 Zellen
    assert sup == pytest.approx(float(Lg.max()), rel=2e-3)
    edge = float(1.0 + ch.rowfun(-c.thr) @ L1[:-1])
    if delta < 2.0:
        assert sup > edge + 0.1  # das Supremum liegt ueber dem Randwert L(-H): bei kleinem Sprung ist der Randwert kein Lorden-Wert
    else:
        assert u_sup == pytest.approx(-c.thr, abs=1e-6)  # beim grossen Sprung liegt es am Rand
    assert abs(Lc[0] - Lc[-1]) > 1.0  # Randwerte -H und +H unterscheiden sich stark


def test_zweiseitig_lorden_mindestens_pollak_mindestens_zero():
    c = _card2()
    for delta in (0.25, 0.5, 1.0, 2.0, 3.0):
        d = c.delays(delta)
        assert d["lorden"] >= d["pollak"] - 1e-9 >= d["zero"] - 2e-9


@pytest.mark.parametrize("delta,seed,runs", [(0.5, 2101, 20000), (1.0, 2102, 20000)])
def test_simulation_am_ort_des_supremums(delta, seed, runs):
    c = _card2()
    ch = c.chain(delta)
    L1 = K.solve_arl(ch)
    sup, u_sup = M.sup_restlauflaenge(ch, L1, c.thr)
    m, se = S.mean_se(S.run_lengths(S.ewma_stepper(LAM, c.thr, True, z0=u_sup), delta, runs, seed))
    assert abs((m - sup) / se) <= 4.0, (m, se, sup)


def test_simulation_am_rand_liegt_unter_dem_supremum():
    c = _card2()
    ch = c.chain(1.0)
    L1 = K.solve_arl(ch)
    sup, _ = M.sup_restlauflaenge(ch, L1, c.thr)
    edge = float(1.0 + ch.rowfun(-c.thr) @ L1[:-1])
    m, se = S.mean_se(S.run_lengths(S.ewma_stepper(LAM, c.thr, True, z0=-c.thr), 1.0, 30000, 2103))
    assert abs((m - edge) / se) <= 4.0
    assert m + 3 * se < sup  # der Randwert ist messbar kleiner als das Supremum (13,33 gegen 13,56)


def test_anteil_alarm_nach_unten_exakt_gegen_simulation():
    c = _card2()
    u0 = -0.97 * c.thr
    p_up = M.anteil_alarm_nach_oben(LAM, c.thr, 1.0, u0)
    rng = S.SplitMix64(2104)
    n, up = 40000, 0
    for _ in range(n):
        z = u0
        while True:
            z = LAM * (S._draw(rng) + 1.0) + (1 - LAM) * z
            if z > c.thr or z < -c.thr:
                up += z > 0
                break
    p, se = S.proportion_se(up, n)
    assert abs((p - p_up) / se) <= 4.0
    assert 1.0 - p_up == pytest.approx(0.0418, abs=5e-4)  # 4,18 % der Läufe enden mit einem Alarm nach unten
    assert 1.0 - M.anteil_alarm_nach_oben(LAM, c.thr, 1.0, -c.thr) == pytest.approx(0.0585, abs=5e-4)  # 5,85 % beim Start im Rand


# --- CUSUM: Lorden gleich zero-state (Kopplung) -------------------------------------------------------------------------------------------------
def test_einseitiger_cusum_schlechtester_zustand_ist_null():
    c = M.build_card("cusum", 0.5, GAMMA)
    L1 = K.solve_arl(K.cusum_chain(0.5, c.thr, 1.0))
    assert int(np.argmax(L1)) == 0 and bool(np.all(np.diff(L1[1:]) < 0))  # alle 61 Knoten: L fällt in S


def test_zweiseitiger_cusum_schlechtester_zustand_ist_zelle_null_null():
    h2 = R.cusum2_h(GAMMA)
    a, Lall = K.cusum2_grid(0.5, h2, 1.0, 40)
    assert Lall.shape == (1600,) and int(np.argmax(Lall)) == 0 and a == Lall[0]  # alle 1600 Zellen: (0, 0) ist die schlechteste


def test_cusum2_richardson_und_eingefrorene_werte():
    h2 = R.cusum2_h(GAMMA)
    a60, _ = K.cusum2_grid(0.5, h2, 1.0, 60)
    a100, _ = K.cusum2_grid(0.5, h2, 1.0, 100)
    assert abs(a60 - a100) < 0.003  # m = 60 gegen 100 unterscheiden sich um höchstens 0,003
    frozen = R.cusum2_frozen(GAMMA, 1.0)[1]
    assert frozen == pytest.approx(R.cusum2_live(h2, 1.0), rel=1e-9)  # eingefroren gleich live
    assert frozen == pytest.approx(9.9246, abs=2e-3)
    a0_60, a0_100 = K.cusum2_grid(0.5, h2, 0.0, 60)[0], K.cusum2_grid(0.5, h2, 0.0, 100)[0]
    assert a0_100 == pytest.approx(370.0, rel=1e-3) and a0_60 < a0_100 < 370.0  # das Gitter nähert sich von unten
    assert (100 ** 2 * a0_100 - 60 ** 2 * a0_60) / (100 ** 2 - 60 ** 2) == pytest.approx(370.0, rel=1e-7)  # Richardson: ARL_0 auf das Ziel geeicht


def test_ewma_zweiseitig_gegen_cusum_lorden_verhaeltnis():
    c = _card2()
    for delta, ratio in ((0.5, 0.9457), (1.0, 1.3659), (2.0, 1.7331)):
        lorden = c.delays(delta)["lorden"]
        assert lorden / R.cusum2_frozen(GAMMA, delta)[1] == pytest.approx(ratio, abs=5e-4)


# --- einseitige Kurve ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("b,seed", [(2, 2201), (5, 2202)])
def test_einseitig_startwert_gegen_simulation(b, seed):
    c = M.build_card("ewma", LAM, GAMMA)
    u = -b * K.sigma_z(LAM)
    exact = M.restlauflaenge_ab(LAM, GAMMA, 1.0, b)
    m, se = S.mean_se(S.run_lengths(S.ewma_stepper(LAM, c.thr, False, z0=u), 1.0, 20000, seed))
    assert abs((m - exact) / se) <= 4.0, (m, se, exact)


def test_einseitig_gitter_verdoppelt_und_gebiet_wird_mit_dem_startwert_verlegt():
    base = M.build_card("ewma", LAM, GAMMA)
    for b in (0, 5, 12, 30):
        v = M.restlauflaenge_ab(LAM, GAMMA, 1.0, b)
        n_default = K.ewma_nodes(LAM, base.thr, min(K.ewma_lower(LAM), -b * K.sigma_z(LAM) - 8 * LAM))
        assert v == pytest.approx(M.restlauflaenge_ab(LAM, GAMMA, 1.0, b, n=int(1.5 * n_default)), rel=2e-3)
    # Falle: bleibt die Untergrenze bei -9 sigma_Z, geht für tiefe Startwerte die Masse verloren und die Zahl ist falsch
    u = -30 * K.sigma_z(LAM)
    wrong = K.arl_ewma(LAM, base.thr, 1.0, False, u0=u, lo=K.ewma_lower(LAM))
    assert wrong < 0.5 * M.restlauflaenge_ab(LAM, GAMMA, 1.0, 30)


def test_einseitige_kurve_waechst_streng_und_hat_die_geplanten_werte():
    r = M.lorden_einseitig(LAM, GAMMA, 1.0)
    assert bool(np.all(np.diff(r["L"]) > 0))  # streng wachsend über alle 61 Punkte (b = 0 bis 30), keine Grenze gemessen
    at = {int(b): float(v) for b, v in zip(r["b"], r["L"]) if float(b) in (0.0, 2.0, 5.0, 10.0, 20.0, 30.0)}
    assert at[0] == pytest.approx(8.2999, abs=5e-5) and at[2] == pytest.approx(11.7647, abs=5e-5) and at[5] == pytest.approx(15.3727, abs=5e-5)
    assert at[10] == pytest.approx(19.4050, abs=1e-3) and at[20] == pytest.approx(24.4065, abs=1e-3) and at[30] == pytest.approx(27.6671, abs=5e-5)
