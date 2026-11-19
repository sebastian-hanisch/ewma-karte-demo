"""Orakel-Test: die exakten Lauflängen und die Maße der Karten gegen ANDERE Rechenwege (ORACLE-PLAYBOOK §2, Zeilen „Warteschlangen“ und „Zeitreihen“).

Rechenweg der Demo: Nyström-Verfahren für die Integralgleichung (ewk_chains / ewk_measures). Orakel hier:
  (a) das Gitter nach Brook/Evans (andere Diskretisierung, Fehler ~ 1/m^2) für ARL_0, ARL_1 und alle vier Maße (Eigenvektor, Besuchszahlen, Matrixfolge),
  (b) die geschlossene Form 1/p für λ = 1 (Shewhart-Karte),
  (c) die Simulation mit dem eigenen Zufallsgenerator an den Rohwerten (Band, |z| <= 4; die Simulation benutzt die Ketten der Demo nicht),
  (d) die Reihe Σ_k E_k[(T - k)+] / E_inf[T] für das zyklische Maß, die Eigenwertzerlegung für das quasistationäre Maß,
  (e) Literatur: Paare (λ, L) mit ARL_0 ≈ 500 als eigene Nachrechnung: tests/test_literatur_ewma.py (keine Tabellenzahl der Quelle als Beleg).
Quelle der Zahlen der Handrechnung: handbeispiele_03.py der Planung, hier in tests/test_handbeispiele.py."""

import numpy as np
import pytest
from scipy.special import ndtr

import ewk_chains as K
import ewk_measures as M
import ewk_scenario as S


def _z(sim, se, exact):
    return (sim - exact) / se


# --- (a) Gitter ------------------------------------------------------------------------------------------------------------------------------
def test_integralgleichung_gegen_gitter_zweiseitig_mit_ordnung_zwei():
    lam, H = 0.1, 2.7 * K.sigma_z(0.1)
    for delta, exact, ref in ((0.0, 368.99373398, [362.6496, 367.3496, 368.5763, 368.8887]), (1.0, 9.73001162, [9.738010, 9.732061, 9.730530, 9.730142])):
        assert K.arl_ewma(lam, H, delta, True) == pytest.approx(exact, rel=1e-8)
        grid = [K.ewma_grid_arl(lam, H, delta, True, m) for m in (20, 40, 80, 160)]
        assert grid == pytest.approx(ref, rel=2e-6)
        errs = [abs(g - exact) for g in grid]
        assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1))
        assert all(3.8 <= errs[i] / errs[i + 1] <= 4.0 for i in range(len(errs) - 1))  # Fehler ~ 1/m^2: Faktor 4 je Verdopplung


def test_cusum_nystroem_gegen_gitter():
    h = 4.095448548886181
    assert K.arl_cusum(0.5, h, 0.0) == pytest.approx(370.0, rel=1e-9)
    for delta, exact in ((0.0, 370.0), (1.0, 8.573036251173228)):
        g = [K.cusum_grid_arl(0.5, h, delta, m) for m in (50, 100, 200, 400)]
        errs = [abs(x - exact) for x in g]
        assert all(errs[i + 1] < errs[i] for i in range(3)) and errs[-1] / exact < 2e-5


@pytest.mark.parametrize("lam", [0.1, 0.2])
def test_vier_masse_gegen_gitter_einseitig(lam):
    """zero-state, Pollak, steady-state, zyklisch aus dem Gitter (Eigenvektor, Besuchszahlen, Matrixfolge) gegen die Nyström-Rechnung der Demo."""
    c = M.build_card("ewma", lam, 370.0)
    R0, _, n_dn, _ = K.ewma_grid(lam, c.thr, 0.0, False, 80)
    _, L1, _, _ = K.ewma_grid(lam, c.thr, 1.0, False, 80)
    n = R0.shape[0]
    e = np.zeros(n)
    e[n_dn] = 1.0
    v, D = e.copy(), []
    for _ in range(600):
        D.append(float(v @ L1 / v.sum()))
        v = v @ R0
    ev, V = np.linalg.eig(R0.T)
    i = int(np.argmax(ev.real))
    q = np.abs(V[:, i].real)
    q /= q.sum()
    occ = np.linalg.solve((np.eye(n) - R0).T, e)
    d = c.delays(1.0)
    assert d["zero"] == pytest.approx(L1[n_dn], rel=2e-3)
    assert d["steady"] == pytest.approx(float(q @ L1), rel=2e-3)
    assert d["cyc"] == pytest.approx(float(occ @ L1 / occ.sum()), rel=2e-3)
    assert d["pollak"] == pytest.approx(max(D), rel=2e-3)
    assert c.pre.eig == pytest.approx(float(ev[i].real), abs=5e-6)


# --- Eigenwert und D_nu gegen unabhängige Matrixpotenzen ---------------------------------------------------------------------------------------
def test_quasistationaer_gegen_eigenvektor_und_dnu_gegen_matrixpotenz():
    c = M.build_card("ewma", 0.1, 370.0)
    ch0 = c.pre.ch0
    ev, V = np.linalg.eig(ch0.M.T)
    i = int(np.argmax(ev.real))
    q = np.abs(V[:, i].real)
    q /= q.sum()
    assert c.pre.q == pytest.approx(q, abs=1e-9)
    assert c.pre.eig == pytest.approx(float(ev[i].real), abs=1e-10)
    L1 = K.solve_arl(c.chain(1.0))
    e = np.zeros(len(L1))
    e[ch0.s] = 1.0
    for nu in (0, 1, 2, 5, 7, 10, 50, 200):
        row = e @ np.linalg.matrix_power(ch0.M, nu)
        assert c.delays(1.0)["D"][nu] == pytest.approx(float(row @ L1 / row.sum()), rel=1e-10)
    assert c.pre.surv[10] == pytest.approx(float((e @ np.linalg.matrix_power(ch0.M, 10)).sum()), rel=1e-12)


def test_zyklisch_gegen_shiryaev_reihe():
    for kind, par in (("cusum", 0.5), ("ewma", 0.1)):
        c = M.build_card(kind, par, 370.0)
        L1 = K.solve_arl(c.chain(1.0))
        assert c.pre.delays(L1)["cyc"] == pytest.approx(M.zyklisch_summe(c, L1), abs=1e-7)  # auf 7 Stellen


# --- (b) λ = 1 ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("delta", [0.0, 0.5, 1.0, 2.0])
def test_lambda_eins_alle_masse_gleich_geschlossene_form(delta):
    c = M.build_card("ewma", 1.0, 370.0, True)
    d = c.delays(delta)
    p = 1.0 / K.shewhart_p(c.thr, delta, True)
    for m in ("zero", "pollak", "steady", "cyc"):
        assert d[m] == pytest.approx(p, rel=1e-9), m
    assert d["lorden"] == pytest.approx(p, rel=1e-9)  # gedächtnislos: auch das Supremum über alle Startwerte


def test_shewhart_dreisigma_karte_370_398():
    p = 2 * float(ndtr(-3.0))
    assert 1.0 / p == pytest.approx(370.398347, rel=1e-8)
    assert K.arl_ewma(1.0, 3.0, 0.0, True) == pytest.approx(370.398347, rel=1e-7)
    assert K.arl_ewma(1.0, 3.0, 1.0, True) == pytest.approx(43.894682, rel=1e-7)
    assert K.arl_ewma(1.0, 3.0, 2.0, True) == pytest.approx(6.302963, rel=1e-7)


# --- (c) Simulation an den Rohwerten ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("lam,L,two,delta,seed", [(0.1, 2.7, True, 1.0, 1201), (0.2, 2.86, True, 0.5, 1202), (0.1, 2.4026, False, 1.0, 1203)])
def test_simulation_gegen_exakt_zero_state(lam, L, two, delta, seed):
    H = L * K.sigma_z(lam)
    exact = K.arl_ewma(lam, H, delta, two)
    m, se = S.mean_se(S.run_lengths(S.ewma_stepper(lam, H, two), delta, 3000, seed))
    assert abs(_z(m, se, exact)) <= 4.0, (m, se, exact)


def test_simulation_cusum_gegen_exakt():
    h = M.build_card("cusum", 0.5, 370.0).thr
    exact = K.arl_cusum(0.5, h, 1.0)
    m, se = S.mean_se(S.run_lengths(S.cusum_stepper(0.5, h), 1.0, 4000, 1204))
    assert abs(_z(m, se, exact)) <= 4.0


def test_simulation_dnu_nach_einlauf_gegen_exakt():
    """D_nu mit nu = 150 (ARL_0 = 100, damit die Läufe kurz bleiben): nur Läufe ohne Alarm vor nu zählen, danach Änderung auf delta = 1."""
    c = M.build_card("ewma", 0.1, 100.0)
    nu = 150
    step0 = S.ewma_stepper(0.1, c.thr, False)[1]
    rng = S.SplitMix64(1205)
    out = []
    for _ in range(3000):
        z, alive = 0.0, True
        for _t in range(nu):
            z, alarm = step0(z, S._draw(rng))
            if alarm:
                alive = False
                break
        if not alive:
            continue
        t = 0
        while True:
            t += 1
            z, alarm = step0(z, S._draw(rng) + 1.0)
            if alarm:
                out.append(t)
                break
    m, se = S.mean_se(out)
    assert len(out) > 400
    assert abs(_z(m, se, c.delays(1.0)["D"][nu])) <= 4.0


@pytest.mark.parametrize("kind,par,seed", [("ewma", 0.1, 1206), ("cusum", 0.5, 1207)])
def test_simulation_zyklisch_gegen_exakt(kind, par, seed):
    """Zyklisch: die Karte läuft unter der Nullhypothese mit Neustart nach jedem Alarm bis tstar, ab tstar + 1 gilt delta = 1; gezählt wird bis zum ersten Alarm danach."""
    c = M.build_card(kind, par, 100.0)
    stepper = S.ewma_stepper(par, c.thr, False) if kind == "ewma" else S.cusum_stepper(par, c.thr)
    s0, step = stepper
    rng = S.SplitMix64(seed)
    out = []
    for _ in range(2500):
        z = s0
        for _t in range(300):
            z, alarm = step(z, S._draw(rng))
            if alarm:
                z = s0
        t = 0
        while True:
            t += 1
            z, alarm = step(z, S._draw(rng) + 1.0)
            if alarm:
                out.append(t)
                break
    m, se = S.mean_se(out)
    assert abs(_z(m, se, c.delays(1.0)["cyc"])) <= 4.0, (m, se)
