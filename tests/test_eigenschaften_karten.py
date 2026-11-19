"""Eigenschaften der Karten und Maße über 20 bis 40 Zufallsinstanzen (fester Zufalls-Seed der Instanzwahl, kein Zufall im Test).

Gedächtnislose Karte (λ = 1): alle Maße gleich 1/p. ARL_0 wächst mit L, ARL_1 fällt mit δ, zweiseitig ARL(δ) = ARL(−δ), Skalierung, Pollak >= zero-state und
>= steady-state (gegen eine eigene Folge D_nu), CUSUM: Pollak = zero-state = Lorden, Verfeinerung des Zustandsraums, Verteilung summiert sich zu 1, gleiche Zufalls-Seeds geben gleiche Ausgabe."""

import numpy as np
import pytest

import ewk_chains as K
import ewk_evaluation as ev
import ewk_measures as M
import ewk_scenario as S
from ewk_rng import SplitMix64

RNG = np.random.default_rng(20261118)  # nur zur Wahl der Instanzen; jeder Test zieht davon seine feste Folge
LAMS = RNG.uniform(0.05, 1.0, 40)
LMULT = RNG.uniform(1.5, 3.2, 40)
DELTAS = RNG.uniform(0.2, 3.0, 40)


def test_arl0_waechst_mit_l_einseitig_und_zweiseitig():
    for lam, L in zip(LAMS[:20], LMULT[:20]):
        for two in (False, True):
            a = [K.arl_ewma(float(lam), lm * K.sigma_z(float(lam)), 0.0, two) for lm in (L, L + 0.3, L + 0.6)]
            assert a[0] < a[1] < a[2], (lam, L, two)


def test_arl1_faellt_mit_delta():
    for lam, L in zip(LAMS[:20], LMULT[:20]):
        H = float(L) * K.sigma_z(float(lam))
        a = [K.arl_ewma(float(lam), H, d, False) for d in (0.0, 0.3, 0.8, 1.5, 3.0)]
        assert all(a[i] > a[i + 1] for i in range(4)), (lam, L)


def test_zweiseitig_spiegelsymmetrisch():
    for lam, L, d in zip(LAMS[:30], LMULT[:30], DELTAS[:30]):
        H = float(L) * K.sigma_z(float(lam))
        assert K.arl_ewma(float(lam), H, float(d), True) == pytest.approx(K.arl_ewma(float(lam), H, -float(d), True), rel=1e-9)


def test_skalierung_mit_zweierpotenz_gleiche_lauflaenge():
    """Beobachtung und Grenze mit c = 2 multipliziert: dieselbe Lauflänge (c = 2 ist in Gleitkomma exakt, also Gleichheit Lauf für Lauf)."""
    lam, H, c = 0.2, 0.62, 2.0
    for seed in range(20):
        rng_a, rng_b = SplitMix64(seed), SplitMix64(seed)
        za = zb = 0.0
        ta = tb = None
        for t in range(1, 2000):
            xa = S._draw(rng_a) + 0.5
            za = lam * xa + (1 - lam) * za
            if ta is None and za > H:
                ta = t
            xb = S._draw(rng_b) * c + 0.5 * c
            zb = lam * xb + (1 - lam) * zb
            if tb is None and zb > H * c:
                tb = t
            if ta is not None and tb is not None:
                break
        assert ta == tb and ta is not None


def _eigene_folge(card, delta, nu_max=400):
    """D_nu aus einer eigenen Schleife (Zeilenvektor mal Matrix), unabhängig von ewk_measures.Pre."""
    ch0 = card.pre.ch0
    L1 = K.solve_arl(card.chain(delta))
    v = np.zeros(len(L1))
    v[ch0.s] = 1.0
    out = []
    for _ in range(nu_max + 1):
        out.append(float(v @ L1 / v.sum()))
        v = v @ ch0.M
    return np.array(out)


def test_pollak_mindestens_zero_state_und_steady_state_gegen_eigene_folge():
    for i in range(12):
        lam, delta = float(LAMS[i]), float(DELTAS[i])
        for kind, par in (("ewma", lam), ("cusum", float(np.round(0.1 + 1.9 * LAMS[i], 3)))):
            card = M.build_card(kind, par, 370.0)
            d = card.delays(delta)
            D = _eigene_folge(card, delta)
            assert d["pollak"] >= d["zero"] - 1e-9 and d["pollak"] >= d["steady"] - 1e-9
            assert d["pollak"] >= D.max() - 1e-7 * d["pollak"]  # nicht kleiner als jedes D_nu der eigenen Folge
            assert d["pollak"] == pytest.approx(max(D.max(), d["steady"]), rel=1e-6)  # und nicht größer als Maximum und Grenzwert
            assert D[-1] == pytest.approx(d["steady"], rel=1e-6)  # D_nu hat für nu -> unendlich den quasistationären Grenzwert


def test_cusum_pollak_gleich_zero_state_gleich_lorden():
    for k in (0.25, 0.5, 1.0, 1.5):
        card = M.build_card("cusum", k, 370.0)
        for delta in (0.25, 0.5, 1.0, 2.0, 3.0, 4.0):
            d = card.delays(delta)
            assert d["nu_pollak"] == 0 and d["pollak"] == d["zero"] == pytest.approx(d["lorden"], rel=1e-12)


def test_zweiseitig_lorden_mindestens_pollak_zufallsinstanzen():
    for i in range(10):
        lam, delta = float(LAMS[i]), float(DELTAS[i])
        d = M.build_card("ewma", lam, 370.0, True).delays(delta)
        assert d["lorden"] >= d["pollak"] - 1e-9


def test_zustandsraum_verfeinerung_aendert_die_arl_um_hoechstens_den_gitterfehler():
    for lam, L in zip(LAMS[:12], LMULT[:12]):
        H = float(L) * K.sigma_z(float(lam))
        lo = K.ewma_lower(float(lam))
        n = K.ewma_nodes(float(lam), H, lo)
        a = K.arl_ewma(float(lam), H, 1.0, False, n=n)
        b = K.arl_ewma(float(lam), H, 1.0, False, n=int(1.6 * n))
        assert a == pytest.approx(b, rel=2e-5), (lam, L)
    h = 3.0
    assert K.arl_cusum(0.5, h, 1.0, 60) == pytest.approx(K.arl_cusum(0.5, h, 1.0, 120), rel=1e-5)


def test_verteilung_der_lauflaenge_summiert_sich_zu_eins_und_hat_das_richtige_mittel():
    card = M.build_card("ewma", 0.2, 100.0)
    pmf, surv = M.lauflaengen_verteilung(card, 1.0, 1500)
    assert pmf.sum() + surv[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(pmf >= -1e-15) and np.all(np.diff(surv) <= 1e-15)
    t = np.arange(1, len(pmf) + 1)
    assert float((t * pmf).sum() + 1500 * surv[-1]) == pytest.approx(card.delays(1.0)["zero"], abs=1e-6)  # Mittel der Verteilung gleich zero-state-Verzögerung
    pmf0, surv0 = M.lauflaengen_verteilung(card, 0.0, 1500)
    assert 1.0 - surv0[100] == pytest.approx(1.0 - np.exp(-100 / 100.0), abs=0.03)  # ARL_0 = 100: bis 100 etwa 63 %


def test_abstimmung_trifft_das_ziel():
    for lam in LAMS[:10]:
        for gamma in (100.0, 370.0, 1000.0):
            c = M.build_card("ewma", round(float(lam), 3), gamma)
            assert c.pre.arl0 == pytest.approx(gamma, rel=1e-8)
    for k in (0.3, 0.9):
        assert M.build_card("cusum", k, 370.0).pre.arl0 == pytest.approx(370.0, rel=1e-8)


def test_gleicher_zufalls_seed_gleiche_ausgabe_und_anderer_seed_andere():
    a = S.make_series(0.1, 0.55, False, 1.0, 100, 300, 35)
    assert a == S.make_series(0.1, 0.55, False, 1.0, 100, 300, 35)
    assert a.x != S.make_series(0.1, 0.55, False, 1.0, 100, 300, 36).x
    s1 = ev.exp_sim(0.1, False, 370, 1.0, 300, 35, 8.3)
    assert s1 == ev.exp_sim(0.1, False, 370, 1.0, 300, 35, 8.3)
    assert s1 != ev.exp_sim(0.1, False, 370, 1.0, 300, 36, 8.3)


def test_alle_verzoegerungen_sind_mindestens_eins_und_surv_faellt():
    for lam in (0.03, 0.2, 1.0):
        c = M.build_card("ewma", lam, 370.0)
        d = c.delays(1.5)
        assert d["zero"] >= 1.0 and d["steady"] >= 1.0 and np.all(d["D"] >= 1.0)
        assert np.all(np.diff(c.pre.surv) <= 1e-15) and c.pre.surv[0] == pytest.approx(1.0)
