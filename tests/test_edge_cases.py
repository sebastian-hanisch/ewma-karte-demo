"""Randfälle (DEMO-PLAYBOOK §12): λ = 1 und λ klein, δ = 0, sehr großer Sprung, ν = 0 und ν groß, Startwert tief unter −10 σ_Z, unerreichbares oder zu kleines ARL₀,
ein Lauf, L so klein, dass sofort alarmiert wird, leere und entartete Eingaben der Hilfsfunktionen."""

import numpy as np
import pytest

import ewk_chains as K
import ewk_constants as C
import ewk_evaluation as ev
import ewk_measures as M
import ewk_scenario as S
import ewk_tuning as T


def test_lambda_eins_ist_die_shewhart_karte_einseitig_und_zweiseitig():
    for two in (False, True):
        c = M.build_card("ewma", 1.0, 370.0, two)
        assert c.thr == pytest.approx(K.shewhart_c(370.0, two), rel=1e-7) and c.pre.arl0 == pytest.approx(370.0, rel=1e-8)


def test_kleinstes_lambda_bleibt_unter_der_knotengrenze_und_trifft_das_ziel():
    c = M.build_card("ewma", C.LAMBDA_MIN, 1000.0)
    assert c.pre.n <= 601 and c.pre.arl0 == pytest.approx(1000.0, rel=1e-8)  # höchstens 600 Knoten plus der Startknoten
    r = M.lorden_einseitig(C.LAMBDA_MIN, 1000.0, 1.0)
    assert r["n_knoten"] <= 600 and np.all(np.isfinite(r["L"]))  # tiefster Startwert: das Gebiet wird verlegt, die Knotenzahl bleibt begrenzt


def test_delta_null_maße_sind_die_der_kette_vor_der_aenderung():
    c = M.build_card("ewma", 0.1, 370.0)
    d = c.delays(0.0)
    assert d["zero"] == pytest.approx(370.0, rel=1e-8)  # zero-state = ARL_0
    assert d["steady"] == pytest.approx(1.0 / (1.0 - c.pre.eig), rel=1e-8)  # mittlere Restlauflänge der quasistationären Verteilung
    a = ev.exp_card(0.1, False, 370, 0.0)
    assert a["arl1"] == pytest.approx(a["arl0"], rel=1e-12)


def test_sehr_grosser_sprung_gibt_verzoegerung_nahe_eins():
    for lam in (0.1, 1.0):
        d = M.build_card("ewma", lam, 370.0).delays(10.0)
        assert 1.0 <= d["zero"] < 1.01 and 1.0 <= d["steady"] < 1.1  # steady-state: ein Teil der Zustände liegt unter 0 und braucht eine Beobachtung mehr
    assert 1.0 <= M.build_card("cusum", 0.5, 370.0).delays(10.0)["zero"] < 1.01


def test_nu_null_und_sehr_grosses_nu_pollak_suche_endet():
    c = M.build_card("ewma", 0.1, 370.0)
    d = c.delays(1.0)
    assert d["nu_pollak"] == 0 and len(d["D"]) == M.NU_MAX + 1
    d15 = c.delays(1.5)
    assert d15["nu_pollak"] is None and d15["pollak"] == pytest.approx(d15["steady"], rel=1e-12)  # Supremum = Grenzwert, kein endliches Argmax
    assert d15["D"][-1] == pytest.approx(d15["steady"], rel=1e-9)  # ν = 1500: eingeschwungen


def test_unerreichbares_oder_zu_kleines_arl0_gibt_eine_meldung():
    for gamma in (1.0, 1.5, 1e7):
        with pytest.raises(ValueError):
            T.ewma_L_live(0.1, gamma, False)
        with pytest.raises(ValueError):
            T.cusum_h_live(0.5, gamma)
    with pytest.raises(ValueError):
        M.build_card("cusum", 0.5, 370.0, True)  # zweiseitiger CUSUM nur als eingefrorene 2D-Kette


def test_l_so_klein_dass_sofort_alarmiert_wird():
    assert 1.0 <= K.arl_ewma(0.1, 1e-4, 0.0, True) < 1.01  # zweiseitig: |Z| > 1e-4 ist fast immer wahr
    assert 1.0 < K.arl_ewma(0.1, 1e-4, 0.0, False) < 10.0  # einseitig: Z > 1e-4 hat etwa die Wahrscheinlichkeit 1/2, aber die Statistik ist korreliert (ARL_0 etwa 4,8)
    assert 1.0 <= K.arl_cusum(0.5, 0.05, 4.0) < 1.001 and K.arl_cusum(0.5, 0.05, 0.0) == pytest.approx(3.43, abs=0.01)  # P(x >= 0,55) = 0,29


def test_ein_simulationslauf_und_leere_eingabe():
    sim = ev.exp_sim(0.1, False, 370, 1.0, 1, 7, 8.3)
    assert sim["runs"] == 1 and sim["mean"] is not None and sim["mean_se"] == float("inf")
    kind, text = ev.verdict_card(ev.exp_card(0.1, False, 370, 1.0), sim, 1.0)
    assert kind in ("success", "warning") and "Simulation (1 Läufe" in text
    assert S.mean_se([]) == (float("nan"), float("inf")) or np.isnan(S.mean_se([])[0])


def test_simulation_des_mittels_nur_wenn_bezahlbar():
    sim = ev.exp_sim(0.1, False, 370, 0.25, 50, 7, 90.0)
    assert sim["mean"] is not None  # exakte Verzögerung 90 <= 150: wird simuliert
    assert ev.exp_sim(0.1, False, 370, 0.0, 50, 7, 370.0)["mean"] is None  # ARL_0 = 370 je Lauf wäre zu teuer: nur der Anteil bis 100


def test_reihe_mit_aenderung_am_anfang_und_am_ende():
    for nu in (0, 200):
        s = S.make_series(0.2, 0.6, True, 2.0, nu, 250, 3)
        assert len(s.x) == len(s.z) == 250 and s.nu == nu
    s = S.make_series(0.2, 0.6, False, 4.0, 0, 250, 3)
    assert s.alarm is not None and s.alarm <= 5  # Sprung 4 ab der ersten Beobachtung: Alarm nach wenigen Beobachtungen


def test_kreuzungsstelle_alle_keine_und_ab():
    d = [0.1, 0.2, 0.3, 0.4]
    assert ev.crossing(d, [0.9, 0.8, 0.7, 0.6])["art"] == "alle"
    assert ev.crossing(d, [1.1, 1.2, 1.3, 1.4])["art"] == "keine"
    assert ev.crossing(d, [0.9, 0.95, 1.05, 1.1]) == {"art": "ab", "delta": 0.3, "wechsel": 1}
    assert ev.crossing(d, [0.9, 1.1, 0.9, 1.1])["wechsel"] == 3  # mehrere Vorzeichenwechsel werden gezählt


def test_gitter_und_integralgleichung_stimmen_auch_an_den_reglerraendern():
    for lam, gamma, two in ((C.LAMBDA_MIN, 100.0, False), (C.LAMBDA_MIN, 1000.0, True), (1.0, 1000.0, True), (0.5, 100.0, False)):
        a = ev.exp_card(lam, two, gamma, 0.25)
        assert abs(a["grid0"] - a["arl0"]) / a["arl0"] < ev.GRID_TOL and abs(a["grid1"] - a["arl1"]) / a["arl1"] < ev.GRID_TOL, (lam, gamma, two)


def test_zweiseitig_startwert_ausserhalb_der_grenzen_ist_nicht_markiert():
    a = ev.exp_lorden(0.1, True, 370, 1.0, 30)
    assert a["start_inside"] is False and a["start_L"] is None
    assert ev.exp_lorden(0.1, True, 370, 1.0, 2)["start_inside"] is True


def test_zweiseitiger_cusum_fuer_sprung_ausserhalb_des_rasters_wird_live_gerechnet():
    a = ev.exp_lorden(0.1, True, 370, 0.35, 0)
    assert a["cusum2_source"].startswith("live") and 20.0 < a["cusum2"][1] < 80.0
    assert ev.exp_lorden(0.1, True, 370, 1.0, 0)["cusum2_source"] == "eingefroren"
