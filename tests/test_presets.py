"""Schnellstarts: Vollständigkeit, gültige Werte, Standard gleich Standardeinstellungen, und je Schnellstart die erwartete Verdikt-Zeile
(DEMO-PLAYBOOK §11a Punkt 7: ein Schnellstart ohne Wirkung meldet keine Wirkung, „Gleichstand“ meldet keinen Sieger)."""

import pytest

import ewk_constants as C
import ewk_evaluation as ev
import ewk_presets as P


def _settings(name: str) -> ev.Settings:
    p = C.PRESETS[name]
    return ev.Settings(**{k: v for k, v in p.items()})


def _all(s: ev.Settings):
    card = ev.exp_card(s.lam, s.two, s.arl0, s.delta)
    sim = ev.exp_sim(s.lam, s.two, s.arl0, s.delta, s.runs, s.seed, card["arl1"])
    cmp = ev.exp_compare(s.lam, s.k, s.arl0, s.delta) if s.design == "fest" else ev.exp_compare_tuned(s.arl0, s.delta)
    nu = ev.exp_nu(s.lam, s.k, s.arl0, s.delta)
    lor = ev.exp_lorden(s.lam, s.two, s.arl0, s.delta, s.b)
    return {"card": ev.verdict_card(card, sim, s.delta)[1], "cmp": ev.verdict_compare(cmp)[1], "nu": ev.verdict_nu(nu, s.vorsprung == "mit")[1], "lorden": ev.verdict_lorden(lor)[1],
            "kern": " ".join(ev.kernfragen(s, card, cmp, nu, lor)), "kind_cmp": ev.verdict_compare(cmp)[0], "kind_card": ev.verdict_card(card, sim, s.delta)[0]}


def test_jeder_schnellstart_hat_hilfetext_und_nur_bekannte_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_schnellstart_werte_liegen_in_den_reglergrenzen_und_auf_dem_raster():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)
                if spec.step:
                    assert round((value - spec.lo) / spec.step, 6) == pytest.approx(round((value - spec.lo) / spec.step), abs=1e-6), (name, key)  # auf dem Reglerraster


def test_standard_schnellstart_ist_die_standardeinstellung():
    first = next(iter(C.PRESETS))
    assert C.PRESETS[first] == {} and _settings(first) == ev.Settings()


def test_setting_specs_decken_jeden_schluessel_der_schnellstarts_und_jede_einstellung_ab():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == set(ev.Settings.__dataclass_fields__)  # jede Einstellung hat einen Regler (Schlüssel gleich Feldname)
    for key, state_key in P.PRESET_KEYS.items():
        spec = P.SETTING_SPECS[state_key]
        assert spec.lo is not None and spec.hi is not None
        if spec.choices is not None:
            assert spec.default in spec.choices and spec.lo == spec.choices[0] and spec.hi == spec.choices[-1]
        else:
            assert spec.lo <= spec.default <= spec.hi


def test_schnellstart_zahlen_in_den_hilfetexten_stimmen_mit_der_rechnung():
    a = _all(_settings("EWMA λ = 0,1 gegen CUSUM (δ = 1)"))
    assert "8,5730" in a["nu"] and "7,9038" in a["nu"]
    nu = ev.exp_nu(0.1, 0.5, 370, 1.0)[False]
    assert ev.de(nu["ewma"]["D"][0], 2) == "8,30"  # Lorden-Schnellstart: „von 8,30 auf 15,37“
    assert ev.de(ev.exp_lorden(0.1, False, 370, 1.0, 5)["selected"], 2) == "15,37"
    assert "8,30" in C.PRESET_HELP["Lorden: Startwert −5 σ_Z"] and "15,37" in C.PRESET_HELP["Lorden: Startwert −5 σ_Z"]


# --- je Schnellstart die erwartete Verdikt-Zeile -----------------------------------------------------------------------------------------------
def test_schnellstart_lambda_0_1_umkehr_zwischen_zero_state_und_steady_state():
    a = _all(_settings("EWMA λ = 0,1 gegen CUSUM (δ = 1)"))
    assert a["kind_cmp"] == "warning" and "zero-state 4 von 8 schneller, 4 langsamer" in a["cmp"] and "steady-state (quasistationär) 3 von 8 schneller, 5 langsamer" in a["cmp"]
    assert "zero-state ab δ = 1,15 nicht mehr schneller, steady-state ab δ = 0,90" in a["cmp"]
    assert "zero-state 0,9681 (schneller)" in a["cmp"] and "steady-state (quasistationär) 1,0443 (langsamer)" in a["cmp"]
    assert "8,5730" in a["nu"] and "8,1598 bei ν = 7" in a["nu"] and "8,2540" in a["nu"] and "nicht monoton" in a["nu"]
    assert "ab −30 σ_Z 27,67" in a["lorden"]


def test_schnellstart_lambda_0_2_zero_state_gewinnt_steady_state_nur_bis_0_5():
    a = _all(_settings("EWMA λ = 0,2 gegen CUSUM (δ = 1)"))
    assert "zero-state 8 von 8 schneller (Verhältnis 0,9132 bis 0,9536)" in a["cmp"]
    assert "steady-state (quasistationär) 2 von 8 schneller, 1 Gleichstand, 5 langsamer (Verhältnis 0,9256 bis 1,0299)" in a["cmp"]
    assert "zero-state auf dem ganzen Gitter schneller, steady-state ab δ = 0,75 nicht mehr schneller" in a["cmp"]


def test_schnellstart_gleichstand_meldet_keinen_sieger():
    a = _all(_settings("Gleichstand im steady-state (λ = 0,2, δ = 0,75)"))
    assert "steady-state (quasistationär) 1,0002 (Gleichstand)" in a["cmp"] and "zyklisch (Shiryaev) 0,9997 (Gleichstand)" in a["cmp"]
    assert "zero-state 0,9510 (schneller)" in a["cmp"]
    sel = a["cmp"].split("Beim gewählten Sprung:")[1].split("„Bestes")[0]
    assert "steady-state (quasistationär) 1,0002 (schneller)" not in sel and "steady-state (quasistationär) 1,0002 (langsamer)" not in sel
    assert "steady-state (quasistationär) 1,0002 (Gleichstand)" in a["kern"] or "steady-state 1,0002 (Gleichstand)" in a["kern"]


def test_schnellstart_kleiner_sprung_alle_vier_masse_schneller():
    a = _all(_settings("Kleiner Sprung δ = 0,5"))
    for part in ("zero-state 0,7857 (schneller)", "Pollak 0,7857 (schneller)", "steady-state (quasistationär) 0,8120 (schneller)", "zyklisch (Shiryaev) 0,8115 (schneller)"):
        assert part in a["cmp"], part
    assert "21,74" in a["card"]  # ARL_1 bei δ = 0,5, λ = 0,1


def test_schnellstart_arl0_100_hat_groesseren_abstand_zwischen_den_massen():
    s = _settings("ARL₀ = 100")
    a = _all(s)
    assert "ARL₀ = 100,00" in a["card"] and "zero-state 0,9260 (schneller)" in a["cmp"] and "steady-state (quasistationär) 1,0662 (langsamer)" in a["cmp"]
    r100, r370 = ev.exp_compare(0.1, 0.5, 100, 1.0)["selected"], ev.exp_compare(0.1, 0.5, 370, 1.0)["selected"]
    assert r100["steady"] - r100["zero"] > r370["steady"] - r370["zero"] > 0  # 0,140 gegen 0,076


def test_schnellstart_mit_vorsprung_senkt_zero_state_und_erhoeht_steady_state():
    a = _all(_settings("Mit Vorsprung"))
    assert "mit Anfangsvorsprung" in a["nu"] and "zero-state −36,4 % / −33,4 %" in a["nu"] and "steady-state +1,3 % / +1,7 %" in a["nu"]
    assert "Pollak −6,6 % / +1,1 %" in a["nu"] and "5,4501 bei ν = 0" in a["nu"]


def test_schnellstart_lorden_startwert_minus_fuenf():
    a = _all(_settings("Lorden: Startwert −5 σ_Z"))
    assert "Gewählt: Startwert −5 σ_Z gibt 15,37 Beobachtungen" in a["lorden"] and "ab Startwert 0 sind es 8,30" in a["lorden"]


def test_schnellstart_lambda_eins_alle_masse_gleich_und_kein_lorden_wachstum():
    s = _settings("λ = 1 ist Shewhart")
    a = _all(s)
    assert "zweiseitig" in a["card"] and "geschlossenen Form 1/p" in a["card"] and "ARL₀ = 370,00" in a["card"]
    assert "kein Gedächtnis" in a["lorden"] and "Supremum der Restlauflänge" not in a["lorden"]
    assert "konstant" in a["nu"]  # D_nu der Shewhart-Karte ist konstant
    cmp = ev.exp_compare(1.0, 0.5, 370, 1.0)
    for m in C.MEASURES:  # λ = 1: alle vier Maße haben dieselbe Verzögerung 26,75 (einseitig)
        assert cmp["selected_values"][m][0] == pytest.approx(cmp["selected_values"]["zero"][0], rel=1e-9)
    one = ev.exp_lorden(1.0, False, 370, 1.0, 5)
    assert "konstant" in ev.verdict_lorden(one)[1] and "ohne Grenze" not in ev.verdict_lorden(one)[1]
