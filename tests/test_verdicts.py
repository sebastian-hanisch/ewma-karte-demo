"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle der Art und jede Zahl im Text steht hier.

Die Verdikte sind reine Funktionen (Art, Text) in ewk_evaluation.py; die App ruft st.<Art>(Text). Die Zeilen unten sind gesetzt, nicht gerechnet: so prüft der Test
die Grenzen (Gleichstand ±0,003, Gitterabweichung 0,5 %, |z| 4) und die Zahlenformate, unabhängig von der Rechnung."""

import numpy as np

import ewk_constants as C
import ewk_evaluation as ev


# --- Format -----------------------------------------------------------------------------------------------------------------------------------
def test_zahlenformat_deutsch():
    assert ev.de(8.3, 2) == "8,30" and ev.de(-1.5, 2) == "−1,50" and ev.de(8.3, 2, sign=True) == "+8,30" and ev.de(0.9536, 4) == "0,9536"
    assert ev.pct(0.6357) == "−36,4 %" and ev.pct(1.0129) == "+1,3 %" and ev.pct(1.0) == "+0,0 %"
    assert ev.sci(2.87e-7) == "2,87·10⁻⁷" and ev.sci(1.0e3) == "1,00·10³"


def test_gleichstand_mit_toleranz_inklusive_der_grenze_von_0_003():
    assert ev.klass(0.9969) == "schneller" and ev.klass(0.9975) == "Gleichstand" and ev.klass(1.0) == "Gleichstand"
    assert ev.klass(1.0025) == "Gleichstand" and ev.klass(1.0031) == "langsamer"
    assert ev.klass(1.0002) == "Gleichstand" and ev.klass(0.9997) == "Gleichstand"  # die zwei Gleichstandszellen der Planung (λ = 0,2, δ = 0,75)
    assert C.TIE == 0.003


# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------
def _card(**kw):
    a = {"lam": 0.1, "two": False, "L": 2.4026, "H": 0.5512, "sigma_z": 0.2294, "arl0": 370.0, "arl1": 8.3, "grid0": 369.8, "grid1": 8.3003,
         "share100": 0.2292, "shewhart0": None, "shewhart1": None}
    a.update(kw)
    return a


def _sim(**kw):
    s = {"runs": 1000, "share": 0.23, "share_se": 0.0133, "mean": 8.2, "mean_se": 0.13}
    s.update(kw)
    return s


def test_verdict_card_nennt_ziel_gitter_und_simulation():
    kind, text = ev.verdict_card(_card(), _sim(), 1.0)
    assert kind == "success"
    for part in ("λ = 0,10, L = 2,4026, einseitig: ARL₀ = 370,00", "(Gitter 369,80, Abweichung 0,05 %)", "im Mittel 8,30 Beobachtungen bis zum Alarm (zero-state; Gitter 8,30)",
                 "22,92 % schon bis Beobachtung 100", "8,20 ± 0,13 gegen exakt 8,30 (z = −0,8)", "23,00 % ± 1,33 (z = +0,1)"):
        assert part in text, part


def test_verdict_card_gitter_ueber_halbem_prozent_ist_ein_fehler_ohne_zahl():
    kind, text = ev.verdict_card(_card(grid0=370.0 * 0.99), _sim(), 1.0)  # 1 % Abweichung
    assert kind == "error" and "keine Zahl wird gemeldet" in text and "1,00 %" in text and "ARL₀ = 370" not in text
    assert ev.verdict_card(_card(grid0=370.0 * 1.0049), _sim(), 1.0)[0] == "success"  # knapp unter 0,5 %
    assert ev.verdict_card(_card(grid1=8.3 * 1.006), _sim(), 1.0)[0] == "error"       # auch ARL_1 wird geprüft


def test_verdict_card_simulation_ausserhalb_von_vier_standardfehlern_ist_eine_warnung():
    assert ev.verdict_card(_card(), _sim(mean=8.3 + 0.13 * 3.9), 1.0)[0] == "success"
    assert ev.verdict_card(_card(), _sim(mean=8.3 + 0.13 * 4.1), 1.0)[0] == "warning"
    assert ev.verdict_card(_card(), _sim(share=0.2292 + 0.0133 * 4.5), 1.0)[0] == "warning"


def test_verdict_card_ohne_sprung_und_ohne_simuliertes_mittel_und_shewhart():
    kind, text = ev.verdict_card(_card(arl1=370.0, grid1=370.0), _sim(mean=None, mean_se=None), 0.0)
    assert kind == "success" and "Ohne Sprung (δ = 0)" in text and "Simulation (" not in text and "Simulierter Anteil" in text
    kind, text = ev.verdict_card(_card(lam=1.0, two=True, shewhart0=370.0, shewhart1=43.86, arl1=43.86, grid1=43.86), _sim(), 1.0)
    assert "zweiseitig" in text and "λ = 1 ist die Shewhart-Karte" in text and "ARL₀ = 370,00, ARL₁ = 43,86" in text


# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------
def _cmp(zero, steady, design="fest", best1=True, cross=True):
    d = np.array(C.DELTA_GRID)
    ratios = {"zero": np.array(zero), "pollak": np.array(zero), "steady": np.array(steady), "cyc": np.array(steady)}
    a = {"design": design, "arl0": 370, "lam": 0.2, "k": 0.5, "table_deltas": d, "table_ratio": ratios,
         "selected": {m: float(ratios[m][3]) for m in C.MEASURES},
         "best1": {"zero": {"verhaeltnis": 0.9536}, "steady": {"verhaeltnis": 1.0170}} if best1 else None,
         "crossing": {"zero": {"art": "alle", "delta": None, "wechsel": 0}, "steady": {"art": "ab", "delta": 0.75, "wechsel": 1}, "pollak": {"art": "alle", "delta": None, "wechsel": 0},
                      "cyc": {"art": "ab", "delta": 0.8, "wechsel": 1}}}
    return a


Z8 = [0.9132, 0.9329, 0.9510, 0.9536, 0.9486, 0.9474, 0.9532, 0.9411]       # λ = 0,2, zero-state
S8 = [0.9256, 0.9629, 1.0002, 1.0170, 1.0264, 1.0297, 1.0291, 1.0299]       # λ = 0,2, steady-state


def test_verdict_compare_zaehlt_je_mass_und_nennt_die_ausnahme():
    kind, text = ev.verdict_compare(_cmp(Z8, S8))
    assert kind == "warning"
    assert "zero-state 8 von 8 schneller (Verhältnis 0,9132 bis 0,9536)" in text
    assert "steady-state (quasistationär) 2 von 8 schneller, 1 Gleichstand, 5 langsamer (Verhältnis 0,9256 bis 1,0299)" in text
    assert "zero-state auf dem ganzen Gitter schneller, steady-state ab δ = 0,75 nicht mehr schneller" in text
    assert "„Bestes λ gegen bestes k“ bei δ = 1 gibt zero-state 0,9536 und steady-state 1,0170" in text
    assert "Die Rangfolge hängt am Maß" in text and "Beim gewählten Sprung: zero-state 0,9536 (schneller)" in text and "steady-state (quasistationär) 1,0170 (langsamer)" in text


def test_verdict_compare_ohne_umkehr_meldet_keine_umkehr():
    kind, text = ev.verdict_compare(_cmp(Z8, [0.95] * 8))
    assert kind == "info" and "Rangfolge hängt am Maß" not in text and "in allen vier Maßen und bei allen geprüften Sprüngen gleich" in text
    kind, text = ev.verdict_compare(_cmp([1.0005] * 8, [1.0] * 8))  # alles Gleichstand: kein Sieger
    assert kind == "info" and "0 von 8 schneller, 8 Gleichstand" in text


def test_verdict_compare_abgestimmt_nennt_orakel_und_keine_kreuzungsstellen():
    a = _cmp([0.88, 0.95, 1.02, 1.02, 1.02, 1.02, 1.02, 1.02], [1.0, 1.02, 1.04, 1.04, 1.04, 1.04, 1.04, 1.04], design="abgestimmt")
    kind, text = ev.verdict_compare(a)
    assert kind == "warning" and "auf den jeweiligen Sprung abgestimmt (Orakel" in text and "im Betrieb unbekannt" in text
    assert "Kreuzungsstelle" not in text and "Beim gewählten Sprung" not in text


# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------
def _nu_side(zero, steady, cyc, pollak, lorden, D):
    return {"zero": zero, "steady": steady, "cyc": cyc, "pollak": pollak, "lorden": lorden, "nu_pollak": 0, "D": np.array(D), "surv": np.ones(len(D)), "thr": 1.0}


def test_form_von_d_nu_und_gleichheit_ab_nu():
    assert ev._shape([3.0, 2.0, 1.5, 1.5]) == "monoton fallend" and ev._shape([1.0, 2.0, 2.0]) == "monoton wachsend"
    assert ev._shape([2.0, 1.0, 1.5]) == "nicht monoton" and ev._shape([2.0, 2.0 + 1e-12, 2.0]) == "konstant"
    assert ev._nu_equal([3.0, 2.0, 1.00004, 1.00001, 1.0], 1.0) == 2 and ev._nu_equal([1.0, 1.0], 1.0) == 0
    assert ev._nu_equal([3.0, 1.0, 3.0, 1.0], 1.0) == 3  # dauerhaft: der letzte Ausreißer zählt


def _nu():
    c_no = _nu_side(8.0, 7.0, 7.1, 8.0, 8.0, [8.0, 7.5, 7.0] + [7.0] * 297)
    e_no = _nu_side(8.0, 8.2, 8.1, 8.2, None, [8.0, 7.5] + [7.8] * 3 + [8.2] * 295)
    c_hs = _nu_side(5.0, 7.07, 7.17, 7.4, 8.08, [5.0, 6.0, 7.07] + [7.07] * 297)
    e_hs = _nu_side(5.6, 8.3, 8.2, 8.3, None, [5.6, 6.0] + [8.3] * 298)
    return {False: {"cusum": c_no, "ewma": e_no}, True: {"cusum": c_hs, "ewma": e_hs}, "lam": 0.1, "k": 0.5, "arl0": 370, "delta": 1.0}


def test_verdict_nu_nennt_form_und_wirkung_des_vorsprungs_je_mass():
    kind, text = ev.verdict_nu(_nu(), False)
    assert kind == "info" and "ohne Anfangsvorsprung" in text
    assert "CUSUM k = 0,50: 8,0000 bei ν = 0, im Grenzwert 7,0000, ab ν = 2 auf 4 Stellen gleich (monoton fallend)" in text
    assert "EWMA λ = 0,10: 8,0000 bei ν = 0, kleinster Wert 7,5000 bei ν = 1 (ν unter 200), im Grenzwert 8,2000 (nicht monoton)" in text
    assert "zero-state −37,5 % / −30,0 %" in text  # 5,0 / 8,0 und 5,6 / 8,0
    assert "Pollak −7,5 % / +1,2 %" in text and "steady-state +1,0 % / +1,2 %" in text and "Lorden (CUSUM) +1,0 %" in text
    assert "Wirkung hängt am Maß" in text
    kind, text = ev.verdict_nu(_nu(), True)
    assert "mit Anfangsvorsprung" in text and "CUSUM k = 0,50: 5,0000 bei ν = 0" in text


def test_verdict_nu_gleichgerichtete_wirkung():
    a = _nu()
    a[True] = {"cusum": dict(a[False]["cusum"], zero=7.0, pollak=7.0, steady=6.0, cyc=6.0, lorden=7.0), "ewma": dict(a[False]["ewma"], zero=7.0, pollak=7.0, steady=7.0, cyc=7.0)}
    assert "in allen gezeigten Maßen in dieselbe Richtung" in ev.verdict_nu(a, False)[1]


# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------
def _lor_one(curve=None):
    b = np.linspace(0, 30, 61)
    L = np.array(curve) if curve is not None else 8.3 + 0.6 * b
    return {"two": False, "lam": 0.1, "arl0": 370, "delta": 1.0, "b": 5, "curve_b": b, "curve_L": L, "H": 0.55, "sigma_z": 0.23,
            "at": {0: 8.2999, 5: 15.3727, 10: 19.404, 30: 27.6671}, "selected": 15.3727, "zero": 8.3, "steady": 8.25, "cusum_lorden": 8.573, "cusum_h": 4.09}


def test_verdict_lorden_einseitig_waechst_oder_ist_konstant_oder_nicht_streng():
    kind, text = ev.verdict_lorden(_lor_one())
    assert kind == "info" and "ab Startwert 0 sind es 8,30 Beobachtungen, ab −5 σ_Z 15,37, ab −10 σ_Z 19,40, ab −30 σ_Z 27,67" in text
    assert "streng, zwischen b = 10 und 30 um etwa 8,3" in text and "nicht bewiesen" in text and "Gewählt: Startwert −5 σ_Z gibt 15,37" in text
    assert "Lorden = zero-state = 8,57" in text and "2,87·10⁻⁷" in text
    assert "Die Kurve ist konstant" in ev.verdict_lorden(_lor_one(np.full(61, 5.0)))[1]
    nicht_streng = np.concatenate([np.linspace(8, 20, 30), np.linspace(20, 18, 31)])
    t = ev.verdict_lorden(_lor_one(nicht_streng))[1]
    assert "nicht überall streng" in t and "nicht bewiesen" not in t


def _lor_two(**kw):
    a = {"two": True, "lam": 0.1, "arl0": 370, "delta": 1.0, "b": 0, "H": 0.62, "L": 2.7, "lorden": 13.5565, "u_sup": -0.909 * 0.62, "edge": 13.3296, "zero": 9.7354,
         "pollak": 9.7354, "steady": 9.5292, "cyc": 9.5317, "up_share_edge": 1 - 0.0585, "cusum2": (4.77, 9.9247), "cusum2_source": "eingefroren",
         "start": 0.0, "start_inside": True, "start_L": 9.7}
    a.update(kw)
    return a


def test_verdict_lorden_zweiseitig_supremum_rand_richtung_und_umkehr():
    kind, text = ev.verdict_lorden(_lor_two())
    assert kind == "info" and "Lorden-Maß 13,56 (Supremum der Restlauflänge L(u) bei u = −0,91 H), am Rand −H sind es 13,33; im zero-state-Maß 9,74" in text
    assert "enden 5,85 % der Läufe mit einem Alarm nach unten" in text
    assert "zero-state = Lorden = 9,92; Verhältnis EWMA/CUSUM zero-state 0,9809, Lorden 1,3659" in text
    assert "Das Urteil kehrt sich zwischen den Maßen um" in text
    _, t = ev.verdict_lorden(_lor_two(zero=11.0, cusum2=(4.77, 9.9247)))  # zero-state der EWMA schon hinten: keine Umkehr
    assert "kehrt sich" not in t
    _, t = ev.verdict_lorden(_lor_two(cusum2=None, up_share_edge=1.0))
    assert "Zweiseitiger CUSUM" not in t and "Alarm nach unten" not in t


def test_verdict_lorden_zweiseitig_ohne_gedaechtnis():
    _, t = ev.verdict_lorden(_lor_two(lorden=43.86, zero=43.86, edge=43.86))
    assert "kein Gedächtnis" in t and "Supremum der Restlauflänge" not in t and "keine Garantie" not in t


# --- Kernfragen -------------------------------------------------------------------------------------------------------------------------------
def test_kernfragen_je_zeile_eine_frage_mit_zahlen():
    s = ev.Settings()
    card = _card()
    cmp = _cmp(Z8, S8)
    cmp["selected"] = {"zero": 0.9681, "pollak": 0.9681, "steady": 1.0443, "cyc": 1.0432}
    lines = ev.kernfragen(s, card, cmp, _nu(), _lor_one())
    assert len(lines) == 4 and [ln[:5] for ln in lines] == ["**1. ", "**2. ", "**3. ", "**4. "]
    assert "exakt 370,00" in lines[0] and "zero-state 0,9681 (schneller)" in lines[1] and "steady-state 1,0443 (langsamer)" in lines[1]
    assert "EWMA 8,0000 bei ν = 0" in lines[2] and "ab −30 σ_Z 27,67" in lines[3] and "ohne Grenze" in lines[3]
    abgestimmt = ev.kernfragen(s, card, _cmp(Z8, S8, design="abgestimmt"), _nu(), _lor_two())
    assert "Beide auf den Sprung abgestimmt" in abgestimmt[1] and "Lorden-Maß 13,56 gegen zero-state 9,74" in abgestimmt[3]
