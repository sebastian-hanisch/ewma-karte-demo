"""Claims: JEDE Zahl der README wird hier nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die großen Tabellen der README werden aus der Rechnung erzeugt (tests/claims_tables.py) und müssen Zeichen für Zeichen in der README stehen; die Planwerte wichtiger Zellen stehen zusätzlich
fest im Test (unabhängig von der Demo-Rechnung, aus Plan und Vormessung, auf vier Stellen). Die Spalte „Art“ der Tabelle „Zahl → Test“ in der README nennt: exakt (Rechnung, relative Toleranz
1e-6 für Gleitkomma) oder Band (Simulation, |z| <= 4). Zahlen, die nur eine Messung der README sind (Rechenzeiten), stehen nicht hier."""

import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

import claims_tables as CT
import ewk_chains as K
import ewk_constants as C
import ewk_evaluation as ev
import ewk_measures as M
import ewk_reference as R
import ewk_scenario as S
import ewk_texts as TX
import ewk_tuning as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
TABLES = CT.all_tables()


def _a(x, y, tol=5e-5):
    return abs(x - y) <= tol


# --- Tabellen: Text gleich Rechnung ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(TABLES))
def test_claim_tabelle_steht_zeichen_fuer_zeichen_in_der_readme(name):
    assert TABLES[name] in README, name


def test_claim_tabelle_vergleich():
    """Planwerte (V12 C, Plan Tabelle B), unabhängig von der Demo: Verhältnis EWMA/CUSUM, vier Maße, acht Sprünge, λ = 0,1 und 0,2."""
    plan = {
        0.1: {"zero": [0.7207, 0.7857, 0.8916, 0.9681, 1.0505, 1.0906, 1.1153, 1.1440], "pollak": [0.7207, 0.7857, 0.8916, 0.9681, 1.0528, 1.0983, 1.1319, 1.1440],
              "steady": [0.7266, 0.8120, 0.9449, 1.0443, 1.1529, 1.2035, 1.2362, 1.2429], "cyc": [0.7264, 0.8115, 0.9440, 1.0432, 1.1514, 1.2018, 1.2345, 1.2414]},
        0.2: {"zero": [0.9132, 0.9329, 0.9510, 0.9536, 0.9486, 0.9474, 0.9532, 0.9411], "pollak": [0.9132, 0.9329, 0.9510, 0.9536, 0.9486, 0.9474, 0.9532, 0.9411],
              "steady": [0.9256, 0.9629, 1.0002, 1.0170, 1.0264, 1.0297, 1.0291, 1.0299], "cyc": [0.9254, 0.9626, 0.9997, 1.0164, 1.0256, 1.0290, 1.0284, 1.0291]},
    }
    for lam, rows in plan.items():
        got = ev.exp_compare(lam, 0.5, 370, 1.0)["table_ratio"]
        for m, ref in rows.items():
            assert np.allclose(got[m], ref, atol=5.1e-5), (lam, m)
    # zero-state und Pollak sind bei λ = 0,2 in allen acht δ gleich, bei λ = 0,1 nur bei δ = 1,5, 2 und 3 verschieden
    g1, g2 = ev.exp_compare(0.1, 0.5, 370, 1.0)["table_ratio"], ev.exp_compare(0.2, 0.5, 370, 1.0)["table_ratio"]
    assert np.allclose(g2["zero"], g2["pollak"], atol=1e-12)
    assert [bool(abs(z - p) > 1e-6) for z, p in zip(g1["zero"], g1["pollak"])] == [False, False, False, False, True, True, True, False]
    # beide Definitionen des steady-state geben in allen 16 Zellen dasselbe Urteil (Gleichstand ±0,003); ohne Toleranz unterscheiden sie sich nur bei λ = 0,2, δ = 0,75
    strikt = [(lam, d) for lam, g in ((0.1, g1), (0.2, g2)) for d, s, c in zip(C.DELTA_GRID, g["steady"], g["cyc"]) if (s < 1) != (c < 1)]
    assert strikt == [(0.2, 0.75)]
    assert all(ev.klass(s) == ev.klass(c) for g in (g1, g2) for s, c in zip(g["steady"], g["cyc"]))


def test_claim_kreuzungsstellen():
    c1, c2 = ev.exp_compare(0.1, 0.5, 370, 1.0)["crossing"], ev.exp_compare(0.2, 0.5, 370, 1.0)["crossing"]
    assert (c1["zero"]["delta"], c1["pollak"]["delta"], c1["steady"]["delta"], c1["cyc"]["delta"]) == (1.15, 1.15, 0.90, 0.90)
    assert c2["zero"]["art"] == c2["pollak"]["art"] == "alle" and c2["steady"]["delta"] == 0.75 and c2["cyc"]["delta"] == 0.80
    assert all(c["wechsel"] == 1 for c in (c1["zero"], c1["steady"], c2["steady"]))  # je ein Vorzeichenwechsel des Gitters
    # zwei „ab“-Zellen liegen im Gleichstandsband ±0,003 (README, Fußnote unter Befund 2)
    for lam, m, d, ref in ((0.1, "zero", 1.15, 1.0003), (0.2, "steady", 0.75, 1.0002)):
        r = ev.exp_compare(lam, 0.5, 370, 1.0)["ratio"][m]
        i = int(np.argmin(np.abs(np.asarray(C.DELTA_CURVE) - d)))
        assert round(float(r[i]), 4) == ref and abs(ref - 1.0) <= C.TIE, (lam, m, d)
    assert "Zellen mit Verhältnis innerhalb von ±0,003 um 1 sind Gleichstand" in README


def test_claim_bestes_gegen_bestes():
    ref = {0.5: (0.8824, 0.8866, 1.0029, 0.9971), 1.0: (0.9536, 0.9536, 1.0170, 1.0164), 2.0: (1.0196, 1.0196, 1.0415, 1.0414), 3.0: (1.0201, 1.0224, 1.0312, 1.0311)}
    for d, row in ref.items():
        t = R.besttable(370, d)
        assert [round(t[m]["verhaeltnis"], 4) for m in C.MEASURES] == list(row), d
    # unabhängig nachgerechnet für δ = 1: bestes λ gegen bestes k über das Raster, direkt aus ewk_measures
    e = {lam: M.build_card("ewma", lam, 370.0).delays(1.0) for lam in C.LAMBDA_BEST}
    c = {k: M.build_card("cusum", k, 370.0).delays(1.0) for k in C.K_GRID}
    for m, v in zip(C.MEASURES, ref[1.0]):
        assert min(x[m] for x in e.values()) / min(x[m] for x in c.values()) == pytest.approx(v, abs=5.1e-5)
    # feste Auslegung λ = 0,2 gegen k = 0,5 gewinnt den zero-state-Vergleich bis δ = 4
    assert bool(np.all(ev.exp_compare(0.2, 0.5, 370, 1.0)["table_ratio"]["zero"] < 1.0))


def test_claim_abhaengigkeit_von_arl0():
    z = {g: ev.exp_compare(0.1, 0.5, g, 1.0)["selected"] for g in (100, 370, 1000)}
    ref = {100: (0.9260, 0.9738, 1.0662, 1.0564), 370: (0.9681, 0.9681, 1.0443, 1.0432), 1000: (0.9792, 0.9792, 1.0367, 1.0364)}
    for g, row in ref.items():
        assert [round(z[g][m], 4) for m in C.MEASURES] == list(row), g
    gaps = [z[g]["steady"] - z[g]["zero"] for g in (100, 370, 1000)]
    assert gaps == pytest.approx([0.140, 0.076, 0.058], abs=6e-4) and gaps[0] > gaps[1] > gaps[2] > 0
    assert all(z[g]["zero"] < 1.0 < z[g]["steady"] for g in z)  # die Richtung der Umkehr bleibt


def test_claim_d_nu():
    n = ev.exp_nu(0.1, 0.5, 370, 1.0)[False]
    c, e = n["cusum"]["D"], n["ewma"]["D"]
    assert [round(float(c[i]), 4) for i in (0, 1, 2, 5, 10, 25)] == [8.5730, 8.3068, 8.1597, 7.9730, 7.9118, 7.9038]
    assert [round(float(e[i]), 4) for i in (0, 1, 5, 10, 25, 50, 100)] == [8.2999, 8.2580, 8.1666, 8.1702, 8.2351, 8.2530, 8.2540]
    assert ev._shape(c) == "monoton fallend" and ev._shape(e) == "nicht monoton"
    assert int(np.argmin(e[:200])) == 7 and round(float(e[7]), 4) == 8.1598
    assert ev._nu_equal(c, n["cusum"]["steady"]) == 22 and round(float(c[25]), 4) == 7.9038
    surv = {"cusum": M.build_card("cusum", 0.5, 370.0).pre.surv, "ewma": M.build_card("ewma", 0.1, 370.0).pre.surv}
    for nu, (sc, se_) in ((100, (0.7696, 0.7708)), (200, (0.5852, 0.5858)), (500, (0.2573, 0.2571))):
        assert (round(float(surv["cusum"][nu]), 4), round(float(surv["ewma"][nu]), 4)) == (sc, se_), nu  # P(kein Alarm bis ν) ohne Änderung
    assert round(n["cusum"]["steady"], 4) == 7.9038 and round(n["ewma"]["steady"], 4) == 8.2540 and round(n["cusum"]["zero"], 4) == 8.5730 and round(n["ewma"]["zero"], 4) == 8.2999


def test_claim_pollak_grenzwert():
    c = M.build_card("ewma", 0.1, 370.0)
    for d, zero, poll in ((1.5, 5.0871, 5.0982), (2.0, 3.7146, 3.7408), (3.0, 2.4880, 2.5250)):
        x = c.delays(d)
        assert x["nu_pollak"] is None and round(x["zero"], 4) == zero and round(x["pollak"], 4) == poll and x["pollak"] == pytest.approx(x["steady"], rel=1e-12)
        assert x["D"][150] == pytest.approx(x["steady"], abs=1e-5)  # von unten erreicht: nach 150 Schritten innerhalb 1e-5
    for d in (1.0, 4.0):
        assert c.delays(d)["nu_pollak"] == 0  # bei δ = 1 und 4 liegt das Maximum bei ν = 0
    cus = M.build_card("cusum", 0.5, 370.0)
    assert all(cus.delays(d)["nu_pollak"] == 0 for d in C.DELTA_GRID)


def test_claim_vorsprung():
    assert round(M.build_card("cusum", 0.5, 370.0, False, True).thr, 4) == 4.1488
    assert round(M.build_card("ewma", 0.1, 370.0, False, True).thr, 4) == 0.5586 and round(M.build_card("ewma", 0.1, 370.0).thr, 4) == 0.5512
    zero_c = [ev.exp_nu(0.1, 0.5, 370, d) for d in (0.5, 1.0, 2.0)]
    ref = {"zero": ((0.7742, 0.6357, 0.6066), (0.7566, 0.6663, 0.6345)), "pollak": ((0.9716, 0.9338, 0.9225), (1.0104, 1.0109, 1.0118)),
           "steady": ((1.0210, 1.0129, 1.0109), (1.0274, 1.0165, 1.0118)), "cyc": ((1.0188, 1.0097, 1.0075), (1.0216, 1.0090, 1.0036))}
    for m, (rc, re_) in ref.items():
        for i, n in enumerate(zero_c):
            assert n[True]["cusum"][m] / n[False]["cusum"][m] == pytest.approx(rc[i], abs=5.1e-5), (m, i)
            assert n[True]["ewma"][m] / n[False]["ewma"][m] == pytest.approx(re_[i], abs=5.1e-5), (m, i)
    lorden = [n[True]["cusum"]["lorden"] / n[False]["cusum"]["lorden"] for n in zero_c]
    assert lorden == pytest.approx([1.0204, 1.0124, 1.0104], abs=5.1e-5)
    # Aussagen im Text: nur im zero-state gewinnen beide (23 bis 39 %); im Pollak-Maß gewinnt es nur der CUSUM; in den übrigen Maßen kostet es 0,4 bis 2,7 %
    zs = [n[True][k]["zero"] / n[False][k]["zero"] for n in zero_c for k in ("cusum", "ewma")]
    assert round((1 - max(zs)) * 100) == 23 and round((1 - min(zs)) * 100) == 39
    assert all(n[True]["cusum"]["pollak"] < n[False]["cusum"]["pollak"] and n[True]["ewma"]["pollak"] > n[False]["ewma"]["pollak"] for n in zero_c)
    kosten = [n[True][k][m] / n[False][k][m] - 1 for n in zero_c for k in ("cusum", "ewma") for m in ("steady", "cyc")] + [x - 1 for x in lorden]
    assert min(kosten) * 100 == pytest.approx(0.36, abs=0.01) and max(kosten) * 100 == pytest.approx(2.74, abs=0.01) and min(kosten) > 0
    # zero-state = Neustartwert: Vorsprung H/2 senkt auch beim Anfang des Zyklus nichts (zyklisch > 1)
    assert all(n[True][k]["cyc"] > n[False][k]["cyc"] for n in zero_c for k in ("cusum", "ewma"))


def test_claim_lorden_einseitig():
    r = M.lorden_einseitig(0.1, 370.0, 1.0)
    at = {float(b): float(v) for b, v in zip(r["b"], r["L"])}
    for b, ref in ((0, 8.2999), (1, 10.1849), (2, 11.7647), (5, 15.3727), (10, 19.4050), (20, 24.4065), (30, 27.6671)):
        assert at[float(b)] == pytest.approx(ref, abs=5.1e-5 if b in (0, 1, 2, 5, 30) else 1e-3), b
    assert f"{at[10.0]:.2f}" == "19.40" and bool(np.all(np.diff(r["L"]) > 0)) and len(r["L"]) == 61
    assert at[30.0] - at[10.0] == pytest.approx(8.26, abs=0.02)  # „etwa +8“ zwischen b = 10 und 30
    assert (round(at[20.0] - at[10.0], 1), round(at[30.0] - at[20.0], 1)) == (5.0, 3.3)  # abnehmender Zuwachs (README, Befund 7)
    assert at[20.0] - at[10.0] > at[30.0] - at[20.0] > 0


def test_claim_lorden_zweiseitig():
    ref = {0.5: (28.2172, 33.3386, -0.749, 29.3223, 35.2537, 0.8004, 0.9457), 1.0: (9.7354, 13.5565, -0.909, 13.3296, 9.9247, 0.9809, 1.3659), 2.0: (4.1803, 6.6860, -1.0, 6.6860, 3.8578, 1.0836, 1.7331)}
    for d, (zero, lor, pos, edge, c2, r0, rl) in ref.items():
        a = ev.exp_lorden(0.1, True, 370, d, 0)
        assert (round(a["zero"], 4), round(a["lorden"], 4), round(a["edge"], 4), round(a["cusum2"][1], 3)) == (zero, lor, edge, round(c2, 3))
        assert a["u_sup"] / a["H"] == pytest.approx(pos, abs=0.001) and a["cusum2"][1] == pytest.approx(c2, abs=1e-3)
        assert (round(a["zero"] / a["cusum2"][1], 4), round(a["lorden"] / a["cusum2"][1], 4)) == (r0, rl)
    a = ev.exp_lorden(0.1, True, 370, 1.0, 0)
    assert round(a["steady"], 4) == 9.5292 and round(a["cyc"], 4) == 9.5317 and round(a["pollak"], 4) == 9.7354  # Plan: EWMA zero = Pollak 9,7354, steady 9,5292, zyklisch 9,5317
    assert round(1.0 - a["up_share_edge"], 4) == 0.0585
    assert round(1.0 - M.anteil_alarm_nach_oben(0.1, a["H"], 1.0, -0.97 * a["H"]), 4) == 0.0418
    assert R.cusum2_h(370) == pytest.approx(4.7738, abs=5e-5)  # zweiseitiger CUSUM, h für ARL_0 = 370


def test_claim_schwellen():
    ref = {"ewma_ein": {0.1: (1.7379, 2.4026, 2.8080), 0.2: (2.0208, 2.5976, 2.9594)}, "ewma_zwei": {0.1: (2.1476, 2.7010, 3.0586)}}
    for sec, rows in ref.items():
        for lam, ls in rows.items():
            assert [round(T.ewma_L(lam, g, sec == "ewma_zwei"), 4) for g in (100.0, 370.0, 1000.0)] == list(ls), (sec, lam)
    assert [round(T.cusum_h(0.5, g), 4) for g in (100.0, 370.0, 1000.0)] == [2.8494, 4.0954, 5.0707]
    assert round(M.build_card("ewma", 0.1, 370.0).thr, 4) == 0.5512 and round(M.build_card("ewma", 0.1, 370.0, True).thr, 4) == 0.6197
    assert round(T.ewma_L(1.0, 370.0, True), 4) == 2.9997 and round(K.shewhart_c(370.0, True), 4) == 2.9997


def test_claim_orakel_gitter_und_shewhart():
    lam, H = 0.1, 2.7 * K.sigma_z(0.1)
    assert round(K.arl_ewma(lam, H, 0.0, True), 8) == 368.99373398
    assert [round(K.ewma_grid_arl(lam, H, 0.0, True, m), 4) for m in (20, 40, 80, 160)] == [362.6495, 367.3496, 368.5763, 368.8887]  # der Plan nannte 362,6496: Rundung auf der Grenze (362,64955)
    errs = [abs(K.ewma_grid_arl(lam, H, 0.0, True, m) - 368.99373398) for m in (20, 40, 80, 160)]
    ratios = [errs[i] / errs[i + 1] for i in range(3)]
    assert min(ratios) >= 3.86 - 0.005 and max(ratios) <= 3.97 + 0.005
    assert [round(K.arl_ewma(1.0, 3.0, d, True), 2) for d in (0.0, 1.0, 2.0)] == [370.40, 43.89, 6.30]
    for d in (0.0, 1.0, 2.0):
        assert K.arl_ewma(1.0, 3.0, d, True) == pytest.approx(1.0 / K.shewhart_p(3.0, d, True), rel=1e-6)  # gleich 1/p auf 6 Stellen
    assert 1.0 / K.shewhart_p(1.0 * 3.0, 0.0, True) == pytest.approx(370.398347, abs=1e-5)
    # einseitige Shewhart-Karte bei δ = 1: 26,746 (so auch Plan 02), hier als λ = 1 einseitig
    assert round(M.build_card("ewma", 1.0, 370.0).delays(1.0)["zero"], 3) == 26.746


def test_claim_kernaussage():
    cmp2 = ev.exp_compare(0.2, 0.5, 370, 1.0)["table_ratio"]
    zero_pct = [(1 - x) * 100 for x in cmp2["zero"]]
    assert round(min(zero_pct), 1) == 4.6 and round(max(zero_pct), 1) == 8.7 and bool(np.all(cmp2["pollak"] == cmp2["zero"]))  # „4,6 bis 8,7 % kürzer“, acht von acht
    assert [round(x, 4) for x in cmp2["steady"][:3]] == [0.9256, 0.9629, 1.0002]
    langer = [(x - 1) * 100 for x in cmp2["steady"][3:]]
    assert round(min(langer), 1) == 1.7 and round(max(langer), 1) == 3.0  # „quasistationär ab δ = 1 um 1,7 bis 3,0 % länger“
    langer_cyc = [(x - 1) * 100 for x in cmp2["cyc"][3:]]
    assert round(min(langer_cyc), 1) == 1.6 and round(max(langer_cyc), 1) == 2.9  # „zyklisch 1,6 bis 2,9 %“; zusammen „1,6 bis 3,0 %“
    assert "1,6 bis 3,0 % länger (zyklisch 1,6 bis 2,9 %, quasistationär 1,7 bis 3,0 %)" in README
    cr = ev.exp_compare(0.1, 0.5, 370, 1.0)["crossing"]
    assert 1.10 < cr["zero"]["delta"] <= 1.15 and 0.85 < cr["steady"]["delta"] <= 0.90
    assert README.count("19,40") >= 2 and "13,56" in README and "9,92" in README and "−0,91 H" in README and "13,33" in README and "9,74" in README


def test_claim_fehlalarm_anteil():
    for lam, two, ref in ((0.1, False, 22.92), (0.2, False, 23.25), (0.1, True, 22.58), (1.0, True, 23.71)):
        a = ev.exp_card(lam, two, 370, 1.0)
        assert round(a["share100"] * 100, 2) == ref, (lam, two)
    assert 1.0 - math.exp(-100 / 370.0) == pytest.approx(0.2371, abs=0.025)  # Größenordnung: exponentielle Näherung
    assert ev.exp_card(1.0, True, 370, 1.0)["shewhart0"] == pytest.approx(370.0, rel=1e-9)


def test_claim_simulation_baender():
    ref = {(0.1, False): (8.2999, ((8.22, 0.13, -0.6), (8.48, 0.12, 1.5))), (0.2, False): (8.1752, ((8.00, 0.15, -1.2), (8.44, 0.15, 1.7))),
           (0.1, True): (9.7354, ((9.74, 0.15, 0.0), (9.88, 0.14, 1.0)))}
    for (lam, two), (exact, rows) in ref.items():
        a = ev.exp_card(lam, two, 370, 1.0)
        assert round(a["arl1"], 4) == exact
        for seed, (m, se, z) in zip((35, 36), rows):
            s = ev.exp_sim(lam, two, 370, 1.0, 1000, seed, a["arl1"])
            assert (round(s["mean"], 2), round(s["mean_se"], 2), round((s["mean"] - a["arl1"]) / s["mean_se"], 1)) == (m, se, z), (lam, two, seed)
            assert abs((s["mean"] - a["arl1"]) / s["mean_se"]) <= 4.0 and abs((s["share"] - a["share100"]) / s["share_se"]) <= 4.0


def test_claim_spielzeug():
    t = R.toy_measures()
    assert (t["arl0"], t["arl0_schlechtester"], t["zero"], t["zyklisch"], t["lorden"]) == (F(4), F(6), F(16, 9), F(22, 9), F(28, 9))
    assert round(t["steady"], 4) == 2.6018 and t["pollak"] == pytest.approx(28 / 9, rel=1e-12)
    assert ev.toy_text() == "zero-state 16/9 = 1,7778; zyklisch 22/9 = 2,4444; steady-state (quasistationär) 2,6018; Pollak = Lorden 28/9 = 3,1111"
    n = ev.exp_nu(0.1, 0.5, 370, 1.0)[False]
    assert round(n["ewma"]["zero"], 4) == 8.2999 and round(n["ewma"]["steady"], 4) == 8.2540 and n["ewma"]["zero"] > n["ewma"]["steady"]  # in der Normal-EWMA liegt zero-state über steady-state


def test_claim_abweichungen_vom_plan():
    r = M.lorden_einseitig(0.1, 370.0, 1.0)
    b10 = float(r["L"][int(np.argmin(np.abs(r["b"] - 10.0)))])
    assert round(b10, 4) == 19.4050 and ev.de(b10, 2) == "19,40" and round(b10, 6) == 19.404956
    assert R.cusum2_h(370) == pytest.approx(4.773829, abs=1e-6)
    assert [ev.de(R.cusum2_frozen(370, d)[1], 4) for d in (0.5, 1.0, 2.0)] == ["35,2537", "9,9247", "3,8578"]
    n = ev.exp_nu(0.1, 0.5, 370, 1.0)[False]["cusum"]
    assert ev._nu_equal(n["D"], n["steady"]) == 22
    assert [round(ev.exp_card(lam, two, 370, 1.0)["share100"] * 100, 2) for lam, two in ((0.1, False), (0.2, False), (0.1, True))] == [22.92, 23.25, 22.58]
    assert [round(x, 4) for x in (M.build_card("ewma", 0.1, 370.0).delays(1.5)["pollak"], M.build_card("ewma", 0.1, 370.0).delays(1.5)["steady"])] == [5.0982, 5.0982]


# --- Text: README und App sind dieselbe Aussage ------------------------------------------------------------------------------------------------
def test_annahmentabelle_der_app_steht_unveraendert_in_der_readme():
    assert TX.ASSUMPTIONS_TABLE.strip() in README


def test_readme_nennt_jede_gefrorene_datei_und_die_regeln_der_linie():
    for name in ("ewk_tuned.json", "ewk_cusum2.json", "ewk_abgestimmt.json", "tools/ewk_make_tuned.py"):
        assert name in README and (ROOT / name).exists(), name
    for text in ("ARL₀ = 370 (gesetzt", "Zufalls-Seed", "quasistationär", "zyklisch", "Shiryaev", "Anfangsvorsprung", "ist eine Auslegung", "nicht belegt", "nicht gelesen"):
        assert text in README, text
    import re

    assert "Saatwert" not in README and "Schwanz" not in README
    assert not re.search(r"\b(trägt|tragen|tragend\w*)\b", README), "„trägt/tragen“ vermeiden: reicht, gilt, zeigt"


# --- gefrorene Daten gegen Neurechnung -------------------------------------------------------------------------------------------------------
def test_gefrorene_daten_gleich_neu_gerechnet():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ewk_make_tuned", ROOT / "tools" / "ewk_make_tuned.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    for name, new in (("ewk_tuned.json", tool.make_tuned()), ("ewk_abgestimmt.json", tool.make_abgestimmt())):
        old = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert tool._compare(name, new, old, 1e-6) == 0, name
        abschnitte = [v for k, v in old.items() if isinstance(v, dict) and k != "meta"]
        assert abschnitte and (all(a.get("quelle") and a.get("stand") for a in abschnitte) or (old.get("quelle") and old.get("stand")))  # Quelle und Stand je Abschnitt
    cusum2 = json.loads((ROOT / "ewk_cusum2.json").read_text(encoding="utf-8"))
    assert cusum2["quelle"] and cusum2["stand"]
    for g, d in (("370.0", 1.0), ("100.0", 0.5)):  # Stichprobe der 2D-Kette (die ganze Datei neu zu rechnen dauert rund 1 Minute: tools/ewk_make_tuned.py --check)
        h = cusum2["werte"][g]["h"]
        assert tool.cusum2_arl(0.5, h, d)[0] == pytest.approx(cusum2["werte"][g]["delta"][repr(d)]["verzoegerung"], rel=1e-6)
        assert tool.cusum2_arl(0.5, h, 0.0)[0] == pytest.approx(float(g), rel=1e-6)  # h ist so gewählt, dass die extrapolierte ARL_0 das Ziel trifft


def test_rng_kopie_gleich_vorlage_falls_vorhanden():
    vorlage = ROOT.parent / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"
    if not vorlage.exists():
        pytest.skip("Vorlage liegt nur im Portfolio-Ordner")
    assert (ROOT / "ewk_rng.py").read_text(encoding="utf-8") == vorlage.read_text(encoding="utf-8")


def test_zufallsfolge_ist_plattformfest():
    """SplitMix64 mit Seed 0: die ersten drei Ausgaben sind die der veröffentlichten Referenz (ganze Zahlen, auf jeder Plattform bitgleich); random() ist 53 Bit gleichverteilt in [0, 1)."""
    g = S.SplitMix64(0)
    assert [g.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    r = S.SplitMix64(35)
    u = [r.random() for _ in range(2000)]
    assert min(u) >= 0.0 and max(u) < 1.0 and abs(float(np.mean(u)) - 0.5) < 0.03
    z = [S.SplitMix64(35).normal() for _ in range(3)]
    assert z == [S.SplitMix64(35).normal() for _ in range(3)]  # gleiche Zufalls-Seeds, gleiche Normalzahl (exact_log, exact_cos)
