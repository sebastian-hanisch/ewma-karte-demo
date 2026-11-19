"""Rechnungen der vier Experimente und ihre Verdikte (reine Funktionen: die App ruft st.<Art>(Text)).

Experiment 1: die Karte laufen lassen und ihr Versprechen nachrechnen (ARL_0, ARL_1, Lauflängenverteilung, Gitter, Simulation).
Experiment 2: welche Karte ist bei gleicher ARL_0 schneller, in welchem Maß (EWMA gegen CUSUM, fest oder auf den Sprung abgestimmt).
Experiment 3: hängt die Verzögerung davon ab, wann die Änderung kommt (D_nu, Anfangsvorsprung).
Experiment 4: gibt es ein schlechtestes Vorleben (Lorden-Maß).

Voraussetzungen überall: unabhängige Beobachtungen N(delta, 1), sigma = 1 bekannt, feste EWMA-Grenzen H = L sigma_Z, Zähleinheit Beobachtungen bis zum Alarm.
"""

from dataclasses import dataclass

import numpy as np

import ewk_chains as K
import ewk_constants as C
import ewk_measures as M
import ewk_reference as R
import ewk_scenario as S

GRID_M = 80          # Zellen je Seite des Brook/Evans-Gitters (Zweitentscheider)
GRID_TOL = 0.005     # Abweichung Gitter gegen Integralgleichung über 0,5 %: ein Fehler statt einer Zahl
NU_PLOT = 1000       # D_nu wird bis nu = 1000 gezeigt
Z_SIM = 4.0          # |z| der Simulation gegen die exakte Zahl: darüber warnt die App


# ----------------------------------------------------------------- Zahlenformat
def de(x: float, nd: int = 2, sign: bool = False) -> str:
    """Deutsche Zahl: Dezimalkomma, echtes Minuszeichen."""
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def sci(x: float) -> str:
    """Zahl in Zehnerpotenz-Schreibweise mit deutschem Komma: 2,87·10⁻⁷."""
    mant, exp = f"{x:.2e}".split("e")
    sup = str(int(exp)).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{sup}"


def pct(ratio: float, nd: int = 1) -> str:
    """Änderung eines Verhältnisses in Prozent mit Vorzeichen: 0,6357 gibt −36,4 %."""
    return de((ratio - 1.0) * 100.0, nd, sign=True) + " %"


def klass(ratio: float) -> str:
    """EWMA gegen CUSUM: Verhältnis unter 1 heißt EWMA schneller; |Verhältnis − 1| <= TIE heißt Gleichstand."""
    if ratio < 1.0 - C.TIE:
        return "schneller"
    if ratio > 1.0 + C.TIE:
        return "langsamer"
    return "Gleichstand"


@dataclass(frozen=True)
class Settings:
    lam: float = C.DEFAULT_LAMBDA
    side: str = C.DEFAULT_SIDE
    arl0: int = C.DEFAULT_ARL0
    k: float = C.DEFAULT_K
    delta: float = C.DEFAULT_DELTA
    nu: int = C.DEFAULT_NU
    nobs: int = C.DEFAULT_NOBS
    seed: int = C.DEFAULT_SEED
    runs: int = C.DEFAULT_RUNS
    design: str = C.DEFAULT_DESIGN
    vorsprung: str = C.DEFAULT_START
    b: int = C.DEFAULT_B

    @property
    def two(self) -> bool:
        return self.side == "zweiseitig"


# =================================================================== Experiment 1: die Karte und ihr Versprechen
def exp_card(lam: float, two: bool, arl0: float, delta: float) -> dict:
    """Exakt (Integralgleichung) und mit dem Gitter als Zweitentscheider: ARL_0, ARL_1 (zero-state), Lauflängenverteilung, Anteil Fehlalarme bis Beobachtung 100."""
    card = M.build_card("ewma", lam, float(arl0), two, False)
    ch1 = card.chain(delta)
    arl1 = float(K.solve_arl(ch1)[ch1.s])
    arl0_exact = card.pre.arl0
    pmf0, surv0 = M.lauflaengen_verteilung(card, 0.0, C.SERIES_TMAX)
    pmf1, surv1 = M.lauflaengen_verteilung(card, delta, C.SERIES_TMAX)
    out = {
        "lam": lam, "two": two, "L": card.mult, "H": card.thr, "sigma_z": K.sigma_z(lam),
        "arl0": arl0_exact, "arl1": arl1,
        "grid0": K.ewma_grid_arl(lam, card.thr, 0.0, two, GRID_M), "grid1": K.ewma_grid_arl(lam, card.thr, delta, two, GRID_M),
        "pmf0": pmf0, "pmf1": pmf1, "surv0": surv0, "surv1": surv1,
        "share100": float(1.0 - surv0[C.SIM_SHARE_T]), "share100_delta": float(1.0 - surv1[C.SIM_SHARE_T]),
        "shewhart0": None, "shewhart1": None,
    }
    if lam == 1.0:  # gedächtnislos: geschlossene Form 1/p
        out["shewhart0"] = 1.0 / K.shewhart_p(card.thr, 0.0, two)
        out["shewhart1"] = 1.0 / K.shewhart_p(card.thr, delta, two)
    return out


def exp_series(lam: float, two: bool, arl0: float, delta: float, nu: int, nobs: int, seed: int) -> S.Series:
    card = M.build_card("ewma", lam, float(arl0), two, False)
    return S.make_series(lam, card.thr, two, delta, nu, nobs, seed)


def exp_sim(lam: float, two: bool, arl0: float, delta: float, runs: int, seed: int, arl1_exact: float) -> dict:
    """Wiederholungssimulation (SplitMix64) als Gegenprobe: Anteil Fehlalarme bis Beobachtung 100 (ohne Änderung) und, wenn bezahlbar, das Mittel der Lauflänge."""
    card = M.build_card("ewma", lam, float(arl0), two, False)
    stepper = S.ewma_stepper(lam, card.thr, two)
    k = S.alarms_within(stepper, 0.0, runs, seed + 2, C.SIM_SHARE_T)
    p, p_se = S.proportion_se(k, runs)
    out = {"runs": runs, "share": p, "share_se": p_se, "mean": None, "mean_se": None}
    if arl1_exact <= C.SIM_MEAN_LIMIT:
        m, se = S.mean_se(S.run_lengths(stepper, delta, runs, seed + 1))
        out["mean"], out["mean_se"] = m, se
    return out


def verdict_card(a: dict, sim: dict, delta: float) -> tuple[str, str]:
    """Verdikt Experiment 1: die App nennt ihre eigenen Zahlen; Gitter über 0,5 % Abweichung ist ein Fehler, keine Zahl."""
    seite = "zweiseitig" if a["two"] else "einseitig"
    dev0 = abs(a["grid0"] - a["arl0"]) / a["arl0"]
    dev1 = abs(a["grid1"] - a["arl1"]) / a["arl1"]
    if max(dev0, dev1) > GRID_TOL:
        return "error", (f"Das Gitter (Zweitentscheider) weicht um {de(max(dev0, dev1) * 100, 2)} % von der Integralgleichung ab (Grenze {de(GRID_TOL * 100, 1)} %): "
                         "keine Zahl wird gemeldet.")
    parts = [f"λ = {de(a['lam'])}, L = {de(a['L'], 4)}, {seite}: ARL₀ = {de(a['arl0'])} Beobachtungen bis zum Fehlalarm (Gitter {de(a['grid0'])}, Abweichung {de(dev0 * 100, 2)} %)."]
    if delta > 0:
        parts.append(f"Bei δ = {de(delta)} und Änderung vor der ersten Beobachtung im Mittel {de(a['arl1'])} Beobachtungen bis zum Alarm (zero-state; Gitter {de(a['grid1'])}).")
    else:
        parts.append("Ohne Sprung (δ = 0) ist die Lauflänge die der Karte vor der Änderung.")
    parts.append(f"Ohne Änderung alarmiert die Karte mit {de(a['share100'] * 100, 2)} % schon bis Beobachtung {C.SIM_SHARE_T} (exakt).")
    kind = "success"
    if sim["mean"] is not None:
        z = (sim["mean"] - a["arl1"]) / sim["mean_se"] if sim["mean_se"] > 0 else 0.0
        parts.append(f"Simulation ({sim['runs']} Läufe, Zufalls-Seed der Demo): {de(sim['mean'])} ± {de(sim['mean_se'])} gegen exakt {de(a['arl1'])} (z = {de(z, 1, sign=True)}).")
        if abs(z) > Z_SIM:
            kind = "warning"
    zs = (sim["share"] - a["share100"]) / sim["share_se"] if sim["share_se"] > 0 else 0.0
    parts.append(f"Simulierter Anteil Fehlalarme bis {C.SIM_SHARE_T}: {de(sim['share'] * 100, 2)} % ± {de(sim['share_se'] * 100, 2)} (z = {de(zs, 1, sign=True)}).")
    if abs(zs) > Z_SIM:
        kind = "warning"
    if a["shewhart0"] is not None:
        parts.append(f"λ = 1 ist die Shewhart-Karte: gleich der geschlossenen Form 1/p, ARL₀ = {de(a['shewhart0'])}"
                     + (f", ARL₁ = {de(a['shewhart1'])}." if delta > 0 else "."))
    return kind, " ".join(parts)


# =================================================================== Experiment 2: wer ist schneller, in welchem Maß
def crossing(deltas, ratios) -> dict:
    """Kreuzungsstelle auf dem Gitter: kleinstes δ, ab dem die EWMA nicht mehr schneller ist; dazu die Zahl der Vorzeichenwechsel des Gitters."""
    r = np.asarray(ratios, dtype=float)
    below = r < 1.0
    wechsel = int(np.sum(np.diff(below.astype(int)) != 0))
    if below.all():
        return {"art": "alle", "delta": None, "wechsel": 0}
    if not below.any():
        return {"art": "keine", "delta": None, "wechsel": 0}
    return {"art": "ab", "delta": float(deltas[int(np.argmax(~below))]), "wechsel": wechsel}


def exp_compare(lam: float, k: float, arl0: float, delta: float) -> dict:
    """Feste Auslegung (ein lambda, ein k für alle Sprünge), einseitig, ARL_0 = arl0: Verzögerungen beider Karten in den vier Maßen über das δ-Gitter."""
    ewma = M.build_card("ewma", lam, float(arl0))
    cus = M.build_card("cusum", k, float(arl0))
    e_rows = {m: [] for m in C.MEASURES}
    c_rows = {m: [] for m in C.MEASURES}
    for d in C.DELTA_CURVE:
        e, c = ewma.delays(d), cus.delays(d)
        for m in C.MEASURES:
            e_rows[m].append(e[m])
            c_rows[m].append(c[m])
    e_rows = {m: np.array(v) for m, v in e_rows.items()}
    c_rows = {m: np.array(v) for m, v in c_rows.items()}
    ratio = {m: e_rows[m] / c_rows[m] for m in C.MEASURES}
    sel_e, sel_c = ewma.delays(delta), cus.delays(delta)
    idx = [C.DELTA_CURVE.index(d) for d in C.DELTA_GRID]
    return {
        "design": "fest", "lam": lam, "k": k, "arl0": arl0, "L": ewma.mult, "h": cus.thr,
        "deltas": np.array(C.DELTA_CURVE), "ratio": ratio, "ewma": e_rows, "cusum": c_rows,
        "table_deltas": np.array(C.DELTA_GRID),
        "table_ratio": {m: ratio[m][idx] for m in C.MEASURES},
        "table_ewma": {m: e_rows[m][idx] for m in C.MEASURES},
        "table_cusum": {m: c_rows[m][idx] for m in C.MEASURES},
        "crossing": {m: crossing(C.DELTA_CURVE, ratio[m]) for m in C.MEASURES},
        "selected": {m: sel_e[m] / sel_c[m] for m in C.MEASURES},
        "selected_values": {m: (sel_e[m], sel_c[m]) for m in C.MEASURES},
        "best1": R.besttable(arl0, 1.0),
    }


def exp_compare_tuned(arl0: float, delta: float) -> dict:
    """Beide Karten auf den wahren Sprung abgestimmt (Orakel, im Betrieb nicht verfügbar): gefrorene Tabelle über die acht Sprünge."""
    rows = {d: R.besttable(arl0, d) for d in C.DELTA_GRID}
    if any(v is None for v in rows.values()):
        raise ValueError(f"für ARL₀ = {arl0} liegt keine abgestimmte Tabelle vor")
    ratio = {m: np.array([rows[d][m]["verhaeltnis"] for d in C.DELTA_GRID]) for m in C.MEASURES}
    return {
        "design": "abgestimmt", "arl0": arl0, "table_deltas": np.array(C.DELTA_GRID), "table_ratio": ratio,
        "table_ewma": {m: np.array([rows[d][m]["ewma"] for d in C.DELTA_GRID]) for m in C.MEASURES},
        "table_cusum": {m: np.array([rows[d][m]["cusum"] for d in C.DELTA_GRID]) for m in C.MEASURES},
        "best_lambda": {m: [rows[d][m]["lambda"] for d in C.DELTA_GRID] for m in C.MEASURES},
        "best_k": {m: [rows[d][m]["k"] for d in C.DELTA_GRID] for m in C.MEASURES},
        "best1": R.besttable(arl0, 1.0),
        "crossing": {m: crossing(C.DELTA_GRID, ratio[m]) for m in C.MEASURES},
    }


def _scope(ratios) -> str:
    cl = [klass(r) for r in ratios]
    n, fast, tie, slow = len(cl), cl.count("schneller"), cl.count("Gleichstand"), cl.count("langsamer")
    parts = [f"{fast} von {n} schneller"]
    if tie:
        parts.append(f"{tie} Gleichstand")
    if slow:
        parts.append(f"{slow} langsamer")
    return ", ".join(parts) + f" (Verhältnis {de(min(ratios), 4)} bis {de(max(ratios), 4)})"


def verdict_compare(a: dict) -> tuple[str, str]:
    """Verdikt Experiment 2: je Maß die Zählung über die acht Sprünge, die Kreuzungsstellen und die Falle (ein Vergleich „bei gleicher ARL₀“ legt das Maß nicht fest)."""
    ratios = a["table_ratio"]
    nd = len(a["table_deltas"])
    verdict_parts = []
    if a["design"] == "fest":
        verdict_parts.append(f"ARL₀ = {a['arl0']} (gesetzt), einseitig, CUSUM k = {de(a['k'])} gegen EWMA λ = {de(a['lam'])} (feste Auslegung für alle Sprünge). "
                             f"EWMA gegen CUSUM bei {nd} Sprüngen δ = {de(a['table_deltas'][0])} bis {de(a['table_deltas'][-1])}: "
                             f"zero-state {_scope(ratios['zero'])}; Pollak {_scope(ratios['pollak'])}; steady-state (quasistationär) {_scope(ratios['steady'])}; "
                             f"zyklisch {_scope(ratios['cyc'])}.")
        cr = a["crossing"]

        def cross_txt(m):
            c = cr[m]
            if c["art"] == "ab":
                return f"ab δ = {de(c['delta'])} nicht mehr schneller"
            return "auf dem ganzen Gitter schneller" if c["art"] == "alle" else "auf dem ganzen Gitter nicht schneller"

        verdict_parts.append(f"Kreuzungsstelle (Gitter δ = 0,1 bis 4 in Schritten von 0,05): zero-state {cross_txt('zero')}, steady-state {cross_txt('steady')}.")
        sel = a["selected"]
        verdict_parts.append("Beim gewählten Sprung: " + "; ".join(f"{C.MEASURE_NAMES[m]} {de(sel[m], 4)} ({klass(sel[m])})" for m in C.MEASURES) + ".")
    else:
        verdict_parts.append(f"ARL₀ = {a['arl0']} (gesetzt), einseitig; beide Karten auf den jeweiligen Sprung abgestimmt (Orakel: das beste λ und das beste k je Sprung und Maß, "
                             f"im Betrieb unbekannt): zero-state {_scope(ratios['zero'])}; steady-state (quasistationär) {_scope(ratios['steady'])}.")
    b1 = a.get("best1")
    if b1 is not None:
        verdict_parts.append(f"„Bestes λ gegen bestes k“ bei δ = 1 gibt zero-state {de(b1['zero']['verhaeltnis'], 4)} und steady-state {de(b1['steady']['verhaeltnis'], 4)}.")
    classes = {klass(r) for m in C.MEASURES for r in ratios[m]}
    if "schneller" in classes and "langsamer" in classes:
        verdict_parts.append("Keine der beiden Karten gewinnt in allen Maßen und bei allen geprüften Sprüngen: Die Rangfolge hängt am Maß, ein Vergleich „bei gleicher ARL₀“ legt es nicht fest.")
        return "warning", " ".join(verdict_parts)
    verdict_parts.append("Hier ist die Reihenfolge in allen vier Maßen und bei allen geprüften Sprüngen gleich; das gilt für diese Auslegung, nicht für jede.")
    return "info", " ".join(verdict_parts)


# =================================================================== Experiment 3: hängt die Verzögerung von nu ab
def _shape(D) -> str:
    d = np.diff(np.asarray(D, dtype=float))
    if np.all(np.abs(d) <= 1e-9):
        return "konstant"
    if np.all(d <= 1e-9):
        return "monoton fallend"
    if np.all(d >= -1e-9):
        return "monoton wachsend"
    return "nicht monoton"


def _nu_equal(D, steady: float, tol: float = 5e-5) -> int:
    """Erstes nu, ab dem D_nu dauerhaft auf 4 Stellen gleich dem Grenzwert ist (|D - Grenzwert| <= 5e-5)."""
    bad = np.nonzero(np.abs(np.asarray(D) - steady) > tol)[0]
    return int(bad[-1] + 1) if len(bad) else 0


def exp_nu(lam: float, k: float, arl0: float, delta: float) -> dict:
    """D_nu beider Karten, ohne und mit Anfangsvorsprung (H/2 bzw. h/2, Schwelle neu auf ARL_0 geeicht), einseitig."""
    out = {"lam": lam, "k": k, "arl0": arl0, "delta": delta}
    for hs in (False, True):
        e = M.build_card("ewma", lam, float(arl0), False, hs)
        c = M.build_card("cusum", k, float(arl0), False, hs)
        de_, dc_ = e.delays(delta), c.delays(delta)
        out[hs] = {
            "ewma": {key: de_[key] for key in ("zero", "pollak", "nu_pollak", "steady", "cyc", "lorden")} | {"D": de_["D"][: NU_PLOT + 1], "surv": e.pre.surv[: NU_PLOT + 1], "thr": e.thr},
            "cusum": {key: dc_[key] for key in ("zero", "pollak", "nu_pollak", "steady", "cyc", "lorden")} | {"D": dc_["D"][: NU_PLOT + 1], "surv": c.pre.surv[: NU_PLOT + 1], "thr": c.thr},
        }
    return out


def verdict_nu(a: dict, vorsprung: bool) -> tuple[str, str]:
    """Verdikt Experiment 3: Form von D_nu je Karte (gewählte Variante) und die Wirkung des Vorsprungs je Maß (immer mit gegen ohne)."""
    sel = a[bool(vorsprung)]
    e, c = sel["ewma"], sel["cusum"]
    delta, arl0 = a["delta"], a["arl0"]
    de200 = e["D"][:200]
    jmin = int(np.argmin(de200))
    nu_c = _nu_equal(c["D"], c["steady"])
    parts = [
        f"δ = {de(delta)}, ARL₀ = {arl0}, einseitig, {'mit' if vorsprung else 'ohne'} Anfangsvorsprung: "
        f"CUSUM k = {de(a['k'])}: {de(c['D'][0], 4)} bei ν = 0, im Grenzwert {de(c['steady'], 4)}"
        + f", ab ν = {nu_c} auf 4 Stellen gleich ({_shape(c['D'])}); "
        f"EWMA λ = {de(a['lam'])}: {de(e['D'][0], 4)} bei ν = 0, kleinster Wert {de(de200[jmin], 4)} bei ν = {jmin} (ν unter 200), im Grenzwert {de(e['steady'], 4)} ({_shape(e['D'])})."
    ]
    n, m_ = a[False], a[True]
    cz, ez = m_["cusum"]["zero"] / n["cusum"]["zero"], m_["ewma"]["zero"] / n["ewma"]["zero"]
    cp, ep = m_["cusum"]["pollak"] / n["cusum"]["pollak"], m_["ewma"]["pollak"] / n["ewma"]["pollak"]
    cs, es = m_["cusum"]["steady"] / n["cusum"]["steady"], m_["ewma"]["steady"] / n["ewma"]["steady"]
    cc, ec = m_["cusum"]["cyc"] / n["cusum"]["cyc"], m_["ewma"]["cyc"] / n["ewma"]["cyc"]
    cl = m_["cusum"]["lorden"] / n["cusum"]["lorden"]
    parts.append(
        f"Vorsprung (Start h/2 bzw. H/2, Schwelle neu auf ARL₀ geeicht), Verzögerung mit gegen ohne, CUSUM / EWMA: zero-state {pct(cz)} / {pct(ez)}; Pollak {pct(cp)} / {pct(ep)}; "
        f"steady-state {pct(cs)} / {pct(es)}; zyklisch {pct(cc)} / {pct(ec)}; Lorden (CUSUM) {pct(cl)}, Lorden (einseitige EWMA) nicht begrenzt."
    )
    ratios = [cz, ez, cp, ep, cs, es, cc, ec, cl]
    if min(ratios) < 1.0 < max(ratios):
        parts.append("Ein Vorsprung ist eine Auslegung, und seine Wirkung hängt am Maß: er senkt die Verzögerung in einigen Maßen und erhöht sie in anderen.")
    else:
        parts.append("Hier wirkt der Vorsprung in allen gezeigten Maßen in dieselbe Richtung.")
    return "info", " ".join(parts)


# =================================================================== Experiment 4: schlechtestes Vorleben (Lorden)
def exp_lorden(lam: float, two: bool, arl0: float, delta: float, b: int) -> dict:
    """Einseitig: Restlauflänge ab Startwert -b sigma_Z (b = 0 bis 30). Zweiseitig: L(u) über [-H, H], Supremum und Ort, Alarmrichtung, zweiseitiger CUSUM (eingefroren)."""
    if not two:
        r = M.lorden_einseitig(lam, float(arl0), delta)
        card = M.build_card("ewma", lam, float(arl0))
        d = card.delays(delta)
        cus = M.build_card("cusum", C.DEFAULT_K, float(arl0))
        dc = cus.delays(delta)
        return {"two": False, "lam": lam, "arl0": arl0, "delta": delta, "b": b, "curve_b": r["b"], "curve_L": r["L"], "H": r["H"], "sigma_z": r["sigma_z"],
                "at": {int(x): float(r["L"][int(np.argmin(np.abs(r["b"] - x)))]) for x in (0, 5, 10, 30)},
                "selected": float(np.interp(b, r["b"], r["L"])), "zero": d["zero"], "steady": d["steady"], "cusum_lorden": dc["lorden"], "cusum_h": cus.thr}
    card = M.build_card("ewma", lam, float(arl0), True)
    ch = card.chain(delta)
    L1 = K.solve_arl(ch)
    d = card.pre.delays(L1)
    n = len(L1) - 1
    us = np.linspace(-card.thr, card.thr, 241)
    Lu = 1.0 + ch.rowfun(us) @ L1[:n]
    sup, u_sup = M.sup_restlauflaenge(ch, L1, card.thr)
    edge = float(1.0 + ch.rowfun(-card.thr) @ L1[:n])
    start = -b * K.sigma_z(lam)
    inside = abs(start) <= card.thr
    cus2 = R.cusum2_frozen(arl0, delta)
    cus2_src = "eingefroren"
    if cus2 is None:
        h2 = R.cusum2_h(arl0)
        cus2 = (h2, R.cusum2_live(h2, delta)) if h2 is not None else None
        cus2_src = "live (Richardson aus m = 60 und 100)"
    return {
        "two": True, "lam": lam, "arl0": arl0, "delta": delta, "b": b, "H": card.thr, "L": card.mult, "sigma_z": K.sigma_z(lam), "curve_u": us, "curve_L": Lu,
        "lorden": sup, "u_sup": u_sup, "edge": edge, "zero": d["zero"], "pollak": d["pollak"], "steady": d["steady"], "cyc": d["cyc"],
        "start": start, "start_inside": bool(inside),
        "start_L": float(1.0 + ch.rowfun(start) @ L1[:n]) if inside else None,
        "up_share_edge": M.anteil_alarm_nach_oben(lam, card.thr, delta, -card.thr),
        "cusum2": cus2, "cusum2_source": cus2_src,
    }


def verdict_lorden(a: dict) -> tuple[str, str]:
    """Verdikt Experiment 4: einseitig wächst die Verzögerung mit der Tiefe des Vorlebens (im gerechneten Bereich ohne Grenze), zweiseitig ist das Lorden-Maß das Supremum im Inneren."""
    delta, arl0, lam = a["delta"], a["arl0"], a["lam"]
    if not a["two"]:
        at = a["at"]
        curve = np.asarray(a["curve_L"])
        s = [f"Einseitig, λ = {de(lam)}, δ = {de(delta)}, ARL₀ = {arl0}: ab Startwert 0 sind es {de(at[0])} Beobachtungen, ab −5 σ_Z {de(at[5])}, ab −10 σ_Z {de(at[10])}, ab −30 σ_Z {de(at[30])}."]
        start = "Startwert 0" if a["b"] == 0 else f"Startwert −{a['b']} σ_Z"
        grow = bool(np.all(np.diff(curve) > 0))
        if grow:
            s.append(f"Die Kurve steigt im gerechneten Bereich (b bis {C.B_MAX}) streng, zwischen b = 10 und 30 um etwa {de(at[30] - at[10], 1)}; "
                     "ohne Grenze gemessen, eine Grenzaussage ist nicht bewiesen.")
        elif float(np.ptp(curve)) < 1e-9:
            s.append("Die Kurve ist konstant: Eine Karte mit λ = 1 hat kein Gedächtnis, das Vorleben spielt keine Rolle.")
        else:
            s.append("Die Kurve steigt hier nicht überall streng; ein Wachstum ohne Grenze ist in diesem Fall nicht gerechnet.")
        s.append(f"Gewählt: {start} gibt {de(a['selected'])} Beobachtungen. Der einseitige CUSUM (k = {de(C.DEFAULT_K)}) hat Lorden = zero-state = {de(a['cusum_lorden'])} "
                 "(der Zustand 0 ist der schlechteste).")
        if grow:
            s.append(f"Das Lorden-Maß gewichtet seltene Vorleben nicht: Z unter −5 σ_Z hat im Sollwert-Betrieb nur etwa die Wahrscheinlichkeit {sci(float(K.PHI(-5.0)))} "
                     "(Näherung Z ≈ N(0, σ_Z²), nicht die quasistationäre Verteilung); welches Maß zum Betrieb passt, entscheidet die Zahl nicht.")
        return "info", " ".join(s)
    gedaechtnislos = abs(a["lorden"] - a["zero"]) <= 1e-9 * a["zero"]
    s = [f"Zweiseitig, λ = {de(lam)}, ARL₀ = {arl0}, δ = {de(delta)}: Lorden-Maß {de(a['lorden'])}"
         + ("" if gedaechtnislos else f" (Supremum der Restlauflänge L(u) bei u = {de(a['u_sup'] / a['H'], 2)} H)")
         + f", am Rand −H sind es {de(a['edge'])}; im zero-state-Maß {de(a['zero'])}."]
    if gedaechtnislos:
        s.append("Die Restlauflänge ist für jeden Startwert gleich: Eine Karte mit λ = 1 hat kein Gedächtnis, Lorden und zero-state fallen zusammen.")
    if a["up_share_edge"] < 1.0 - 1e-9:
        s.append(f"Startet die Karte bei −H, enden {de((1 - a['up_share_edge']) * 100, 2)} % der Läufe mit einem Alarm nach unten (jede Meldung zählt).")
    if a["cusum2"] is not None:
        c2 = a["cusum2"][1]
        s.append(f"Zweiseitiger CUSUM (k = {de(C.DEFAULT_K)}, {a['cusum2_source']}): zero-state = Lorden = {de(c2)}; Verhältnis EWMA/CUSUM zero-state {de(a['zero'] / c2, 4)}, Lorden {de(a['lorden'] / c2, 4)}.")
        if a["zero"] / c2 < 1.0 < a["lorden"] / c2:
            s.append("Das Urteil kehrt sich zwischen den Maßen um: die EWMA ist im zero-state-Maß vorn, im Lorden-Maß hinten.")
    if not gedaechtnislos:
        s.append("Die EWMA hat unter dem Lorden-Maß keine Garantie, und das Maß gewichtet seltene Vorleben nicht; welches Maß zum Betrieb passt, entscheidet die Zahl nicht.")
    return "info", " ".join(s)


# =================================================================== Spielzeug (Expander)
def toy_text() -> str:
    t = R.toy_measures()
    return (f"zero-state {t['zero']} = {de(float(t['zero']), 4)}; zyklisch {t['zyklisch']} = {de(float(t['zyklisch']), 4)}; "
            f"steady-state (quasistationär) {de(t['steady'], 4)}; Pollak = Lorden {t['lorden']} = {de(float(t['lorden']), 4)}")


# =================================================================== Kurzantworten auf die vier Kernfragen
def kernfragen(s: Settings, card: dict, cmp: dict, nu: dict, lor: dict) -> list[str]:
    """Eine Zeile je Kernfrage, aus denselben Rechnungen wie die Experimente (live, keine festen Zahlen)."""
    seite = "zweiseitig" if s.two else "einseitig"
    lines = [f"**1. Was verspricht die EWMA-Karte, und stimmt es?** Ziel ARL₀ = {s.arl0}: exakt {de(card['arl0'])} Beobachtungen bis zum Fehlalarm ({seite}, λ = {de(s.lam)}, L = {de(card['L'], 4)}), "
             f"Gitter {de(card['grid0'])}; bei δ = {de(s.delta)} im zero-state {de(card['arl1'])} Beobachtungen bis zum Alarm."]
    if cmp["design"] == "fest":
        sel = cmp["selected"]
        lines.append(f"**2. Welche Karte ist bei gleicher ARL₀ schneller, und in welchem Maß?** Einseitig, bei δ = {de(s.delta)}, EWMA λ = {de(s.lam)} gegen CUSUM k = {de(s.k)}: zero-state {de(sel['zero'], 4)} ({klass(sel['zero'])}), "
                     f"steady-state {de(sel['steady'], 4)} ({klass(sel['steady'])}); Verhältnis unter 1 heißt EWMA schneller.")
    else:
        b1 = cmp["best1"]
        lines.append(f"**2. Welche Karte ist bei gleicher ARL₀ schneller, und in welchem Maß?** Beide auf den Sprung abgestimmt (Orakel), bei δ = 1: zero-state {de(b1['zero']['verhaeltnis'], 4)} ({klass(b1['zero']['verhaeltnis'])}), "
                     f"steady-state {de(b1['steady']['verhaeltnis'], 4)} ({klass(b1['steady']['verhaeltnis'])}); Verhältnis unter 1 heißt EWMA schneller.")
    sel = nu[s.vorsprung == "mit"]
    lines.append(f"**3. Hängt die Verzögerung davon ab, wann die Änderung kommt?** Bei δ = {de(s.delta)}: EWMA {de(sel['ewma']['D'][0], 4)} bei ν = 0 und {de(sel['ewma']['steady'], 4)} im Grenzwert, "
                 f"CUSUM {de(sel['cusum']['D'][0], 4)} und {de(sel['cusum']['steady'], 4)} ({'mit' if s.vorsprung == 'mit' else 'ohne'} Anfangsvorsprung).")
    if lor["two"]:
        lines.append(f"**4. Gibt es ein schlechtestes Vorleben?** Zweiseitig: Lorden-Maß {de(lor['lorden'])} gegen zero-state {de(lor['zero'])} (Supremum bei u = {de(lor['u_sup'] / lor['H'], 2)} H).")
    else:
        at = lor["at"]
        wachsend = at[30] > at[10] + 1e-9
        lines.append(f"**4. Gibt es ein schlechtestes Vorleben?** Einseitig: ab Startwert 0 sind es {de(at[0])}, ab −5 σ_Z {de(at[5])}, ab −30 σ_Z {de(at[30])} Beobachtungen"
                     + ("; im gerechneten Bereich ohne Grenze." if wachsend else "; die Tiefe des Vorlebens ändert nichts (kein Gedächtnis)."))
    return lines
