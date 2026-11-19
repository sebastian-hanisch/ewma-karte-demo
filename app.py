"""EWMA-Karte: wer schneller ist, hängt am Maß - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Teil der Reihe „Änderungserkennung“: Vergleich der EWMA-Karte mit dem CUSUM bei gleicher mittlerer Zeit bis zum Fehlalarm (ARL₀) in fünf Maßen der Verzögerung.
Aufbau nach DEMO-PLAYBOOK §2: Schnellstart in der Seitenleiste, Kernfragen vor den Experimenten, Verdikte aus reinen Funktionen (ewk_evaluation.py).

Lauffähig mit: streamlit run app.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # kleine dichte Löser: BLAS-Threads machen sie 100-mal langsamer

import streamlit as st  # noqa: E402

import ewk_constants as C  # noqa: E402
import ewk_evaluation as ev  # noqa: E402
import ewk_reference as R  # noqa: E402
import ewk_texts as TX  # noqa: E402
from ewk_layout import dunkles_design, kopf_css  # noqa: E402
from ewk_presets import (  # noqa: E402
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ewk_visualization import build_lorden_one, build_lorden_two, build_nu, build_ratio, build_runlength, build_series  # noqa: E402

st.set_page_config(page_title="EWMA-Karte – Sebastian Hanisch", layout="wide")


# --- Rechnungen (deterministisch, zwischengespeichert; der Schlüssel enthält alle Eingaben inklusive Zufalls-Seed) ----------------------------
@st.cache_data(show_spinner="Rechne die Karte exakt (Integralgleichung und Gitter) …")
def _card(lam, two, arl0, delta):
    return ev.exp_card(lam, two, arl0, delta)


@st.cache_data(show_spinner=False)
def _series(lam, two, arl0, delta, nu, nobs, seed):
    return ev.exp_series(lam, two, arl0, delta, nu, nobs, seed)


@st.cache_data(show_spinner="Wiederholungssimulation, dauert bis zu etwa 2 Sekunden …")
def _sim(lam, two, arl0, delta, runs, seed, arl1):
    return ev.exp_sim(lam, two, arl0, delta, runs, seed, arl1)


@st.cache_data(show_spinner="Vergleiche beide Karten über alle Sprünge, dauert etwa 1 Sekunde …")
def _compare(lam, k, arl0, delta):
    return ev.exp_compare(lam, k, arl0, delta)


@st.cache_data(show_spinner=False)
def _compare_tuned(arl0, delta):
    return ev.exp_compare_tuned(arl0, delta)


@st.cache_data(show_spinner="Rechne die Folge D_ν beider Karten …")
def _nu(lam, k, arl0, delta):
    return ev.exp_nu(lam, k, arl0, delta)


@st.cache_data(show_spinner="Rechne die Restlauflänge ab jedem Startwert …")
def _lorden(lam, two, arl0, delta, b):
    return ev.exp_lorden(lam, two, arl0, delta, b)


def _table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


# --- Kopf, Einleitung -------------------------------------------------------------------------------------------------------------------------
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 EWMA-Karte: wer schneller ist, hängt am Maß")
st.markdown(TX.INTRO)

with st.expander("So funktioniert die EWMA-Karte", expanded=True):
    st.markdown(TX.HOW_IT_WORKS)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, der aktive Schnellstart ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Karte**")
    lam = st.slider("Glättung λ der EWMA", *bounds("lam_slider"), step=C.LAMBDA_STEP, key="lam_slider", format="%.2f",
                    help="Je kleiner λ, desto länger das Gedächtnis der EWMA; λ = 1 ist die Shewhart-Karte. Gilt in allen Experimenten (Untergrenze 0,03).")
    side = st.radio("Seite der Karte", C.SIDES, key="side_radio", horizontal=True,
                    help="Einseitig = Alarm nur bei Anstieg. Wirkt in Experiment 1 und 4; die Experimente 2 und 3 sind einseitig.")
    arl0 = st.select_slider("Ziel ARL₀ (Beobachtungen bis zum Fehlalarm)", options=C.ARL0_CHOICES, key="arl0_select",
                            help="Gesetztes Ziel (370, nicht 370,398 der 3σ-Karte): die App rechnet die Grenze L bzw. die Schwelle h dazu.")
    k = st.slider("k des CUSUM (Experimente 2 und 3)", *bounds("k_slider"), step=C.K_STEP, key="k_slider", format="%.3f",
                  help="Referenzwert des einseitigen CUSUM; k = 0,5 ist auf den Sprung δ = 1 abgestimmt (k = δ/2).")
    st.markdown("**Sprung und Reihe**")
    delta = st.slider("Sprung δ (in Einheiten von σ)", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider", format="%.2f",
                      help="Sprung des Mittelwerts; δ = 0 heißt keine Änderung.")
    nu = st.slider("Änderung nach ν Beobachtungen (Experiment 1)", *bounds("nu_slider"), step=C.NU_STEP, key="nu_slider",
                   help="Zahl der Beobachtungen vor der Änderung in der gezeigten Reihe.")
    nobs = st.slider("Länge der Reihe (Experiment 1)", *bounds("nobs_slider"), step=C.NOBS_STEP, key="nobs_slider")
    seed = st.number_input("Zufalls-Seed der Reihe und der Simulation", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Reihe würfeln", width="stretch", on_click=randomize_seed)
    runs = st.slider("Simulationsläufe R (Experiment 1)", *bounds("runs_slider"), step=C.RUNS_STEP, key="runs_slider",
                     help="Gegenprobe der exakten Zahlen; mehr Läufe brauchen mehr Zeit (bei 2000 Läufen bis etwa 2 Sekunden).")
    st.markdown("**Vergleich und Vorleben**")
    design = st.radio("Auslegung (Experiment 2)", C.DESIGNS, key="design_radio",
                      format_func=lambda x: {"fest": "fest (ein λ, ein k für alle Sprünge)", "abgestimmt": "auf den Sprung abgestimmt (Orakel)"}[x],
                      help="Abgestimmt: das beste λ und das beste k je Sprung, im Betrieb unbekannt (eingefrorene Tabelle); dann zeigt Experiment 2 nicht das λ und k der Seitenleiste.")
    vorsprung = st.radio("Anfangsvorsprung (Experiment 3)", C.START, key="start_radio", horizontal=True,
                         format_func=lambda x: {"ohne": "ohne", "mit": "mit H/2 bzw. h/2"}[x],
                         help="Start bei H/2 (EWMA) bzw. h/2 (CUSUM), Schwelle neu auf die ARL₀ geeicht. Eine Auslegung, keine Zähleinheit.")
    b = st.slider("Tiefe des Vorlebens b (Experiment 4)", *bounds("b_slider"), step=1, key="b_slider",
                  help="Startwert −b·σ_Z der EWMA. Zweiseitig wird der Startwert in der Kurve markiert, solange er innerhalb der Grenzen liegt.")

sync_query_params({
    "lam_slider": lam, "side_radio": side, "arl0_select": int(arl0), "k_slider": k, "delta_slider": delta, "nu_slider": int(nu), "nobs_slider": int(nobs),
    "seed_input": int(seed), "runs_slider": int(runs), "design_radio": design, "start_radio": vorsprung, "b_slider": int(b),
})

s = ev.Settings(float(lam), side, int(arl0), float(k), float(delta), int(nu), int(nobs), int(seed), int(runs), design, vorsprung, int(b))

# --- Rechnungen ---------------------------------------------------------------------------------------------------------------------------
card = _card(s.lam, s.two, s.arl0, s.delta)
sim = _sim(s.lam, s.two, s.arl0, s.delta, s.runs, s.seed, card["arl1"])
series = _series(s.lam, s.two, s.arl0, s.delta, s.nu, s.nobs, s.seed)
cmp = _compare(s.lam, s.k, s.arl0, s.delta) if s.design == "fest" else _compare_tuned(s.arl0, s.delta)
nu_res = _nu(s.lam, s.k, s.arl0, s.delta)
lor = _lorden(s.lam, s.two, s.arl0, s.delta, s.b)

# --- Ergebnis: die Karte laufen lassen (Experiment 1) -------------------------------------------------------------------------------------
st.markdown("## 🎯 Ergebnis: die Karte und ihr Versprechen")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ARL₀ exakt", ev.de(card["arl0"]), delta=f"Ziel {s.arl0}", delta_color="off")
m2.metric("Verzögerung (zero-state)", ev.de(card["arl1"]), delta=f"bei δ = {ev.de(s.delta)}", delta_color="off")
m3.metric(f"Fehlalarm bis Beobachtung {C.SIM_SHARE_T}", f"{ev.de(card['share100'] * 100, 1)} %", delta="ohne Änderung, exakt", delta_color="off")
if series.alarm is None:
    alarm_note = f"Änderung nach {s.nu} Beobachtungen"
elif series.alarm <= s.nu:
    alarm_note = f"Fehlalarm: die Änderung kommt erst nach {s.nu} Beobachtungen"
else:
    alarm_note = f"{series.alarm - s.nu} Beobachtungen nach der Änderung"
m4.metric("Erster Alarm der gezeigten Reihe", "kein Alarm" if series.alarm is None else f"Beobachtung {series.alarm}", delta=alarm_note, delta_color="off")
st.plotly_chart(build_series(series, card["H"], s.two, s.nu), width="stretch", key="series_chart")
st.caption(f"Eine ausgedachte Reihe mit Zufalls-Seed {s.seed}: x_t ~ N(0, 1) bis zur Beobachtung {s.nu}, danach N({ev.de(s.delta)}, 1). Grenze H = {ev.de(card['H'], 4)} "
           f"(= {ev.de(card['L'], 4)}·σ_Z mit σ_Z = {ev.de(card['sigma_z'], 4)}); gestrichelt rot. Ein Alarm vor der Änderung ist ein Fehlalarm.")

# --- Kernfragen ---------------------------------------------------------------------------------------------------------------------------
st.markdown("## ❓ Kernfragen")
for line in ev.kernfragen(s, card, cmp, nu_res, lor):
    st.markdown(line)

st.markdown("---")

# --- Experiment 1 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Was verspricht die EWMA-Karte, und stimmt es?")
st.caption("Wirksam: λ, Seite, Ziel ARL₀, Sprung δ, Zufalls-Seed, Läufe. Exakt (Integralgleichung) gegen Gitter (Zweitentscheider, bei mehr als 0,5 % Abweichung ein Fehler statt einer Zahl) gegen Simulation.")
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(build_runlength(card), width="stretch", key="runlength_chart")
with c2:
    _table([
        {"Größe": "ARL₀ (Integralgleichung)", "Wert": ev.de(card["arl0"], 3)},
        {"Größe": "ARL₀ (Gitter, 80 Zellen je Seite)", "Wert": ev.de(card["grid0"], 3)},
        {"Größe": "Verzögerung zero-state (Integralgleichung)", "Wert": ev.de(card["arl1"], 3)},
        {"Größe": "Verzögerung zero-state (Gitter)", "Wert": ev.de(card["grid1"], 3)},
        {"Größe": f"Fehlalarm bis {C.SIM_SHARE_T}, exakt", "Wert": f"{ev.de(card['share100'] * 100, 2)} %"},
        {"Größe": f"Fehlalarm bis {C.SIM_SHARE_T}, simuliert ({sim['runs']} Läufe)", "Wert": f"{ev.de(sim['share'] * 100, 2)} % ± {ev.de(sim['share_se'] * 100, 2)}"},
        {"Größe": f"Verzögerung simuliert ({sim['runs']} Läufe)", "Wert": "nur bei exakter Verzögerung bis 150" if sim["mean"] is None else f"{ev.de(sim['mean'], 2)} ± {ev.de(sim['mean_se'], 2)}"},
    ])
kind, text = ev.verdict_card(card, sim, s.delta)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Welche Karte ist bei gleicher ARL₀ schneller, und in welchem Maß?")
st.caption("Einseitig, Zähleinheit Beobachtungen bis zum Alarm. Wirksam: Ziel ARL₀, Auslegung; bei „fest“ auch λ, k und der markierte Sprung δ. Verhältnis unter 1 heißt: die EWMA ist schneller.")
if s.design == "abgestimmt":
    st.warning("Im Betrieb nicht verfügbar: „Bestes λ gegen bestes k“ stimmt beide Karten auf den wahren Sprung ab, den man nicht kennt. Gezeigt sind die Sprünge der Tabelle (λ aus dem Gitter "
               f"{', '.join(ev.de(x, 2) for x in C.LAMBDA_BEST)}, k aus {', '.join(ev.de(x, 3) for x in C.K_GRID)}).")
st.plotly_chart(build_ratio(cmp, s.delta), width="stretch", key="ratio_chart")
rows = []
for i, d in enumerate(cmp["table_deltas"]):
    row = {"δ": ev.de(float(d), 2)}
    for m in C.MEASURES:
        row[C.MEASURE_NAMES[m]] = ev.de(float(cmp["table_ratio"][m][i]), 4)
    rows.append(row)
st.markdown("**Verhältnis EWMA / CUSUM** (unter 1: EWMA schneller; |Verhältnis − 1| ≤ 0,003 gilt als Gleichstand)")
_table(rows)
with st.expander("Verzögerungen in Beobachtungen (EWMA / CUSUM)"):
    rows = []
    for i, d in enumerate(cmp["table_deltas"]):
        row = {"δ": ev.de(float(d), 2)}
        for m in C.MEASURES:
            row[C.MEASURE_NAMES[m]] = f"{ev.de(float(cmp['table_ewma'][m][i]), 4)} / {ev.de(float(cmp['table_cusum'][m][i]), 4)}"
        rows.append(row)
    _table(rows)
    if s.design == "abgestimmt":
        st.markdown("**Abgestimmte Auslegung je Sprung (steady-state):** bestes λ der EWMA / bestes k des CUSUM")
        _table([{"δ": ev.de(float(d), 2), "bestes λ": ev.de(cmp["best_lambda"]["steady"][i], 2), "bestes k": ev.de(cmp["best_k"]["steady"][i], 3)} for i, d in enumerate(cmp["table_deltas"])])
kind, text = ev.verdict_compare(cmp)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Hängt die Verzögerung davon ab, wann die Änderung kommt?")
st.caption("Einseitig. Wirksam: λ, k, Ziel ARL₀, Sprung δ, Anfangsvorsprung (Auslegung; die Tabelle zeigt immer mit gegen ohne). Gestrichelt: steady-state (quasistationär), gepunktet: zyklisch.")
fig_d, fig_s = build_nu(nu_res, s.vorsprung == "mit")
st.plotly_chart(fig_d, width="stretch", key="nu_chart")
st.plotly_chart(fig_s, width="stretch", key="nu_surv_chart")
rows = []
for m in (*C.MEASURES, "lorden"):
    row = {"Maß": C.MEASURE_NAMES[m]}
    for key, name in (("cusum", "CUSUM"), ("ewma", "EWMA")):
        a0, a1 = nu_res[False][key][m], nu_res[True][key][m]
        if a0 is None:
            row[f"{name} ohne / mit Vorsprung"] = "nicht begrenzt (einseitig)"
            row[f"{name}: mit gegen ohne"] = "-"
        else:
            row[f"{name} ohne / mit Vorsprung"] = f"{ev.de(a0, 4)} / {ev.de(a1, 4)}"
            row[f"{name}: mit gegen ohne"] = ev.pct(a1 / a0)
    rows.append(row)
_table(rows)
kind, text = ev.verdict_nu(nu_res, s.vorsprung == "mit")
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Gibt es ein schlechtestes Vorleben?")
st.caption("Wirksam: λ, Seite, Ziel ARL₀, Sprung δ, Tiefe b. Der zweiseitige CUSUM (k = 0,5) ist vorgerechnet (2D-Kette, eingefroren); die Sprünge außerhalb des Rasters rechnet die App live in etwa 1 Sekunde.")
if s.two:
    st.plotly_chart(build_lorden_two(lor), width="stretch", key="lorden_chart")
    rows = [
        {"Maß": "zero-state", "EWMA": ev.de(lor["zero"], 4), "CUSUM (zweiseitig, k = 0,5)": "-" if lor["cusum2"] is None else ev.de(lor["cusum2"][1], 4)},
        {"Maß": "Pollak", "EWMA": ev.de(lor["pollak"], 4), "CUSUM (zweiseitig, k = 0,5)": "-"},
        {"Maß": "steady-state (quasistationär)", "EWMA": ev.de(lor["steady"], 4), "CUSUM (zweiseitig, k = 0,5)": "-"},
        {"Maß": "zyklisch (Shiryaev)", "EWMA": ev.de(lor["cyc"], 4), "CUSUM (zweiseitig, k = 0,5)": "-"},
        {"Maß": "Lorden (Supremum über alle Startwerte)", "EWMA": ev.de(lor["lorden"], 4), "CUSUM (zweiseitig, k = 0,5)": "-" if lor["cusum2"] is None else ev.de(lor["cusum2"][1], 4)},
        {"Maß": "Randwert L(−H) (nur bei großem Sprung gleich dem Lorden-Wert)", "EWMA": ev.de(lor["edge"], 4), "CUSUM (zweiseitig, k = 0,5)": "-"},
    ]
    _table(rows)
    if not lor["start_inside"]:
        st.caption(f"Der gewählte Startwert −{s.b} σ_Z liegt außerhalb der Grenzen (H = {ev.de(lor['L'], 2)} σ_Z) und ist in der Kurve nicht markiert.")
else:
    st.plotly_chart(build_lorden_one(lor), width="stretch", key="lorden_chart")
    _table([{"Tiefe b": str(x), "Startwert": f"−{x} σ_Z", "Beobachtungen bis zum Alarm": ev.de(lor["at"][x], 2)} for x in (0, 5, 10, 30)])
kind, text = ev.verdict_lorden(lor)
getattr(st, kind)(text)

st.markdown("---")

# --- Maßtabelle, Spielzeug, Grenzen ------------------------------------------------------------------------------------------------------
with st.expander("📋 Welches Maß heißt was"):
    st.markdown(TX.MEASURE_TABLE)

with st.expander("🧸 Das Spielzeug: dieselben fünf Zahlen in Brüchen"):
    toy = R.toy_measures()
    st.markdown(
        "Ein **ausgedachtes** Modell mit zwei Zuständen: EWMA mit λ = 1/2, Beobachtungen +1 mit Wahrscheinlichkeit q, sonst −1, Alarm bei Z ≥ 1/2. Vor der Änderung q = 1/2, danach q = 3/4. "
        f"Dann ist die ARL₀ gleich {toy['arl0']} (Start P) und die Verzögerung: {ev.toy_text()}. Reihenfolge: zero-state < zyklisch < steady-state < Pollak = Lorden. "
        "Richtung und Größe der Abstände gelten in diesem Modell, nicht für die Normal-EWMA (dort liegt der zero-state-Wert über dem steady-state-Wert: 8,2999 gegen 8,2540 bei λ = 0,1, δ = 1, ARL₀ = 370)."
    )

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.ASSUMPTIONS_TABLE)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(TX.MATH_HELP)
    st.markdown(TX.MATH_BODY)
    for f in TX.MATH_FORMULAS:
        st.latex(f)
    st.markdown(TX.MATH_AFTER)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
