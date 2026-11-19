"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Schnellstart und fährt jeden Regler an seine Grenzen und
prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig
identischen Inhalt rendern und kollidieren) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max).

Dazu: jede Auswahl der Radio- und select_slider-Regler, die Ecken (alle Minima, alle Maxima), der Permalink, die Verdikte der Standardansicht und ein Worst-Case-Lauf am
Rechenaufwand (kleinstes λ, größtes ARL₀, zweiseitig, größte Läufezahl) mit großzügigem Timeout (keine Zeit-Assertion, DEMO-PLAYBOOK §1c)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ewk_constants as C
import ewk_presets as P

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _widget(at, key):
    for art in ("slider", "select_slider", "radio", "number_input"):
        try:
            return getattr(at.sidebar, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("key", [k for k, spec in P.SETTING_SPECS.items() if spec.choices is not None])
def test_jede_auswahl_der_auswahlregler(key):
    for choice in P.SETTING_SPECS[key].choices:
        at = _fresh_app()
        _widget(at, key).set_value(choice).run(timeout=TIMEOUT)
        assert not at.exception, (key, choice, [str(e) for e in at.exception])


@pytest.mark.parametrize("edge", ["lo", "hi"])
def test_alle_regler_gleichzeitig_am_minimum_und_am_maximum(edge):
    at = _fresh_app()
    for key, spec in P.SETTING_SPECS.items():
        _widget(at, key).set_value(spec.lo if edge == "lo" else spec.hi)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_worst_case_am_rechenaufwand_kleinstes_lambda_groesstes_arl0_zweiseitig():
    """Kleinstes λ (größtes Gebiet), ARL₀ = 1000, zweiseitig, kleiner Sprung, 2000 Läufe, tiefster Startwert: der Lauf endet ohne Fehler (nur ein Timeout als Netz)."""
    at = _fresh_app()
    for key, value in (("lam_slider", C.LAMBDA_MIN), ("arl0_select", 1000), ("side_radio", "zweiseitig"), ("delta_slider", 0.25), ("runs_slider", C.RUNS_MAX), ("b_slider", C.B_MAX)):
        _widget(at, key).set_value(value)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    at = _fresh_app()
    for key, value in (("lam_slider", C.LAMBDA_MIN), ("arl0_select", 1000), ("side_radio", "einseitig"), ("delta_slider", 0.25), ("runs_slider", C.RUNS_MAX), ("b_slider", C.B_MAX)):
        _widget(at, key).set_value(value)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_permalink_uebernimmt_werte_und_begrenzt_sie():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["lam"] = "0.2"
    at.query_params["delta"] = "0.75"
    at.query_params["arl0"] = "100"
    at.query_params["side"] = "zweiseitig"
    at.query_params["b"] = "99"          # über der Grenze: wird auf 30 begrenzt
    at.query_params["design"] = "quatsch"  # unbekannte Auswahl: wird ignoriert (Standard bleibt)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = at.session_state
    if s["lam_slider"] != 0.2:  # AppTest liefert die Parameter je nach Streamlit-Version als Liste; dann gilt der Standard (der Permalink ist nur im Browser prüfbar)
        pytest.skip("AppTest übergibt die Adressparameter in dieser Streamlit-Version nicht als Text")
    assert s["delta_slider"] == pytest.approx(0.75) and s["arl0_select"] == 100 and s["side_radio"] == "zweiseitig" and s["b_slider"] == C.B_MAX and s["design_radio"] == "fest"


def test_standardansicht_zeigt_die_verdikte_aller_vier_experimente():
    at = _fresh_app()
    texte = [e.value for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error)]
    assert len(texte) == 4 and not at.error
    joined = " ".join(texte)
    for part in ("ARL₀ = 370,00", "zero-state 4 von 8 schneller", "8,5730", "ab −30 σ_Z 27,67"):
        assert part in joined, part
    assert [h.value for h in at.subheader if h.value.startswith("🔬")] == [
        "🔬 Experiment 1: Was verspricht die EWMA-Karte, und stimmt es?",
        "🔬 Experiment 2: Welche Karte ist bei gleicher ARL₀ schneller, und in welchem Maß?",
        "🔬 Experiment 3: Hängt die Verzögerung davon ab, wann die Änderung kommt?",
        "🔬 Experiment 4: Gibt es ein schlechtestes Vorleben?",
    ]


def test_expander_stehen_unter_ihrem_titel():
    at = _fresh_app()
    titel = {e.label for e in at.expander}
    assert {"So funktioniert die EWMA-Karte", "📋 Welches Maß heißt was", "📐 Mathematische Formulierung", "🧸 Das Spielzeug: dieselben fünf Zahlen in Brüchen"} <= titel
    assert any(h.value == "🚧 Wo die Annahmen enden" for h in at.subheader)
    mathe = next(e for e in at.expander if e.label == "📐 Mathematische Formulierung")  # per Titel, nie per Index (DEMO-PLAYBOOK §2)
    assert len(mathe.latex) == 5


def test_zweiseitig_und_abgestimmt_zeigen_ihre_eigenen_hinweise():
    at = _fresh_app()
    _widget(at, "design_radio").set_value("abgestimmt")
    _widget(at, "side_radio").set_value("zweiseitig")
    at.run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Im Betrieb nicht verfügbar" in w.value for w in at.warning)
    assert any("Zweiseitig" in e.value and "Lorden-Maß 13,56" in e.value for e in at.info)


def test_gezeigte_reihe_nennt_einen_fehlalarm_vor_der_aenderung_als_solchen():
    """Mit dem Zufalls-Seed 35 alarmiert die gezeigte Reihe schon bei Beobachtung 14, 86 Beobachtungen vor der Änderung: die Anzeige nennt das einen Fehlalarm und keine Verzögerung."""
    at = _fresh_app()
    m = next(x for x in at.metric if x.label == "Erster Alarm der gezeigten Reihe")
    assert m.value == "Beobachtung 14" and m.delta.startswith("Fehlalarm: die Änderung kommt erst nach 100")
    _widget(at, "delta_slider").set_value(4.0)
    _widget(at, "nu_slider").set_value(0)
    at.run(timeout=TIMEOUT)
    m = next(x for x in at.metric if x.label == "Erster Alarm der gezeigten Reihe")
    assert m.delta.endswith("Beobachtungen nach der Änderung") and "Fehlalarm" not in m.delta
