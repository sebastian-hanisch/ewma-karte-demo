"""Permalink (Begrenzen), Presets („Schnellstart“) und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Jeder Regler der Seitenleiste steht in SETTING_SPECS: so hebt jede Änderung die Markierung des aktiven Schnellstarts auf (tests/test_layout.py prüft das je Regler).
Ein Schnellstart legt nur die Regler fest, die er nennt; alle übrigen stehen danach auf ihrem Standardwert."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import ewk_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: object = None
    hi: object = None
    step: float | None = None
    choices: tuple | None = None   # Auswahlregler (Radio, select_slider): erlaubte Werte; lo und hi sind dann der erste und der letzte


def _num(url, caster, default, lo, hi, step=None):
    return SettingSpec(url, caster, default, lo, hi, step)


def _choice(url, caster, default, choices):
    return SettingSpec(url, caster, default, choices[0], choices[-1], None, tuple(choices))


SETTING_SPECS = {
    "lam_slider": _num("lam", float, C.DEFAULT_LAMBDA, C.LAMBDA_MIN, C.LAMBDA_MAX, C.LAMBDA_STEP),
    "side_radio": _choice("side", str, C.DEFAULT_SIDE, C.SIDES),
    "arl0_select": _choice("arl0", int, C.DEFAULT_ARL0, C.ARL0_CHOICES),
    "k_slider": _num("k", float, C.DEFAULT_K, C.K_MIN, C.K_MAX, C.K_STEP),
    "delta_slider": _num("delta", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX, C.DELTA_STEP),
    "nu_slider": _num("nu", int, C.DEFAULT_NU, C.NU_MIN, C.NU_MAX_SLIDER, C.NU_STEP),
    "nobs_slider": _num("n", int, C.DEFAULT_NOBS, C.NOBS_MIN, C.NOBS_MAX, C.NOBS_STEP),
    "seed_input": _num("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "runs_slider": _num("runs", int, C.DEFAULT_RUNS, C.RUNS_MIN, C.RUNS_MAX, C.RUNS_STEP),
    "design_radio": _choice("design", str, C.DEFAULT_DESIGN, C.DESIGNS),
    "start_radio": _choice("start", str, C.DEFAULT_START, C.START),
    "b_slider": _num("b", int, C.DEFAULT_B, C.B_MIN, C.B_MAX, 1),
}
# Schlüssel in C.PRESETS -> Schlüssel des Reglers
PRESET_KEYS = {
    "lam": "lam_slider", "side": "side_radio", "arl0": "arl0_select", "k": "k_slider", "delta": "delta_slider", "nu": "nu_slider", "nobs": "nobs_slider",
    "seed": "seed_input", "runs": "runs_slider", "design": "design_radio", "vorsprung": "start_radio", "b": "b_slider",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def _snap(value, spec: SettingSpec):
    """Wert aus der Adresszeile auf Grenzen und Reglerraster bringen (ein Wert abseits des Rasters würde der Regler anders darstellen, als die App rechnet)."""
    if spec.choices is not None:
        return value if value in spec.choices else None
    value = max(spec.lo, min(spec.hi, value))
    if spec.step:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = max(spec.lo, min(spec.hi, value))
        value = round(value, 6)
    return spec.caster(value)


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen und das Raster begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = _snap(spec.caster(qp[spec.url_param]), spec)
            except (ValueError, TypeError):
                continue
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt die im Schnellstart genannten Regler; alle anderen auf ihren Standardwert."""
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = preset.get(key, SETTING_SPECS[state_key].default)


def _same(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def aktives_preset():
    """Name des Schnellstarts, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Schnellstart ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(_same(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default)) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; der aktive Schnellstart ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
