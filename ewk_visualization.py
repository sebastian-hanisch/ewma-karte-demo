"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ewk_constants as C

_NU_TICKS = [0, 1, 2, 5, 10, 25, 50, 100, 200, 500, 1000]


def _lock(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=40, b=45), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="x unified")
    return fig


def build_series(series, thr: float, two: bool, nu: int) -> go.Figure:
    """Oben die Beobachtungen x_t (Änderung nach nu Beobachtungen), unten die EWMA-Statistik Z_t mit ihrer Grenze; der erste Alarm ist markiert."""
    t = np.arange(1, len(series.x) + 1)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.5, 0.5], subplot_titles=("Beobachtungen x_t", "EWMA-Statistik Z_t und Grenze"))
    fig.add_trace(go.Scatter(x=t, y=np.round(series.x, 4), mode="lines+markers", marker=dict(size=3), line=dict(width=1, color=C.COLOR_GREY), name="x_t"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.round(series.z, 5), mode="lines", line=dict(width=2, color=C.COLOR_EWMA), name="Z_t"), row=2, col=1)
    for y in ([thr, -thr] if two else [thr]):
        fig.add_hline(y=y, line=dict(color=C.COLOR_ALARM, dash="dash", width=1), row=2, col=1)
    fig.add_vline(x=nu + 0.5, line=dict(color="rgba(80,80,80,0.6)", dash="dot", width=1))
    if series.alarm is not None:
        fig.add_trace(go.Scatter(x=[series.alarm], y=[series.z[series.alarm - 1]], mode="markers", marker=dict(size=11, color=C.COLOR_ALARM, symbol="x"), name="erster Alarm"), row=2, col=1)
    fig.update_xaxes(title_text="Beobachtung t", row=2, col=1)
    return _lock(fig, 460)


def build_runlength(a: dict) -> go.Figure:
    """Wahrscheinlichkeit, bis zur Beobachtung t alarmiert zu haben (P(Lauflänge <= t)), ohne Änderung und mit dem Sprung; die Linie bei 100 markiert den Fehlalarm-Anteil."""
    t = np.arange(0, len(a["surv0"]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.round(1.0 - a["surv0"], 6), mode="lines", line=dict(color=C.COLOR_GREY, width=2), name="ohne Änderung (δ = 0)"))
    fig.add_trace(go.Scatter(x=t, y=np.round(1.0 - a["surv1"], 6), mode="lines", line=dict(color=C.COLOR_EWMA, width=2), name="mit Sprung (zero-state)"))
    fig.add_vline(x=C.SIM_SHARE_T, line=dict(color="rgba(80,80,80,0.6)", dash="dot", width=1))
    fig.update_xaxes(title_text="Beobachtung t (logarithmisch)", type="log", range=[0, np.log10(len(t) - 1)])
    fig.update_yaxes(title_text="P(Alarm bis t)", range=[0, 1.02])
    return _lock(fig, 340)


def build_ratio(a: dict, delta: float) -> go.Figure:
    """Verhältnis EWMA/CUSUM je Maß über den Sprung (unter 1 heißt EWMA schneller); fest: Gitter 0,1 bis 4, abgestimmt: die acht Sprünge der Tabelle."""
    fig = go.Figure()
    fest = a["design"] == "fest"
    x = a["deltas"] if fest else a["table_deltas"]
    for m in C.MEASURES:
        y = a["ratio"][m] if fest else a["table_ratio"][m]
        width = 3.5 if m == "steady" else 2
        fig.add_trace(go.Scatter(x=x, y=np.round(y, 5), mode="lines" if fest else "lines+markers", line=dict(color=C.MEASURE_COLORS[m], width=width,
                                 dash="dot" if m in ("pollak", "cyc") else "solid"), name=C.MEASURE_NAMES[m]))
    fig.add_hline(y=1.0, line=dict(color="rgba(80,80,80,0.8)", width=1))
    fig.add_vline(x=delta, line=dict(color="rgba(80,80,80,0.5)", dash="dot", width=1))
    fig.update_xaxes(title_text="Sprung δ", range=[0, 4.05])
    fig.update_yaxes(title_text="Verzögerung EWMA / CUSUM")
    return _lock(fig, 380)


def build_nu(a: dict, vorsprung: bool) -> tuple[go.Figure, go.Figure]:
    """D_nu beider Karten über nu (logarithmische Achse, x = nu + 1) mit Grenzwert und zyklischem Wert; darunter die Wahrscheinlichkeit, bis nu keinen Alarm zu haben."""
    sel = a[bool(vorsprung)]
    fig = go.Figure()
    fig2 = go.Figure()
    for key, name, color in (("cusum", f"CUSUM k = {a['k']:.3g}".replace(".", ","), C.COLOR_CUSUM), ("ewma", f"EWMA λ = {a['lam']:.3g}".replace(".", ","), C.COLOR_EWMA)):
        d = sel[key]
        nu = np.arange(len(d["D"]))
        fig.add_trace(go.Scatter(x=nu + 1, y=np.round(d["D"], 5), mode="lines", line=dict(color=color, width=2.5), name=f"D_ν {name}"))
        fig.add_hline(y=d["steady"], line=dict(color=color, dash="dash", width=1))
        fig.add_hline(y=d["cyc"], line=dict(color=color, dash="dot", width=1))
        fig2.add_trace(go.Scatter(x=nu + 1, y=np.round(d["surv"], 6), mode="lines", line=dict(color=color, width=2), name=name))
    ticks = [n for n in _NU_TICKS if n <= len(sel["ewma"]["D"]) - 1]
    for f in (fig, fig2):
        f.update_xaxes(type="log", tickvals=[n + 1 for n in ticks], ticktext=[str(n) for n in ticks], title_text="Beobachtungen vor der Änderung ν (logarithmisch)")
    fig.update_yaxes(title_text="Verzögerung D_ν (Beobachtungen)")
    fig2.update_yaxes(title_text="P(kein Alarm vor ν)", range=[0, 1.02])
    return _lock(fig, 380), _lock(fig2, 300)


def build_lorden_one(a: dict) -> go.Figure:
    """Einseitige EWMA: erwartete Beobachtungen bis zum Alarm ab Startwert −b σ_Z; der einseitige CUSUM hat als Lorden-Maß seinen zero-state-Wert."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a["curve_b"], y=np.round(a["curve_L"], 5), mode="lines", line=dict(color=C.COLOR_EWMA, width=3), name="EWMA ab Startwert −b σ_Z"))
    fig.add_trace(go.Scatter(x=[a["b"]], y=[a["selected"]], mode="markers", marker=dict(size=12, color=C.COLOR_ALARM), name="gewählt"))
    fig.add_hline(y=a["cusum_lorden"], line=dict(color=C.COLOR_CUSUM, dash="dash", width=2), annotation_text="CUSUM (Lorden = zero-state)", annotation_position="bottom right")
    fig.update_xaxes(title_text="Tiefe des Vorlebens b (Startwert −b σ_Z)")
    fig.update_yaxes(title_text="Beobachtungen bis zum Alarm")
    return _lock(fig, 360)


def build_lorden_two(a: dict) -> go.Figure:
    """Zweiseitige EWMA: Restlauflänge L(u) über den Startwert in Einheiten von H; das Supremum liegt im Inneren, nicht am Rand −H."""
    h = a["H"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.round(a["curve_u"] / h, 4), y=np.round(a["curve_L"], 5), mode="lines", line=dict(color=C.COLOR_EWMA, width=3), name="EWMA: L(u)"))
    fig.add_trace(go.Scatter(x=[a["u_sup"] / h], y=[a["lorden"]], mode="markers", marker=dict(size=12, color=C.COLOR_ALARM, symbol="diamond"), name="Supremum (Lorden-Maß)"))
    fig.add_trace(go.Scatter(x=[-1.0], y=[a["edge"]], mode="markers", marker=dict(size=10, color="rgba(80,80,80,0.9)", symbol="square"), name="Randwert L(−H)"))
    if a["start_inside"] and a["start_L"] is not None:
        fig.add_trace(go.Scatter(x=[a["start"] / h], y=[a["start_L"]], mode="markers", marker=dict(size=11, color="#4c78a8", symbol="circle-open", line=dict(width=3)), name="gewählter Startwert"))
    fig.add_hline(y=a["zero"], line=dict(color=C.COLOR_GREY, dash="dot", width=2), annotation_text="zero-state (Start 0)", annotation_position="bottom right")
    if a["cusum2"] is not None:
        fig.add_hline(y=a["cusum2"][1], line=dict(color=C.COLOR_CUSUM, dash="dash", width=2), annotation_text="zweiseitiger CUSUM (zero-state = Lorden)", annotation_position="top left")
    fig.update_xaxes(title_text="Startwert u in Einheiten von H", range=[-1.02, 1.02])
    fig.update_yaxes(title_text="Beobachtungen bis zum Alarm")
    return _lock(fig, 380)
