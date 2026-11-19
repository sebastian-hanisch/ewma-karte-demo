"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenbudget (Messung: README, Abschnitt Rechenzeiten): Die Knotenzahl der EWMA-Kette ist höchstens 600 (Löser O(n^3), n = 131 bei lambda = 0,1 einseitig),
die Folge D_nu läuft bis NU_MAX = 1500 Schritte (O(NU_MAX n^2)); die Untergrenze lambda >= 0,03 gilt in allen Experimenten, darunter wird das Gebiet zu groß."""

# --- Karte ------------------------------------------------------------------------------------------------------------------------------
LAMBDA_MIN, LAMBDA_MAX, DEFAULT_LAMBDA, LAMBDA_STEP = 0.03, 1.0, 0.1, 0.01
SIDES = ("einseitig", "zweiseitig")
DEFAULT_SIDE = "einseitig"
ARL0_CHOICES = (100, 370, 1000)  # gesetzte Ziele (370, nicht 370,398 der 3-Sigma-Karte)
DEFAULT_ARL0 = 370
K_MIN, K_MAX, DEFAULT_K, K_STEP = 0.125, 2.0, 0.5, 0.125
# --- Sprung und Reihe -------------------------------------------------------------------------------------------------------------------
DELTA_MIN, DELTA_MAX, DEFAULT_DELTA, DELTA_STEP = 0.0, 4.0, 1.0, 0.05
NU_MIN, NU_MAX_SLIDER, DEFAULT_NU, NU_STEP = 0, 200, 100, 10      # Beobachtungen VOR der Änderung (Experiment 1)
NOBS_MIN, NOBS_MAX, DEFAULT_NOBS, NOBS_STEP = 250, 600, 300, 50   # Länge der gezeigten Reihe; NU_MAX_SLIDER < NOBS_MIN: die Änderung liegt immer in der Reihe
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS, RUNS_STEP = 200, 2000, 1000, 100
SEED_MAX, DEFAULT_SEED = 999999, 35
# --- Vergleich, Vorleben ----------------------------------------------------------------------------------------------------------------
DESIGNS = ("fest", "abgestimmt")
DEFAULT_DESIGN = "fest"
START = ("ohne", "mit")  # Anfangsvorsprung H/2 bzw. h/2 (Auslegung, keine Zähleinheit)
DEFAULT_START = "ohne"
B_MIN, B_MAX, DEFAULT_B = 0, 30, 0  # Startwert -b sigma_Z der einseitigen EWMA (Experiment 4)

# --- Raster der gefrorenen Rechnungen und der Vergleiche --------------------------------------------------------------------------------
LAMBDA_GRID = (0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0)  # Schwellen in ewk_tuned.json
LAMBDA_BEST = (0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)             # „bestes lambda“ (wie V1 3e / V12 C2)
K_GRID = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)                  # „bestes k“; Schwellen in ewk_tuned.json
DELTA_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)           # die acht Sprünge der Tabellen (Plan 03, V12 C)
DELTA_CURVE = tuple(round(0.1 + 0.05 * i, 2) for i in range(79))  # 0,1 bis 4 in Schritten von 0,05 für Kurven und Kreuzungsstellen
MEASURES = ("zero", "pollak", "steady", "cyc")
MEASURE_NAMES = {"zero": "zero-state", "pollak": "Pollak", "steady": "steady-state (quasistationär)", "cyc": "zyklisch (Shiryaev)", "lorden": "Lorden"}
TIE = 0.003  # |Verhältnis - 1| <= TIE heißt Gleichstand (die zwei Gleichstandszellen der Linienplanung liegen bei 0,0002 bis 0,0003)
SERIES_TMAX = 1500  # Horizont der Lauflängenverteilung
SIM_SHARE_T = 100   # „Fehlalarm vor Beobachtung 100“
SIM_MEAN_LIMIT = 150.0  # die Simulation des Mittels wird nur angeboten, wenn die exakte Verzögerung höchstens so groß ist (Rechenzeit je Lauf)

# --- Farben -----------------------------------------------------------------------------------------------------------------------------
COLOR_EWMA = "#2F6B65"
COLOR_CUSUM = "#c0763a"
COLOR_GREY = "rgba(150,150,150,0.8)"
COLOR_ALARM = "#c0392b"
MEASURE_COLORS = {"zero": "#4c78a8", "pollak": "#f58518", "steady": "#2F6B65", "cyc": "#b279a2", "lorden": "#c0392b"}

PRESETS = {
    "EWMA λ = 0,1 gegen CUSUM (δ = 1)": {},
    "EWMA λ = 0,2 gegen CUSUM (δ = 1)": {"lam": 0.2},
    "Gleichstand im steady-state (λ = 0,2, δ = 0,75)": {"lam": 0.2, "delta": 0.75},
    "Kleiner Sprung δ = 0,5": {"delta": 0.5},
    "ARL₀ = 100": {"arl0": 100},
    "Mit Vorsprung": {"vorsprung": "mit"},
    "Lorden: Startwert −5 σ_Z": {"b": 5},
    "λ = 1 ist Shewhart": {"lam": 1.0, "side": "zweiseitig"},
}
PRESET_HELP = {
    "EWMA λ = 0,1 gegen CUSUM (δ = 1)": "Voreinstellung: einseitig, ARL₀ = 370, CUSUM k = 0,5, Sprung δ = 1. Hier kehrt sich die Reihenfolge zwischen zero-state und steady-state um.",
    "EWMA λ = 0,2 gegen CUSUM (δ = 1)": "Mit λ = 0,2 ist die EWMA im zero-state-Maß bei allen acht Sprüngen schneller als der CUSUM mit k = 0,5, im steady-state-Maß ab δ = 1 nicht mehr.",
    "Gleichstand im steady-state (λ = 0,2, δ = 0,75)": "Hier liegt das steady-state-Verhältnis bei 1,0002 (quasistationär) und 0,9997 (zyklisch): die Reihenfolge ist offen, im zero-state-Maß ist die EWMA vorn.",
    "Kleiner Sprung δ = 0,5": "Bei einem kleinen Sprung ist die EWMA mit λ = 0,1 in allen vier Maßen schneller als der CUSUM mit k = 0,5.",
    "ARL₀ = 100": "Kürzere Zeit bis zum Fehlalarm: der Unterschied zwischen zero-state und steady-state ist größer als bei ARL₀ = 370.",
    "Mit Vorsprung": "Anfangsvorsprung H/2 bzw. h/2 (Schwelle neu auf ARL₀ geeicht): er senkt die zero-state-Verzögerung und erhöht die steady-state-Verzögerung.",
    "Lorden: Startwert −5 σ_Z": "Die einseitige EWMA startet tief im Negativen: die Verzögerung steigt von 8,30 auf 15,37 Beobachtungen (λ = 0,1, δ = 1).",
    "λ = 1 ist Shewhart": "Mit λ = 1 hat die Karte kein Gedächtnis: alle Maße sind gleich, die ARL ist die geschlossene Form 1/p der Shewhart-Karte.",
}
