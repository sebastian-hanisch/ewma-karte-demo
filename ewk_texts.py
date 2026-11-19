"""Texte der App (Einleitung, Erklärung, Maßtabelle, Grenzen, Formeln). Zwei Fassungen desselben Textes (App und README) werden von tests/test_claims.py verglichen,
nicht von Hand synchron gehalten: ASSUMPTIONS_TABLE steht in beiden."""

INTRO = """
**EWMA-Karte** und **CUSUM** sind zwei Karten mit Gedächtnis, die melden, dass sich der Mittelwert einer Messreihe geändert hat. Die Frage „welche Karte ist besser?“ wird oft mit einer Zahl
je Sprung beantwortet, bei gleicher mittlerer Zeit bis zum Fehlalarm (**ARL₀**). Diese Seite zeigt, dass die Antwort am **Maß der Verzögerung** hängt: Zwischen den Maßen **kehrt sich die
Reihenfolge der beiden Karten um**. Die Reihen sind ausgedacht (Normalwerte mit bekanntem Sprung); gezählt werden Beobachtungen bis zum Alarm, im Modell mit bekanntem σ und unabhängigen Werten.
Die Zahlen sind **exakt** gerechnet, das heißt numerisch ohne Zufall (Integralgleichung, ein Gitter als Zweitentscheider); die Simulation ist nur eine Gegenprobe.

Die Rekursion Z_t = λ·x_t + (1 − λ)·Z_(t−1) ist dieselbe wie bei der einfachen exponentiellen Glättung der Prognose-Reihe
([`exponential-smoothing-demo`](https://github.com/sebastian-hanisch/exponential-smoothing-demo), dort α statt λ); hier wird Z_t gegen eine Grenze gelesen, nicht als Prognose verwendet.
Was ein Alarm in einem echten Betrieb nützt, wird hier nicht behauptet.

Weiter unten: vier Experimente, der Abschnitt **🚧 Wo die Annahmen enden**, die Tabelle **Welches Maß heißt was**, ein Spielzeug in Brüchen und der Expander **📐 Mathematische Formulierung**.
"""

HOW_IT_WORKS = """
1. **Die Karte.** Die EWMA-Statistik ist Z_t = λ·x_t + (1 − λ)·Z_(t−1) mit Start Z₀ = 0 (Sollwert). Je kleiner λ, desto länger das Gedächtnis; **λ = 1 ist die Shewhart-Karte** ohne Gedächtnis.
   Alarm bei Z_t > H (einseitig, Anstieg) bzw. |Z_t| > H (zweiseitig), mit der festen Grenze H = L·σ_Z und σ_Z = √(λ/(2 − λ)).
2. **Die Abstimmung.** L wird so gewählt, dass die mittlere Zeit bis zum Fehlalarm genau das Ziel ARL₀ erreicht (hier 100, 370 oder 1000 Beobachtungen, **gesetzt**, nicht 370,398 der 3σ-Karte).
   Der CUSUM S_t = max(0, S_(t−1) + x_t − k) bekommt seine Schwelle h auf dieselbe Weise.
3. **Die Verzögerung.** Nach einem Sprung δ zählt man die Beobachtungen bis zum Alarm. Wann der Sprung kommt und in welchem Zustand die Karte dann steht, legt das **Maß** fest:
   zero-state, Pollak, steady-state, zyklisch (Shiryaev) oder Lorden. Das ist der Streit dieser Seite.
"""

MEASURE_TABLE = """
ν = Zahl der Beobachtungen **vor** der Änderung (letzte Beobachtung vor der Änderung); die Alarmbeobachtung zählt mit (T − ν ≥ 1). D_ν = E_ν[T − ν | T > ν] ist die erwartete Zahl
Beobachtungen bis zum Alarm, wenn die Änderung nach ν Beobachtungen kommt und bis dahin kein Alarm war.

| Name | Definition | Was es ist |
|---|---|---|
| **zero-state** | D₀: Änderung vor der ersten Beobachtung, Start im Sollwert | Zähleinheit der Startannahme |
| **Pollak** | Supremum über ν von D_ν | Maß |
| **steady-state** (ohne Zusatz immer **quasistationär**) | Grenzwert von D_ν für ν → ∞ | Zähleinheit der Startannahme |
| **zyklisch** (Shiryaev) | Σ_k E_k[(T − k)⁺] / E_∞[T]: die Karte startet nach jedem Alarm neu und läuft lange | Maß (Shiryaev) |
| **Lorden** | Supremum über ν **und über alle möglichen Vorgeschichten** (alle erreichbaren Zustände der Karte) | Maß |

Das quasistationäre Maß heißt nie Shiryaev-Maß; beide sind gerechnet und unterscheiden sich ohne Vorsprung in der Verzögerung in der Voreinstellung (λ = 0,1, k = 0,5, ARL₀ = 370) um weniger als 0,1 %; bei kleinem λ und kleiner ARL₀ um mehrere Prozent (bis 5,6 % bei λ = 0,03, ARL₀ = 100, δ = 2). Der **Anfangsvorsprung** (Start bei h/2 bzw. H/2) ist eine
**Auslegung** der Karte, weder Zähleinheit noch Maß: Er ändert den Startzustand, die Zähleinheit bleibt zero-state bzw. steady-state. Für den einseitigen CUSUM liegt das Pollak-Maximum bei ν = 0, und
Lorden ist gleich zero-state (der Zustand 0 ist der schlechteste); bei λ = 1 (Shewhart) sind alle fünf Zahlen gleich, weil die Karte kein Gedächtnis hat.
"""

ASSUMPTIONS_TABLE = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **σ und der Sollwert sind bekannt** | Alle Zahlen gelten für bekannte Parameter; bei aus Daten geschätzten Parametern ist die ARL₀ nicht mehr die gesetzte: nicht gemessen. | folgende Teile der Reihe (Parameter schätzen) |
| **Beobachtungen sind unabhängig** | Bei Autokorrelation ändern sich ARL₀ und ARL₁ der EWMA: nicht gemessen. | folgende Teile der Reihe (abhängige Werte) |
| **Normalverteilung** | Schiefe Verteilungen ändern die ARL₀ der EWMA: nicht gemessen. | - |
| **Feste Grenzen H = L·σ_Z** | Zeitabhängige Grenzen (Steiner 1999, nicht gelesen) sind nicht gerechnet. | - |
| **Ausgedachte Reihen** | Es gibt keine reale Reihe, an der dieser Vergleich gemessen wäre; ein Nutzen einer Alarmregel in Logistik, Lager oder Fertigung ist nicht belegt. | - |
| **Eine feste Auslegung** | Gemessen sind λ = 0,1 und 0,2 gegen k = 0,5 (fest) und das beste λ gegen das beste k (abgestimmt, im Betrieb unbekannt); andere feste Auslegungen (λ = 0,05, 0,3) sind nicht gemessen. | - |
| **Lorden-Maß der einseitigen EWMA** | Nur das Wachstum bis b = 30 ist gerechnet, eine Grenzaussage nicht; „nur Alarm in die richtige Richtung“ (zweiseitig) ist nicht gemessen. | - |
| **Anfangsvorsprung** | Nur H/2 bzw. h/2 ist gemessen. | - |
| **Verfahren und Maß** | Das Shiryaev-Maß ist gerechnet, das Shiryaev-Roberts-Verfahren nicht. | - |
"""

MATH_HELP = """
**Lesehilfe.** Gegeben: Beobachtungen x_t ~ N(δ, 1), ein Ziel ARL₀ und eine Karte (EWMA mit λ oder CUSUM mit k). Gesucht: die erwartete Zahl Beobachtungen bis zum Alarm nach einem Sprung δ, in fünf Maßen.

| Symbol | Bedeutung |
|---|---|
| x_t | Beobachtung t, vor der Änderung N(0, 1), danach N(δ, 1) (σ = 1 bekannt) |
| δ | Sprung des Mittelwerts in Einheiten von σ |
| λ, Z_t | Glättung der EWMA (0 < λ ≤ 1) und ihre Statistik, Z₀ = 0 |
| σ_Z, L, H | σ_Z = √(λ/(2 − λ)); Grenze H = L·σ_Z; L wird auf die ARL₀ abgestimmt |
| k, h, S_t | Referenzwert, Schwelle und Statistik des CUSUM, S₀ = 0 |
| T | Lauflänge: Beobachtungen bis zum Alarm (die Alarmbeobachtung zählt mit) |
| ν | Zahl der Beobachtungen vor der Änderung |
| D_ν | erwartete Zahl Beobachtungen T − ν bis zum Alarm, wenn die Änderung nach ν Beobachtungen kommt und bis dahin kein Alarm war |
| L(u) | erwartete Zahl Beobachtungen bis zum Alarm ab dem Zustand u (Restlauflänge) |
| b | Tiefe des Vorlebens: Startwert −b·σ_Z |
"""

MATH_BODY = r"""
**Die Karten und die Maße.** Das Modell ist vollständig angegeben (Karten, Alarmregel, Maße; einseitig, zweiseitig mit |Z_t| > H) mit 0 < λ ≤ 1, L > 0, k > 0, h > 0 und φ als Dichte der
Standardnormalverteilung; die numerische Lösung steht darunter nur beschrieben.
"""

MATH_FORMULAS = (
    r"Z_t = \lambda\,x_t + (1-\lambda)\,Z_{t-1}, \qquad Z_0 = 0, \qquad T = \min\{t \ge 1 : Z_t > H\}, \qquad H = L\,\sigma_Z, \quad \sigma_Z = \sqrt{\tfrac{\lambda}{2-\lambda}}",
    r"S_t = \max(0,\; S_{t-1} + x_t - k), \qquad S_0 = 0, \qquad T = \min\{t \ge 1 : S_t \ge h\}",
    r"L(u) = 1 + \int_{\ell}^{H} \frac{1}{\lambda}\,\varphi\!\left(\frac{y-(1-\lambda)u}{\lambda} - \delta\right) L(y)\,dy \qquad \text{(EWMA; } \ell = -H \text{ zweiseitig, einseitig so tief, dass darunter keine Masse bleibt)}",
    r"D_\nu = \mathrm{E}_\nu\!\left[\,T - \nu \mid T > \nu\,\right], \qquad \text{zero-state} = D_0, \quad \text{Pollak} = \sup_\nu D_\nu, \quad \text{steady-state} = \lim_{\nu\to\infty} D_\nu",
    r"\text{zyklisch (Shiryaev)} = \frac{\sum_{k \ge 0} \mathrm{E}_k\!\left[(T-k)^+\right]}{\mathrm{E}_\infty[T]}, \qquad \text{Lorden} = \sup_{\nu}\ \sup_{\text{Vorgeschichte}} \mathrm{E}\!\left[T - \nu \mid T > \nu,\ \text{Zustand}\right]",
)

MATH_AFTER = """
Die Integralgleichung wird nach Nyström mit Gauss-Legendre-Knoten gelöst (höchstens 600 Knoten; bei δ = 0 ist L(0) die ARL₀); als zweiter, unabhängiger Rechenweg dient das Gitter nach Brook/Evans
(Fehler ∝ 1/m²), bei λ = 1 die geschlossene Form 1/p der Shewhart-Karte. Der CUSUM läuft als Kette mit dem Zustand 0 und 60 Knoten; der zweiseitige CUSUM als 2D-Kette (Richardson aus 60 und 100 Zellen je Seite, eingefroren).
Die Folge D_ν läuft bis ν = 1500; das quasistationäre Maß ist die bedingte Verteilung am Ende der Folge, geprüft gegen den Eigenvektor.

**Literatur und Lesestufen.** Gelesen (im Quellenbericht der Linienplanung): Moustakides, Polunchenko, Tartakovsky (2009), arXiv 0908.4119, und Bartroff, Tartakovsky (2024), arXiv 2403.18782 (Lorden-, Pollak- und Shiryaev-Maß,
Konvention für ν); deren Verzögerungsspalten sind hier nicht übernommen. Lucas, Saccucci (1990), Technometrics 32(1): im Quellenbericht nur als Text mit Zeichenverlusten (Scan) geführt, aus ihr stehen hier nur Definitionen
(zero-state, steady-state, Anhang A), keine Tabellenzahl. **Nicht gelesen:** Roberts (1959), Page (1954), Lorden (1971, nur das Abstract), Pollak (1985), Moustakides (1986), Brook, Evans (1972), Steiner (1999).
Alles hier ist Lehrbuchwissen; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Rechenwegen.
"""
