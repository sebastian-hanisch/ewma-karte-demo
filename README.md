# EWMA-Karte: wer schneller ist, hängt am Maß – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-ewma-karte-demo.streamlit.app/)**

Stück 3 der **Änderungserkennung** („Wie schnell merkt man, dass sich etwas geändert hat?“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Vorgänger ist der CUSUM gegen die Karte ohne Gedächtnis (Stück 2), Nachfolger das Schätzen der Parameter (Stück 4). Die Kante ist ein **Fix**: Stück 2 endet mit der Annahme, der Vergleich zweier Karten bei
gleicher ARL₀ sei **eine Zahl je Sprung δ**; diese Seite hebt sie auf: Die Verzögerung hängt am **Maß**, und die Rangfolge zweier Karten kehrt sich dabei um. Vehikel: ausgedachte Normalreihen mit bekanntem Sprung.
Die Reihen sind ausgedacht (Normalwerte mit bekanntem Sprung); in den gelesenen Quellen gibt es keine reale Reihe, an der dieser Vergleich gemessen wäre. Gezählt werden Beobachtungen bis zum Alarm, im Modell mit
bekanntem σ und unabhängigen Werten. Was die Zahlen für einen echten Betrieb heißen, wird hier nicht behauptet; ein Nutzen einer Alarmregel in Logistik, Lager oder Fertigung ist nicht belegt.

## Warum dieses Problem

**„Welche Karte ist besser?“ hat ohne Maß, Auslegung und Zähleinheit keine Antwort.** Bei gleicher ARL₀ = 370 (gesetzt, nicht 370,398 der 3σ-Karte; einseitig, σ bekannt, unabhängig, normal) ist die Verzögerung der
EWMA mit λ = 0,2 im zero-state- und im Pollak-Maß bei allen acht geprüften Sprüngen δ = 0,25 bis 4 um 4,6 bis 8,7 % kürzer als die des CUSUM mit k = 0,5 (in Prozent der CUSUM-Verzögerung); im steady-state-Maß
(quasistationär und zyklisch) gilt das nur bis δ = 0,5 (Verhältnis 0,9256 und 0,9629), bei δ = 0,75 ist es gleichauf, und ab δ = 1 ist die Verzögerung der EWMA um 1,6 bis 3,0 % länger (zyklisch 1,6 bis 2,9 %, quasistationär 1,7 bis 3,0 %). Mit λ = 0,1 kehrt sich die
Reihenfolge im zero-state-Maß zwischen δ = 1,10 und 1,15 um (bei 1,15 liegt das Verhältnis mit 1,0003 schon im Gleichstandsband), im steady-state-Maß schon zwischen 0,85 und 0,90. Im Lorden-Maß wächst die Verzögerung der einseitigen EWMA mit der Tiefe des Vorlebens im gerechneten
Bereich ohne Grenze (δ = 1, λ = 0,1: ab Startwert 0 sind es 8,30 Beobachtungen, ab −5 σ_Z 15,37, ab −10 σ_Z 19,40, ab −30 σ_Z 27,67; eine Grenzaussage ist nicht bewiesen). Die zweiseitige EWMA hat im Lorden-Maß 13,56
gegen 9,92 des zweiseitigen CUSUM; der schlechteste Startwert liegt im Inneren bei etwa −0,91 H, am Rand −H sind es 13,33, im zero-state-Maß dagegen 9,74 gegen 9,92. „EWMA ist besser bei kleinen Sprüngen“ ist deshalb
ohne Maß, Auslegung und Zähleinheit kein Satz. Die Falle ist, einen Vergleich „bei gleicher ARL₀“ für vollständig zu halten: er legt das Maß der Verzögerung nicht fest, und „bestes λ gegen bestes k“ stimmt beide Karten
auf den wahren Sprung ab, den man im Betrieb nicht kennt. Gilt für unabhängige Normalwerte mit bekanntem σ, feste EWMA-Grenzen und die Zähleinheit Beobachtungen bis zum Alarm; die Zahlen sind exakt gerechnet, das heißt numerisch ohne Zufall
(Integralgleichung, Gitter als Zweitentscheider), die Simulation ist nur Gegenprobe.

Die Kernfragen der App: (1) Was verspricht die EWMA-Karte, und stimmt es? (2) Welche Karte ist bei gleicher ARL₀ schneller, und in welchem Maß? (3) Hängt die Verzögerung davon ab, wann die Änderung kommt?
(4) Gibt es ein schlechtestes Vorleben?

## Befunde und Korrekturen gegenüber dem Plan und der Vormessung

Zahlen aus Plan und Vormessung sind Erwartungen; hier wurde neu gemessen (eigene Rechnung, Zufallsquelle SplitMix64, zwei Zufalls-Seeds für die Simulation). Die exakten Zahlen der Vormessung (Tabellen 1 bis 8 und 10 unten, die Wahrscheinlichkeiten ohne Alarm bis ν) wurden bestätigt;
verschoben hat sich Folgendes (jede Zeile ist ein Test in `tests/test_claims.py`):

| Plan oder Vormessung | Neu gemessen | Grund |
|---|---|---|
| Lorden einseitig, ab −10 σ_Z: 19,41 | **19,40** (19,4050 in vier Stellen) | Der Plan rundete die vierstellige Zahl noch einmal; der Wert ist 19,404956 |
| Zweiseitiger CUSUM, zero-state = Lorden: 35,2534 / 9,9246 / 3,8578 | **35,2537 / 9,9247 / 3,8578** (δ = 0,5 / 1 / 2) | h = 4,773829 aus der Nullstellensuche statt des gerundeten 4,7738; Richardson aus 60 und 100 Zellen je Seite. Alle Verhältnisse gleich auf vier Stellen |
| D_ν des CUSUM „ab ν ≈ 25“ gleich dem Grenzwert | auf vier Stellen gleich ab **ν = 22**; D(25) = 7,9038 gilt weiter | Die Plan-Zahl war eine Tabellenstelle, kein Einschwingpunkt |
| Maß als Regler in Experiment 2 | alle vier Maße zugleich | Eine Auswahl hätte drei Maße verdeckt; die Hauptlinie (steady-state quasistationär) ist dicker gezeichnet |
| „Anteil Lauf mit Alarm vor Beobachtung 100“ (Shewhart: 23,71 %) | EWMA λ = 0,1 einseitig: **22,92 %**, λ = 0,2: 23,25 %, zweiseitig λ = 0,1: 22,58 % | Jede Karte hat ihren eigenen Anteil; 23,71 % gehört der Shewhart-Karte (λ = 1 zweiseitig) |
| Pollak-Argmax bei δ = 1,5 bis 3 | **kein endliches ν**: das Supremum ist der Grenzwert ν → ∞ | D_ν erreicht ihn von unten; die App meldet „Grenzwert“ statt eines Rausch-Argmax |
| Simulationen der Vormessung (numpy) | eigener Generator (SplitMix64): andere Zahlen, alle innerhalb von 4 Standardfehlern | Die Demo misst neu; Bänder statt Gleichheit |
| Zufallsquelle `normal()` für alles | `normal()` nur für die gezeigte Reihe, die Simulation mit `random()` und der Umkehrfunktion der Normalverteilung | Rechenzeit (rund 100 µs gegen rund 1 µs je Zahl) |
| Port 10033 (Plan) | 10044 (Auftrag) | Der Plan nannte einen freien Port; vergeben wurde später ein anderer |
| Gitter, 20 Zellen: 362,6496 | **362,6495** (362,64955 in sechs Stellen) | Der Plan rundete zweimal; der Wert liegt auf der Rundungsgrenze |
| λ von 0,03 bis 1 (Plan) | Regler in Schritten von 0,01; gefroren sind die Schwellen nur für neun λ | Andere λ rechnet die App live (Nullstellensuche, rund 0,05 s) |

Die Kante wurde gegen Plan 02, 04 und 09 gelesen: Plan 02 überlässt steady-state und Anfangsvorsprung ausdrücklich diesem Stück und rechnet den CUSUM nur im zero-state (dort Lorden = zero-state); die Zahl 26,746
der Shewhart-Karte bei δ = 1 stimmt (hier als λ = 1 einseitig: 26,7460). Plan 04 und 09 zitieren dieses Stück nur als Vorgänger beziehungsweise Quelle der Zeilen. Es blieb bei `fix`.

## Modell

Beobachtungen x_t ~ N(δ, 1), Ziel 0, σ = 1 bekannt, vor der Änderung δ = 0. **EWMA:** Z_t = λ·x_t + (1 − λ)·Z_(t−1), Z₀ = 0, feste Grenze H = L·σ_Z mit σ_Z = √(λ/(2 − λ)); Alarm bei Z_t > H (einseitig)
bzw. |Z_t| > H (zweiseitig). **CUSUM:** S_t = max(0, S_(t−1) + x_t − k), Alarm bei S_t ≥ h. **Lauflänge** = Beobachtungen bis zum Alarm (die Alarmbeobachtung zählt mit); **Fehlalarm** = Alarm vor der Änderung, je Zeit
gezählt (ARL₀). Das ARL₀-Ziel ist überall 370 (gesetzt): einseitig gehört dazu h = 4,0954 (CUSUM k = 0,5), zweiseitig h = 4,7738; „h = 4,77“ ohne „zweiseitig“ steht nirgends. Formeln, Symboltabelle und Maße stehen im
📐-Expander der App; die Tabelle „Welches Maß heißt was“ in einem eigenen Expander.

**Die fünf Zahlen.** ν = Zahl der Beobachtungen vor der Änderung (Moustakides/Bartroff; bei Lorden 1971 ist ν die erste Beobachtung danach, Unterschied 1). D_ν = E_ν[T − ν | T > ν]. **zero-state** = D₀;
**Pollak** = sup_ν D_ν; **steady-state** (ohne Zusatz immer **quasistationär**) = lim D_ν; **zyklisch** = Σ_k E_k[(T − k)⁺] / E_∞[T] (das **Shiryaev-Maß**); **Lorden** = Supremum auch über alle Vorgeschichten.
Der **Anfangsvorsprung** ist eine Auslegung (Startwert ungleich 0), weder Zähleinheit noch Maß. Zahlen, die ein anderes Stück der Linie besitzt (Gate, Phase I), stehen hier nicht.

## Methodik

* **Exakt:** Integralgleichung für die Restlauflänge L(u), Nyström-Verfahren mit Gauss-Legendre-Knoten (EWMA bis 600 Gauss-Knoten plus Startknoten, CUSUM 60 Gauss-Knoten plus Zustand 0). Die Maße kommen aus einer Rechnung: D_ν aus der Zustandsverteilung der Kette
  vor der Änderung (Folge bis ν = 1500), steady-state aus ihrem Grenzwert, zyklisch aus den Besuchszahlen, Lorden zweiseitig als Supremum von L(u) (Raster mit 4001 Punkten plus Verfeinerung).
* **Zweitentscheider in der App:** Gitter nach Brook/Evans mit 80 Zellen je Seite; weicht es um mehr als 0,5 % von der Integralgleichung ab, zeigt die App einen Fehler statt einer Zahl.
* **Abstimmung auf ARL₀:** Nullstellensuche der Grenze L bzw. der Schwelle h; für das Raster der Voreinstellungen gefroren (`ewk_tuned.json`), sonst live. Der zweiseitige CUSUM (2D-Kette) ist vorgerechnet (`ewk_cusum2.json`).
* **Simulation** nur als Gegenprobe: eigener Generator SplitMix64 (die ganzen Zufallszahlen und die Normalzahlen der gezeigten Reihe sind auf jeder Plattform gleich; die Normalzahlen der Simulation kommen aus der Umkehrfunktion der Normalverteilung des Systems und gelten nur als Band), Läufe bis zu 2000, Band aus dem Standardfehler.
* **Auslegung im Vergleich:** „fest“ (ein λ, ein k für alle Sprünge, live) oder „abgestimmt“ (das beste λ der Auswahl 0,03 / 0,05 / 0,1 / 0,2 / 0,3 / 0,5 / 0,8 gegen das beste k der Auswahl 0,125 bis 2, je Sprung und Maß; im Betrieb unbekannt, gefroren in
  `ewk_abgestimmt.json`).

## Befunde (gemessen, keine Behauptungen)

Alle Tabellen gelten für unabhängige N(δ, 1)-Beobachtungen mit bekanntem σ, feste EWMA-Grenzen, ARL₀ = 370 (gesetzt), einseitig, wo nicht „zweiseitig“ steht; Zähleinheit Beobachtungen bis zum Alarm; exakt gerechnet.
Jede Tabelle steht Zahl für Zahl in `tests/test_claims.py` (sie wird aus der Rechnung erzeugt und mit der README verglichen); die Spalte der unabhängigen Prüfung nennt den Orakel-Test.

**1. Verhältnis EWMA / CUSUM (k = 0,5), je Maß; unter 1 heißt: die Verzögerung der EWMA ist kürzer** (feste Auslegung λ = 0,1 und 0,2). Test `test_claim_tabelle_vergleich`; Orakel: Gitter, Eigenvektor, Reihe,
Simulation (`tests/test_oracle_ewma_karte.py`).

| δ | λ = 0,1: zero-state | λ = 0,1: Pollak | λ = 0,1: steady-state | λ = 0,1: zyklisch | λ = 0,2: zero-state | λ = 0,2: Pollak | λ = 0,2: steady-state | λ = 0,2: zyklisch |
|---|---|---|---|---|---|---|---|---|
| 0,25 | 0,7207 | 0,7207 | 0,7266 | 0,7264 | 0,9132 | 0,9132 | 0,9256 | 0,9254 |
| 0,50 | 0,7857 | 0,7857 | 0,8120 | 0,8115 | 0,9329 | 0,9329 | 0,9629 | 0,9626 |
| 0,75 | 0,8916 | 0,8916 | 0,9449 | 0,9440 | 0,9510 | 0,9510 | 1,0002 | 0,9997 |
| 1,00 | 0,9681 | 0,9681 | 1,0443 | 1,0432 | 0,9536 | 0,9536 | 1,0170 | 1,0164 |
| 1,50 | 1,0505 | 1,0528 | 1,1529 | 1,1514 | 0,9486 | 0,9486 | 1,0264 | 1,0256 |
| 2,00 | 1,0906 | 1,0983 | 1,2035 | 1,2018 | 0,9474 | 0,9474 | 1,0297 | 1,0290 |
| 3,00 | 1,1153 | 1,1319 | 1,2362 | 1,2345 | 0,9532 | 0,9532 | 1,0291 | 1,0284 |
| 4,00 | 1,1440 | 1,1440 | 1,2429 | 1,2414 | 0,9411 | 0,9411 | 1,0299 | 1,0291 |

Gleichstandszellen mit |Verhältnis − 1| ≤ 0,003: λ = 0,2, δ = 0,75 (1,0002 quasistationär, 0,9997 zyklisch); beide Definitionen des steady-state geben sonst dasselbe Urteil.

**2. Kreuzungsstellen** (Gitter δ = 0,1 bis 4 in Schritten von 0,05; kleinstes δ, ab dem die EWMA nicht mehr schneller ist). Test `test_claim_kreuzungsstellen`.

| EWMA λ | zero-state | Pollak | steady-state | zyklisch |
|---|---|---|---|---|
| 0,1 | ab δ = 1,15 | ab δ = 1,15 | ab δ = 0,90 | ab δ = 0,90 |
| 0,2 | auf dem ganzen Gitter schneller | auf dem ganzen Gitter schneller | ab δ = 0,75 | ab δ = 0,80 |

Zellen mit Verhältnis innerhalb von ±0,003 um 1 sind Gleichstand (λ = 0,1 zero-state: 1,0003 bei δ = 1,15; λ = 0,2 steady-state: 1,0002 bei δ = 0,75).

**3. „Bestes λ gegen bestes k“** (bestes λ der Auswahl gegen bestes k der Auswahl, beide auf den wahren Sprung abgestimmt, im Betrieb unbekannt). Test `test_claim_bestes_gegen_bestes`. Die feste Auslegung λ = 0,2 gegen k = 0,5 gewinnt den zero-state-Vergleich dagegen bis δ = 4:
**Deutung (nicht gemessen):** k = 0,5 ist auf δ = 1 abgestimmt (k = δ/2) und bei großen δ schwach; der Vorsprung der EWMA dort spiegelt die Auslegung, keinen Vorteil der EWMA als Karte.

| δ | zero-state | Pollak | steady-state | zyklisch | bestes λ / k (zero-state) |
|---|---|---|---|---|---|
| 0,5 | 0,8824 | 0,8866 | 1,0029 | 0,9971 | 0,05 / 0,250 |
| 1,0 | 0,9536 | 0,9536 | 1,0170 | 1,0164 | 0,20 / 0,500 |
| 2,0 | 1,0196 | 1,0196 | 1,0415 | 1,0414 | 0,50 / 1,000 |
| 3,0 | 1,0201 | 1,0224 | 1,0312 | 1,0311 | 0,80 / 1,500 |

Die Zeile δ = 3 der Vormessung V1 3e (1,0721 zero-state) rechnete mit einem kleineren λ-Gitter (bestes λ 0,5); mit dem Gitter bis 0,8 gilt 1,0201 (so hier).

**4. Abhängigkeit von ARL₀** (λ = 0,1 gegen k = 0,5, δ = 1). Test `test_claim_abhaengigkeit_von_arl0`. Der Abstand zwischen zero-state- und steady-state-Verhältnis schrumpft mit ARL₀ (0,140 / 0,076 / 0,058), die Richtung der Umkehr bleibt.

| ARL₀ | zero-state | Pollak | steady-state | zyklisch |
|---|---|---|---|---|
| 100 | 0,9260 | 0,9738 | 1,0662 | 1,0564 |
| 370 | 0,9681 | 0,9681 | 1,0443 | 1,0432 |
| 1000 | 0,9792 | 0,9792 | 1,0367 | 1,0364 |

**5. D_ν hängt an ν** (δ = 1, ARL₀ = 370). Test `test_claim_d_nu`. Wahrscheinlichkeit, bis ν keinen Alarm zu haben, für CUSUM / EWMA: 0,7696 / 0,7708 (ν = 100), 0,5852 / 0,5858 (ν = 200), 0,2573 / 0,2571 (ν = 500). Beim CUSUM fällt D_ν monoton auf den steady-state-Wert; bei der EWMA ist D_ν nicht monoton (Mulde mit dem Minimum 8,1598 bei ν = 7).

| ν | CUSUM k = 0,5 | EWMA λ = 0,1 |
|---|---|---|
| 0 | 8,5730 | 8,2999 |
| 1 | 8,3068 | 8,2580 |
| 2 | 8,1597 | 8,2235 |
| 5 | 7,9730 | 8,1666 |
| 7 | 7,9330 | 8,1598 |
| 10 | 7,9118 | 8,1702 |
| 25 | 7,9038 | 8,2351 |
| 50 | 7,9038 | 8,2530 |
| 100 | 7,9038 | 8,2540 |
| 200 | 7,9038 | 8,2540 |
| Grenzwert ν → ∞ | 7,9038 | 8,2540 |

Für δ = 1,5 bis 3 liegt das Pollak-Supremum der EWMA λ = 0,1 nicht bei einem endlichen ν, sondern ist der Grenzwert ν → ∞ (Test `test_claim_pollak_grenzwert`):

| δ | zero-state | Pollak (= Grenzwert ν → ∞) | steady-state |
|---|---|---|---|
| 1,5 | 5,0871 | 5,0982 | 5,0982 |
| 2,0 | 3,7146 | 3,7408 | 3,7408 |
| 3,0 | 2,4880 | 2,5250 | 2,5250 |

**6. Der Vorsprung** (Start H/2 bzw. h/2, Schwelle neu auf ARL₀ = 370 geeicht; CUSUM h = 4,1488, EWMA λ = 0,1 H = 0,5586; Verzögerung mit ÷ ohne Vorsprung im selben Maß). Test `test_claim_vorsprung`.
Er gewinnt nur im zero-state-Maß (um 23 bis 39 %); im Pollak-Maß gewinnt ihn der CUSUM, die EWMA nicht; im steady-state-Maß, zyklisch und im Lorden-Maß kostet er 0,4 bis 2,7 %. Nur H/2 bzw. h/2 ist gemessen.

| δ | Maß | CUSUM: mit ÷ ohne Vorsprung | EWMA λ = 0,1: mit ÷ ohne Vorsprung |
|---|---|---|---|
| 0,5 | zero-state | 0,7742 | 0,7566 |
| 0,5 | Pollak | 0,9716 | 1,0104 |
| 0,5 | steady-state (quasistationär) | 1,0210 | 1,0274 |
| 0,5 | zyklisch (Shiryaev) | 1,0188 | 1,0216 |
| 0,5 | Lorden | 1,0204 | unbegrenzt (einseitig) |
| 1,0 | zero-state | 0,6357 | 0,6663 |
| 1,0 | Pollak | 0,9338 | 1,0109 |
| 1,0 | steady-state (quasistationär) | 1,0129 | 1,0165 |
| 1,0 | zyklisch (Shiryaev) | 1,0097 | 1,0090 |
| 1,0 | Lorden | 1,0124 | unbegrenzt (einseitig) |
| 2,0 | zero-state | 0,6066 | 0,6345 |
| 2,0 | Pollak | 0,9225 | 1,0118 |
| 2,0 | steady-state (quasistationär) | 1,0109 | 1,0118 |
| 2,0 | zyklisch (Shiryaev) | 1,0075 | 1,0036 |
| 2,0 | Lorden | 1,0104 | unbegrenzt (einseitig) |

**7. Lorden-Maß, einseitige EWMA** (λ = 0,1, δ = 1, ARL₀ = 370): erwartete Beobachtungen bis zum Alarm ab Startwert −b·σ_Z, streng wachsend in b über alle 61 Punkte von b = 0 bis 30, mit abnehmendem Zuwachs (b = 10 bis 20: +5,0; b = 20 bis 30: +3,3);
eine Grenzaussage ist nicht bewiesen. Test `test_claim_lorden_einseitig`; Orakel: Simulation ab dem Startwert, Verdopplung der Knotenzahl (`tests/test_oracle_lorden.py`).

| Startwert | Beobachtungen bis zum Alarm (λ = 0,1, δ = 1, ARL₀ = 370) |
|---|---|
| 0 | 8,2999 |
| −1 σ_Z | 10,1849 |
| −2 σ_Z | 11,7647 |
| −5 σ_Z | 15,3727 |
| −10 σ_Z | 19,4050 |
| −20 σ_Z | 24,4065 |
| −30 σ_Z | 27,6671 |

**8. Lorden-Maß, zweiseitig** (λ = 0,1, ARL₀ = 370): das Supremum von L(u) über alle Startwerte liegt bei kleinem Sprung im Inneren, nicht am Rand −H; bei δ = 2 fällt es mit dem Randwert zusammen (der Randwert ist nur bei großem Sprung gleich dem Lorden-Wert). Zweiseitiger CUSUM (k = 0,5,
h = 4,7738): schlechtester Zustand ist die Zelle (0, 0), Lorden gleich zero-state. Test `test_claim_lorden_zweiseitig`; Orakel: eigene Rastersuche, Gitter mit 161 Zellen, Simulation am Ort des Supremums, alle 1600 Zellen des CUSUM.

| δ | EWMA zero-state | EWMA Lorden (Supremum) | Ort des Supremums | Randwert L(−H) | CUSUM (zero-state = Lorden) | Verhältnis zero-state | Verhältnis Lorden |
|---|---|---|---|---|---|---|---|
| 0,5 | 28,2172 | 33,3386 | −0,749 H | 29,3223 | 35,2537 | 0,8004 | 0,9457 |
| 1,0 | 9,7354 | 13,5565 | −0,909 H | 13,3296 | 9,9247 | 0,9809 | 1,3659 |
| 2,0 | 4,1803 | 6,6860 | −1,000 H | 6,6860 | 3,8578 | 1,0836 | 1,7331 |

Preis der Zählweise „jede Meldung zählt“: Startet die zweiseitige EWMA bei −0,97 H, enden 4,18 % der Läufe (δ = 1) mit einem Alarm nach unten, bei −H sind es 5,85 % (exakt; Simulation mit 40 000 Läufen im Band). Das Lorden-Maß „nur Alarm in die
richtige Richtung“ ist nicht gemessen.

**9. Spielzeug in Brüchen** (EWMA mit λ = 1/2, Beobachtungen ±1, Alarm bei Z ≥ 1/2; ausgedachtes Modell). Vor der Änderung q = 1/2: ARL₀ = 4 (Start P), schlechtester Zustand 6; danach q = 3/4: zero-state 16/9, zyklisch 22/9,
steady-state (quasistationär) 2,6018 (goldener Schnitt im Gewichtsverhältnis), Pollak = Lorden = 28/9, in dieser Reihenfolge. P(Lauflänge = t) = Fib(t − 2)/2^t für t ≥ 3. Test `test_claim_spielzeug` und `tests/test_handbeispiele.py` (Aufzählung aller 2^t Folgen).
Richtung und Größe der Abstände gelten dort, nicht für die Normal-EWMA (dort liegt der zero-state-Wert über dem steady-state-Wert: 8,2999 gegen 8,2540).

**10. Schwellen für ARL₀ = 100 / 370 / 1000.** EWMA einseitig: λ = 0,1 gibt L = 1,7379 / 2,4026 / 2,8080 (bei 370: H = 0,5512), λ = 0,2 gibt L = 2,5976 bei 370, zweiseitig λ = 0,1 gibt L = 2,7010 (H = 0,6197) bei 370; CUSUM k = 0,5
einseitig h = 2,8494 / 4,0954 / 5,0707, zweiseitig h = 4,7738 bei 370. Test `test_claim_schwellen`. **λ = 1 ist die Shewhart-Karte:** ARL₀ = 370,40 / 43,89 / 6,30 bei δ = 0 / 1 / 2 (L = 3 zweiseitig), gleich 1/p auf 6 Stellen; mit dem
Ziel 370 gehört L = 3,00 (2,9997) zweiseitig. Test `test_claim_orakel_gitter_und_shewhart`. Zweiseitig λ = 0,1, L = 2,7: ARL₀ = 368,99373398 (Integralgleichung) gegen das Gitter mit 20 / 40 / 80 / 160 Zellen 362,6495 / 367,3496 / 368,5763 / 368,8887
(Abstand fällt je Verdopplung um den Faktor 3,86 bis 3,97, Ordnung zwei).

**11. Simulation gegen exakt** (SplitMix64, 1000 Läufe, ARL₀ = 370 einseitig, δ = 1, Zufalls-Seed 35 und 36; Test `test_claim_simulation_baender`). Die Simulation ist eine Gegenprobe: alle |z| ≤ 4.

| Karte | exakt (zero-state) | Seed 35: simuliert (z) | Seed 36: simuliert (z) |
|---|---|---|---|
| EWMA λ = 0,1 einseitig | 8,2999 | 8,22 ± 0,13 (−0,6) | 8,48 ± 0,12 (+1,5) |
| EWMA λ = 0,2 einseitig | 8,1752 | 8,00 ± 0,15 (−1,2) | 8,44 ± 0,15 (+1,7) |
| EWMA λ = 0,1 zweiseitig | 9,7354 | 9,74 ± 0,15 (+0,0) | 9,88 ± 0,14 (+1,0) |

## Zahl → Test (Band oder exakt)

| Zahl in der README | Test | Art |
|---|---|---|
| Schwellen L, H, h (Befund 10) | `test_claim_schwellen` | exakt (rel. 1e-6) |
| Gitter gegen Integralgleichung, Shewhart 1/p | `test_claim_orakel_gitter_und_shewhart` | exakt bis auf Gitterfehler |
| Tabellen 1 bis 8 (alle Zellen) | `test_claim_tabelle_*`, `test_claim_kreuzungsstellen`, `test_claim_bestes_gegen_bestes`, `test_claim_abhaengigkeit_von_arl0`, `test_claim_d_nu`, `test_claim_pollak_grenzwert`, `test_claim_vorsprung`, `test_claim_lorden_einseitig`, `test_claim_lorden_zweiseitig` | exakt (Text gleich Rechnung, Zahlen gegen die Planwerte auf 4 Stellen) |
| Kernaussage „Warum dieses Problem“ | `test_claim_kernaussage` | exakt |
| Spielzeug in Brüchen | `test_claim_spielzeug` | Brüche exakt |
| Fehlalarm bis Beobachtung 100 (22,92 % usw.) | `test_claim_fehlalarm_anteil` | exakt; Simulation im Band |
| Simulation (Tabelle 11) | `test_claim_simulation_baender` | Band (z-Wert höchstens 4) |
| Abweichungen gegenüber dem Plan | `test_claim_abweichungen_vom_plan` | exakt |
| Rechenzeiten | nicht im Test (Messung dieser README) | Messung |
| Gefrorene Daten | `test_gefrorene_daten_gleich_neu_gerechnet` | rel. 1e-6 |

## Ehrliche Grenzen

Was die Demo nicht zeigt oder nicht kann; negative Befunde stehen gleichrangig. **Keine der beiden gemessenen Auslegungen gewinnt in allen Maßen und bei allen δ, und wer gewinnt, hängt an der Auslegung.** Die EWMA hat unter dem Lorden-Maß
keine Garantie (einseitig wächst die Verzögerung im gerechneten Bereich, b bis 30, streng und ohne erkennbare Schranke, zweiseitig 13,56 gegen 9,92 bei δ = 1). Das Lorden-Maß gewichtet seltene Vorleben nicht (Z unter −5 σ_Z: etwa 2,87·10⁻⁷ im Sollwert-Betrieb, Näherung); welches Maß zu
einem Betrieb passt, entscheidet keine Zahl dieser Seite.

**Annahmen und ihre Grenzen** (dieselbe Tabelle wie in der App, Abschnitt „🚧 Wo die Annahmen enden“):
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

🚧 **Bewusst nicht enthalten / nicht gemessen:**

* Zeitabhängige EWMA-Grenzen (Steiner 1999, nicht gelesen); alle Zahlen gelten für feste Grenzen H = L·σ_Z.
* Shiryaev–Roberts-Verfahren: das Shiryaev-Maß ist gerechnet, das Verfahren nicht.
* Lorden für b → ∞ (einseitig): nur das Wachstum bis b = 30; Lorden „nur Alarm in die richtige Richtung“ (zweiseitig) nicht; beim zweiseitigen CUSUM ist „schlechtester Zustand = (0, 0)“ für δ = 1 auf dem Gitter (m = 40, 1600 Zellen)
  geprüft, für alle δ nur als Kopplungsargument.
* Andere feste Auslegungen (λ = 0,05, 0,3; andere k als 0,5 in der festen Auslegung) und andere Vorsprünge als H/2 und h/2.
* Autokorrelation, geschätzte Parameter, Schiefe: nicht hier. Die Aussage „kleines λ braucht mehr Phase-I-Daten“ ist umstritten (Jones u. a. 2004 gegen Saleh u. a. 2016) und gehört nicht in diese Demo.
* Reale Daten: keine. Ein zweites Vehikel (Residuen des Erzeugers von `niveausprung-demo`) ist nicht gemessen.
* Plattform: die Zahlen sind unter Windows mit Python 3.14 gerechnet; die exakten Werte sind bis auf Rundung plattformfest, Simulationen gelten als Band. Die Linux-CI-Prüfung (Docker) ist vor dem ersten Push noch zu machen.
* Browser: die App lief headless im Browser-Fenster der Entwicklungsumgebung; alle acht Schnellstarts, die Umschalter „zweiseitig“, „abgestimmt“ und „mit Vorsprung“ und der Expander mit den Formeln liefen ohne Fehlermeldung und ohne KaTeX-Fehler (DOM geprüft). Ein Bildschirmfoto kam nicht zustande (Zeitüberschreitung), das Aussehen ist deshalb nicht gesehen.

Nachbarn: `niveausprung-demo` nutzt die EWMA nur als Nachführer des Niveaus in der Prognose (Kosten, Prognose-Reset, Bestätigungsregel) – hier die Maßfrage mit exaktem Orakel, dort Kosten; nicht vermischen
([README](https://github.com/sebastian-hanisch/niveausprung-demo)). Dieselbe Rekursion mit α statt λ steht in [`exponential-smoothing-demo`](https://github.com/sebastian-hanisch/exponential-smoothing-demo).
Die Technik der ersten Passage (eine kleine Matrix, (I − R)⁻¹·1) steht auch in [`markov-queue-demo`](https://github.com/sebastian-hanisch/markov-queue-demo) und [`konventionswechsel-demo`](https://github.com/sebastian-hanisch/konventionswechsel-demo);
ein Test mit festen Blicken und unbegrenzter Laufzeit der Gegenseite steht in [`sequential-test-demo`](https://github.com/sebastian-hanisch/sequential-test-demo).

## Rechenzeiten

Gemessen auf dem Entwicklungsrechner (Windows, Python 3.14, `OMP_NUM_THREADS=1`), ein Lauf, kein Mittel, Zeit nur als Hinweis; auf der Streamlit Cloud ist mit der zwei- bis dreifachen Zeit zu rechnen (nicht gemessen):

| Schritt | Standard (λ = 0,1, ARL₀ = 370) | schwerster Fall (λ = 0,03, ARL₀ = 1000, 2000 Läufe) |
|---|---|---|
| Experiment 1 exakt (Karte, Gitter, Lauflängenverteilung) | 0,05 s | 0,03 s |
| Experiment 1 Simulation (1000 / 2000 Läufe) | 0,10 s | 0,43 s |
| Experiment 2 fest (79 Sprünge, beide Karten, vier Maße) | 0,36 s | 0,70 s |
| Experiment 3 (ohne und mit Vorsprung) | 0,04 s | 0,08 s |
| Experiment 4 einseitig / zweiseitig | 0,03 s / 0,02 s | 0,05 s / 0,06 s |
| zweiseitiger CUSUM live (Sprung außerhalb des gefrorenen Rasters) | 1,2 s | – |
| Schwelle abstimmen (live, ein λ) | 0,05 s | 0,04 s |
| Import der Module | 0,8 s | – |

Der gesamte erste Lauf der App im AppTest dauert rund 2,7 s (mit Import von Streamlit und Plotly).

## Tests

Die Suite hat **177 Tests**; sie läuft seriell in rund 88 s (118 s bei Last durch parallele Sitzungen), mit `pytest -n 8` in rund 61 s (Windows, Python 3.14, je ein Lauf, Zeit nur als Hinweis; größter Einzelposten: das Neurechnen der gefrorenen Daten mit rund 10 s und die AppTest-Läufe mit
zusammen rund 30 s). Zufall nur mit festen Zufalls-Seeds, Gleitkomma mit Toleranz (relativ 1e-6 bis 1e-9), Simulationen als Band (z-Wert höchstens 4), Brüche und Ganzzahlen exakt.

| Datei | Tests | Inhalt |
|---|---|---|
| `tests/test_handbeispiele.py` | 13 | Handbeispiele der Planung: Spielzeug-EWMA in Brüchen (Aufzählung aller 2^t Folgen als Orakel), die fünf Zahlen im Spielzeug, Varianzformel, 5-Zellen-Gitter, λ = 1 |
| `tests/test_oracle_ewma_karte.py` | 18 | Orakel: Gitter gegen Integralgleichung (Ordnung zwei), alle vier Maße aus Eigenvektor, Besuchszahlen und Matrixfolge, Reihe für das zyklische Maß, 1/p für λ = 1, Simulation (zero-state, D_ν, zyklisch) |
| `tests/test_oracle_lorden.py` | 16 | Orakel zum Lorden-Maß: Supremum im Inneren (eigene Rastersuche, Gitter, Simulation), Alarmrichtung, CUSUM (schlechtester Zustand über alle Knoten und alle 1600 Zellen), einseitige Kurve |
| `tests/test_literatur_ewma.py` | 3 | Paare (λ, L) mit ARL₀ ≈ 500 als eigene Nachrechnung |
| `tests/test_eigenschaften_karten.py` | 12 | Eigenschaften über 10 bis 40 Zufallsinstanzen (fester Seed): Monotonie, Symmetrie, Skalierung, Pollak gegen eigene Folge, CUSUM-Gleichheit, Verfeinerung, Verteilung, Abstimmung |
| `tests/test_edge_cases.py` | 14 | Randfälle: λ = 1 und kleinstes λ, δ = 0, großer Sprung, ν = 0 und groß, unerreichbares ARL₀, ein Lauf, Startwert tief, Kreuzungsstelle, Live-Weg des 2D-CUSUM |
| `tests/test_verdicts.py` | 16 | Verdikte an gesetzten Werten: Gleichstand ±0,003, Gitterabweichung 0,5 %, z-Wert 4, jede Zahl im Text, jeder Zweig |
| `tests/test_presets.py` | 13 | Schnellstarts: Vollständigkeit, Raster, Standard gleich Standardeinstellung, je Schnellstart die erwartete Verdikt-Zeile (Gleichstand ohne Sieger) |
| `tests/test_claims.py` | 30 | jede Zahl der README (Tabellen Zeichen für Zahl), Textgleichheit App und README, gefrorene Daten gegen Neurechnung, Kopie des Zufallsgenerators |
| `tests/test_app.py` | 15 | AppTest: jeder Knopf, jeder Regler an Minimum und Maximum, jede Auswahl, alle Minima und Maxima zugleich, Worst-Case-Lauf, Permalink, Verdikte der Standardansicht, Fehlalarm der gezeigten Reihe |
| `tests/test_layout.py` | 24 | Schnellstart oben in der Seitenleiste, fester Kopf, Hervorhebung für jeden Regler |
| `tests/test_footer.py`, `tests/test_runtime_imports.py` | 3 | Footer (App und README), jede Bibliothek steht in `requirements.txt` |

Plattform: die Orakel- und Claim-Tests rechnen mit relativer Toleranz oder als Band; die Linux-CI (Docker) ist vor dem ersten Push noch zu prüfen.

## Quellen und Lesestufen

Alles hier ist Lehrbuchwissen; neu ist nur die Vorführbarkeit mit Zähleinheit und exakten Rechenwegen. **Gelesen (W, laut Quellenbericht der Linienplanung):** Moustakides, Polunchenko, Tartakovsky (2009), arXiv 0908.4119, und Bartroff,
Tartakovsky (2024), arXiv 2403.18782: Lorden-, Pollak- und Shiryaev-Maß, Konvention für ν (ihre Verzögerungsspalten sind um 0,27 bzw. 0,51 zu klein und werden nicht zitiert). Lucas, Saccucci (1990), Technometrics 32(1): im
Quellenbericht als Text mit Zeichenverlusten geführt (Scan; die Vormessung nennt sie „nicht lesbar“); es stehen nur Definitionen (zero-state, steady-state, Anhang A) und die Aussage zur Größe des Unterschieds, keine Tabellenzahl;
die Paare (λ, L) mit ARL₀ ≈ 500 sind eigene Nachrechnung (500 ± 0,5). **Nicht gelesen:** Roberts (1959), Page (1954), Lorden (1971, nur das Abstract), Pollak (1985), Moustakides (1986), Brook, Evans (1972), Steiner (1999). Alle Zahlen
dieser README sind eigene Nachrechnung in diesem Modell.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kopf, Ergebnis, Kernfragen, vier Experimente, Maßtabelle, Spielzeug, Grenzen, Formeln |
| `ewk_chains.py` | Ketten: Nyström-Integralgleichung (CUSUM, EWMA), Gitter nach Brook/Evans, 2D-Kette, Shewhart geschlossen |
| `ewk_measures.py` | die fünf Zahlen (Pre, Card), Lorden-Supremum, einseitige Kurve, Lauflängenverteilung, Alarmrichtung |
| `ewk_tuning.py` | Abstimmung auf ARL₀: gefrorene Werte (`ewk_tuned.json`) oder Nullstellensuche live |
| `ewk_evaluation.py` | die vier Experimente als Rechnung, Verdikte (reine Funktionen), Kurzantworten, Zahlenformat |
| `ewk_reference.py` | eingefrorene Daten lesen (`ewk_cusum2.json`, `ewk_abgestimmt.json`), Spielzeug in Brüchen |
| `ewk_scenario.py` | ausgedachte Reihe, Simulation mit SplitMix64 |
| `ewk_rng.py` | Zufallsgenerator SplitMix64 (Kopie der Vorlage aus dem Portfolio, plattformfeste Normalzahlen) |
| `ewk_constants.py` | Regler-Grenzen, Raster, Farben, Schnellstarts mit Hilfetexten |
| `ewk_presets.py` | Permalink (Begrenzen, Einrasten), Schnellstart (aktiver hervorgehoben), Zufalls-Seed-Button |
| `ewk_layout.py` | fester Kopf (Stylesheet) |
| `ewk_texts.py` | Texte der App (Einleitung, Erklärung, Maßtabelle, Grenzen, Formeln) |
| `ewk_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `ewk_tuned.json`, `ewk_cusum2.json`, `ewk_abgestimmt.json` | gefrorene Rechnungen mit Quelle und Stand je Abschnitt |
| `tools/ewk_make_tuned.py` | erzeugt die drei Dateien; `--check` rechnet neu und vergleicht auf 6 Stellen |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | die Tests des Abschnitts „Tests“; dazu `tests/claims_tables.py` (erzeugt die Tabellen der README) und `tests/conftest.py` |

## Bewusst nicht umgesetzt

* Kein PDF-Export (die Seite ist eine Rechenansicht, kein Bericht).
* Kein „Hotelling“ und keine MEWMA: Mehrkanalkarten sind ein eigenes Stück der Linie (die Bezeichnung „Hotelling“ steht im Portfolio auch für ein Standortmodell).
* Kein Gate-Fall (Lkw-Tor): er reißt an der Unabhängigkeit und gehört in ein anderes Stück der Linie.
* Kein Wegweiser „welche Karte wann“: gehört zur Verfahrenswahl am Ende der Linie.
* Keine Rückverweise in `niveausprung-demo` und in die Konzepte-Spezifikation; keine Website-Einbindung (Karte, Footer-Text, Erklärseiten) – das sind eigene Schritte mit eigener Freigabe.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Lokal lief die Demo mit Python `3.14`, die CI läuft mit `3.12`; die in der Streamlit Cloud eingestellte Python-Version ist beim Deploy hier nachzuliefern (offen).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gefrorene Daten neu rechnen und vergleichen: `python tools/ewk_make_tuned.py --check` (rund 75 s; mit `--schnell` ohne die 2D-Kette rund 6 s).

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html).
